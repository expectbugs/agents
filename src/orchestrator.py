#!/usr/bin/env python3
"""
Multi-Agent Orchestration System
Based on LangGraph for managing multiple AI agents
"""

from typing import Annotated, Dict, Any, Optional, List
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import operator
import redis.asyncio as aioredis
import json
import asyncio
from datetime import datetime
import logging
from abc import ABC, abstractmethod
try:
    from .config import Config
    from .logging_config import get_logger
except ImportError:
    from config import Config
    from logging_config import get_logger
import re
import html
import os

def task_queue_reducer(existing_tasks: List[Dict[str, Any]], new_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Custom task queue reducer that properly handles task updates and prevents duplicates.
    
    Args:
        existing_tasks: Current tasks in the queue
        new_tasks: New tasks to add or updates to existing tasks
        
    Returns:
        Updated task queue with no duplicates and proper status tracking
    """
    if not existing_tasks:
        existing_tasks = []
    if not new_tasks:
        return existing_tasks
    
    # Create a map of existing tasks by ID for fast lookup
    task_map = {task.get("id", f"task_{i}"): task for i, task in enumerate(existing_tasks)}
    
    # Process new tasks
    for new_task in new_tasks:
        task_id = new_task.get("id", f"new_task_{len(task_map)}")
        
        if task_id in task_map:
            # Update existing task (especially status changes)
            existing_task = task_map[task_id]
            existing_task.update(new_task)
        else:
            # Add new task
            task_map[task_id] = new_task
    
    # Return updated task list, sorted by creation order
    return list(task_map.values())

# Use standardized logging
logger = get_logger(__name__)

# Define the multi-agent state following official MessagesState pattern
class MultiAgentState(MessagesState):
    """State shared across all agents - inherits messages from MessagesState"""
    # messages field inherited from MessagesState with proper add_messages reducer
    current_agent: str
    current_model: str
    gpu_memory: Dict[str, int]
    context: Dict[str, Any]
    task_queue: Annotated[List[Dict[str, Any]], task_queue_reducer]  # CRITICAL FIX: Use custom reducer
    agent_states: Dict[str, Dict[str, Any]]
    error_count: int
    metadata: Dict[str, Any]

class BaseAgent(ABC):
    """Base class for all agents with async context manager support"""
    
    def __init__(self, name: str, model_name: Optional[str] = None):
        self.name = name
        self.model_name = model_name or "default"
        self.redis_client = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize agent resources"""
        try:
            if self._initialized:
                logger.warning(f"Agent {self.name} already initialized")
                return
                
            redis_config = Config.get_redis_config()
            self.redis_client = await aioredis.from_url(**redis_config)
            
            # Test connection with timeout
            await asyncio.wait_for(self.redis_client.ping(), timeout=10)
            self._initialized = True
            logger.info(f"Agent {self.name} initialized with secure Redis connection")
            
        except asyncio.TimeoutError:
            logger.error(f"Redis connection timeout for agent {self.name}")
            raise ConnectionError("Redis connection timeout")
        except Exception as e:
            logger.error(f"Failed to initialize Redis for agent {self.name}: {e}")
            self.redis_client = None
            raise
    
    async def cleanup(self):
        """Cleanup agent resources"""
        try:
            if self.redis_client:
                await self.redis_client.aclose()
                logger.info(f"Agent {self.name} Redis connection closed")
        except Exception as e:
            logger.warning(f"Error during cleanup for agent {self.name}: {e}")
        finally:
            self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        if not self._initialized:
            await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        if exc_type:
            logger.error(f"Agent {self.name} exited with exception: {exc_val}")
        return False  # Don't suppress exceptions
    
    @abstractmethod
    async def process(self, state: MultiAgentState) -> Dict[str, Any]:
        """Process the current state and return updates"""
        pass
    
    async def save_state(self, state_data: Dict[str, Any]):
        """Save agent state to Redis"""
        key = f"agent:{self.name}:state"
        await self.redis_client.set(key, json.dumps(state_data))
        await self.redis_client.expire(key, 3600)  # 1 hour TTL
    
    async def load_state(self) -> Optional[Dict[str, Any]]:
        """Load agent state from Redis"""
        key = f"agent:{self.name}:state"
        data = await self.redis_client.get(key)
        return json.loads(data) if data else None

class OrchestratorAgent(BaseAgent):
    """Main orchestrator that routes tasks to appropriate agents"""
    
    def __init__(self):
        super().__init__("orchestrator")
        self.agent_registry = {}
    
    def register_agent(self, agent_name: str, agent_instance: BaseAgent):
        """Register an agent with the orchestrator"""
        self.agent_registry[agent_name] = agent_instance
        logger.info(f"Registered agent: {agent_name}")
    
    async def process(self, state: MultiAgentState) -> Dict[str, Any]:
        """
        Orchestrator processing following LangGraph patterns.
        Analyzes request and determines which agent should handle it.
        """
        logger.info("Orchestrator processing request...")
        
        # Get the last message
        if not state.get("messages"):
            logger.warning("No messages found in state")
            return {
                "current_agent": "end", 
                "error_count": state.get("error_count", 0) + 1
            }
        
        # Check for pending tasks first
        task_queue = state.get("task_queue", [])
        pending_tasks = [t for t in task_queue if t.get("status") == "pending"]
        
        if pending_tasks:
            # We have pending tasks, route to executor
            next_agent = "executor"
            response_content = f"🔄 **Task Queue Processing**\n\nFound {len(pending_tasks)} pending tasks. Routing to executor."
            logger.info(f"Found {len(pending_tasks)} pending tasks, routing to executor")
        else:
            # Analyze the user request to determine appropriate agent
            last_message = state["messages"][-1]
            content = last_message.content.lower() if hasattr(last_message, 'content') else ""
            
            # Intelligent routing based on content analysis - CHECK SPECIFIC KEYWORDS FIRST
            if any(keyword in content for keyword in ["search for", "search", "find", "lookup", "research"]):
                next_agent = "web_search"  # Web search requests go directly to web search
                response_content = f"🔍 **Web Search Mode**\n\nSearching for: {content[:100]}..."
                logger.info("Routing to web_search for search query")
            elif any(keyword in content for keyword in ["scraper", "scraping", "crawl", "extract", "web scraper", "web scraping"]):
                next_agent = "planner"  # Complex web tasks need planning first
                response_content = f"🕷️ **Web Scraper Planning**\n\nPlanning web scraper for: {content[:100]}..."
                logger.info("Routing to planner for web scraper task")
            elif any(keyword in content for keyword in ["plan", "analyze", "design", "strategy", "approach"]):
                next_agent = "planner"
                response_content = f"📋 **Planning Mode**\n\nAnalyzing request for comprehensive planning: {content[:100]}..."
                logger.info("Routing to planner based on planning keywords")
            elif any(keyword in content for keyword in ["execute:", "run:", "install"]) and not any(keyword in content for keyword in ["plan", "create", "build"]):
                # Only route to executor for explicit execution commands without complexity
                next_agent = "executor" 
                response_content = f"⚡ **Execution Mode**\n\nExecuting request: {content[:100]}..."
                logger.info("Routing to executor based on execution keywords")
            else:
                # Default to web search for simple queries
                next_agent = "web_search"
                response_content = f"🔍 **Web Search Mode**\n\nSearching for: {content[:100]}..."
                logger.info("Default routing to web_search for simple queries")
        
        # Save orchestrator state
        await self.save_state({
            "last_routing": next_agent,
            "message_content": content[:200] if 'content' in locals() else "task_queue_processing",
            "timestamp": datetime.now().isoformat(),
            "pending_tasks": len(pending_tasks),
            "routing_reason": "task_queue" if pending_tasks else "content_analysis"
        })
        
        # Create response message
        response_message = AIMessage(content=response_content)
        
        # Return state updates with routing decision
        return {
            "messages": [response_message],
            "current_agent": next_agent,  # CRITICAL: This tells routing function where to go
            "metadata": {
                **state.get("metadata", {}),
                "orchestrator_timestamp": datetime.now().isoformat(),
                "routed_to": next_agent
            }
        }

class InputValidator:
    """Input validation and sanitization for security"""
    
    @staticmethod
    def sanitize_user_input(user_input: str) -> str:
        """Sanitize and validate user input"""
        if not isinstance(user_input, str):
            raise ValueError("Input must be a string")
            
        # Check length
        if len(user_input) > Config.MAX_INPUT_LENGTH:
            raise ValueError(f"Input too long. Maximum {Config.MAX_INPUT_LENGTH} characters")
            
        if len(user_input.strip()) < 1:
            raise ValueError("Input cannot be empty")
        
        # HTML escape to prevent injection
        sanitized = html.escape(user_input.strip())
        
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',               # JavaScript protocol
            r'on\w+\s*=',                # Event handlers
            r'eval\s*\(',                # eval() calls
            r'exec\s*\(',                # exec() calls
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    @staticmethod
    def validate_thread_id(thread_id: Optional[str]) -> Optional[str]:
        """Validate thread ID format"""
        if thread_id is None:
            return None
            
        if not isinstance(thread_id, str):
            raise ValueError("Thread ID must be a string")
            
        # Allow only alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', thread_id):
            raise ValueError("Thread ID contains invalid characters")
            
        if len(thread_id) > 100:
            raise ValueError("Thread ID too long")
            
        return thread_id

class MultiAgentOrchestrator:
    """Main orchestration system using LangGraph"""
    
    def __init__(self, use_memory: bool = True):
        self.graph = None
        self.memory = InMemorySaver() if use_memory else None
        self.agents = {}
        self.redis_client = None
        self.validator = InputValidator()
        
    async def initialize(self):
        """Initialize the orchestration system"""
        try:
            # Connect to Redis with secure configuration
            redis_config = Config.get_redis_config()
            self.redis_client = await aioredis.from_url(**redis_config)
            
            # Test Redis connection with timeout
            await asyncio.wait_for(self.redis_client.ping(), timeout=10)
            
            # Initialize orchestrator
            self.orchestrator = OrchestratorAgent()
            await self.orchestrator.initialize()
            
            logger.info("Multi-agent orchestration system initialized with secure configuration")
            
        except asyncio.TimeoutError:
            logger.error("Redis connection timeout during initialization")
            raise ConnectionError("Redis connection timeout")
        except ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            raise
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize orchestration system: {e}")
            raise RuntimeError(f"Initialization failed: {str(e)}")
    
    def add_agent(self, name: str, agent: BaseAgent):
        """Add an agent to the system"""
        self.agents[name] = agent
        self.orchestrator.register_agent(name, agent)
        logger.info(f"Added agent: {name}")
    
    def build_graph(self):
        """Build the LangGraph workflow following official patterns"""
        # Create the graph with proper state type
        workflow = StateGraph(MultiAgentState)
        
        # Add orchestrator node
        workflow.add_node("orchestrator", self._create_node_handler(self.orchestrator))
        
        # Add all registered agent nodes
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, self._create_node_handler(agent))
        
        # Set entry point to orchestrator
        workflow.set_entry_point("orchestrator")
        
        # Create routing map for orchestrator - SINGLE conditional edge
        routing_map = {}
        for agent_name in self.agents.keys():
            routing_map[agent_name] = agent_name
        routing_map["end"] = END  # Use proper END constant
        
        # Add SINGLE conditional edge from orchestrator
        workflow.add_conditional_edges(
            "orchestrator",
            self._orchestrator_routing,
            routing_map
        )
        
        # Add edges from agents - each agent decides its own next step
        for agent_name in self.agents.keys():
            workflow.add_conditional_edges(
                agent_name,
                self._agent_completion_routing,
                {
                    "continue": "orchestrator",  # Return to orchestrator for next task
                    "end": END  # Terminate workflow
                }
            )
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.memory)
        logger.info("Workflow graph built successfully with proper LangGraph patterns")
    
    def _create_node_handler(self, agent: BaseAgent):
        """Create a node handler for an agent"""
        async def handler(state: MultiAgentState) -> Dict[str, Any]:
            return await agent.process(state)
        return handler
    
    def _orchestrator_routing(self, state: MultiAgentState) -> str:
        """
        Orchestrator routing function following LangGraph patterns.
        This function determines routing BASED ON the current_agent field
        that was set by the orchestrator node execution.
        """
        # Check for termination conditions first
        if state.get("error_count", 0) > 3:
            logger.info("Terminating due to too many errors")
            return "end"
        
        # Get the routing decision from current_agent field (set by orchestrator)
        current_agent = state.get("current_agent", "end")
        
        # Validate the routing decision
        if current_agent in self.agents:
            logger.info(f"Routing to agent: {current_agent}")
            return current_agent
        elif current_agent == "end":
            logger.info("Terminating workflow as requested")
            return "end"
        else:
            logger.warning(f"Invalid agent specified: {current_agent}, terminating")
            return "end"
    
    def _agent_completion_routing(self, state: MultiAgentState) -> str:
        """
        Agent completion routing function.
        Returns 'continue' to return to orchestrator or 'end' to terminate.
        """
        # Check error conditions
        if state.get("error_count", 0) > 3:
            logger.info("Agent terminating due to errors")
            return "end"
        
        # Check if agent has set current_agent to "end"
        if state.get("current_agent") == "end":
            logger.info("Agent requested termination")
            return "end"
        
        # Check for pending tasks that might need orchestrator routing
        task_queue = state.get("task_queue", [])
        pending_tasks = [t for t in task_queue if t.get("status") == "pending"]
        
        if pending_tasks:
            # Check if all pending tasks are for the same agent
            agent_types = set(t.get("agent", "unknown") for t in pending_tasks)
            if len(agent_types) == 1 and "executor" in agent_types:
                # All tasks are for executor, continue in executor
                logger.info("All pending tasks for executor, continuing without orchestrator")
                return "end"  # Let executor handle its own tasks
            else:
                # Mixed agent types, need orchestrator routing
                logger.info("Mixed agent types in pending tasks, returning to orchestrator")
                return "continue"
        
        # No pending tasks, terminate
        logger.info("No pending tasks, agent terminating")
        return "end"
    
    async def process_request(self, user_input: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user request through the multi-agent system"""
        try:
            # Validate and sanitize inputs
            sanitized_input = self.validator.sanitize_user_input(user_input)
            validated_thread_id = self.validator.validate_thread_id(thread_id)
            
            logger.info(f"Processing request: {sanitized_input[:100]}...")
            
            initial_state = MultiAgentState(
                messages=[HumanMessage(content=sanitized_input)],
                current_agent="orchestrator",
                current_model="",
                gpu_memory={},
                context={},
                task_queue=[],
                agent_states={},
                error_count=0,
                metadata={"start_time": datetime.now().isoformat()}
            )
        except ValueError as e:
            logger.warning(f"Input validation failed: {e}")
            return {
                "success": False,
                "error": f"Input validation error: {str(e)}",
                "messages": []
            }
        
        config = {"configurable": {"thread_id": validated_thread_id}} if validated_thread_id else {}
        
        try:
            result = await self.graph.ainvoke(initial_state, config)
            return {
                "success": True,
                "messages": result.get("messages", []),
                "final_agent": result.get("current_agent"),
                "metadata": result.get("metadata", {})
            }
        except asyncio.TimeoutError:
            logger.error("Request processing timed out")
            return {
                "success": False,
                "error": "Request processing timed out",
                "messages": initial_state["messages"]
            }
        except ConnectionError as e:
            logger.error(f"Connection error during request processing: {e}")
            return {
                "success": False,
                "error": f"Connection error: {str(e)}",
                "messages": initial_state["messages"]
            }
        except ValueError as e:
            logger.error(f"Invalid data during request processing: {e}")
            return {
                "success": False,
                "error": f"Invalid data: {str(e)}",
                "messages": initial_state["messages"]
            }
        except KeyError as e:
            logger.error(f"Missing required data during request processing: {e}")
            return {
                "success": False,
                "error": f"Missing required data: {str(e)}",
                "messages": initial_state["messages"]
            }
        except Exception as e:
            logger.critical(f"Unexpected error processing request: {e}")
            # Re-raise unexpected errors in development
            if os.getenv('ENVIRONMENT') == 'development':
                raise
            return {
                "success": False,
                "error": "An unexpected error occurred",
                "messages": initial_state["messages"]
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get the current system status"""
        status = {
            "agents": list(self.agents.keys()),
            "orchestrator_active": bool(self.orchestrator),
            "memory_enabled": bool(self.memory),
            "redis_connected": False
        }
        
        # Check Redis connection
        try:
            await self.redis_client.ping()
            status["redis_connected"] = True
        except Exception as e:
            logger.error(f"ORCHESTRATOR HEALTH CHECK FAILURE: Redis connection test failed: {str(e)}")
            status["redis_connected"] = False
            status["redis_error"] = str(e)
        
        return status
    
    async def cleanup(self):
        """Cleanup system resources"""
        # Cleanup all agents
        for agent in self.agents.values():
            await agent.cleanup()
        
        await self.orchestrator.cleanup()
        
        if self.redis_client:
            try:
                await self.redis_client.aclose()
                logger.info("Redis client closed successfully")
            except Exception as e:
                logger.error(f"ORCHESTRATOR CLEANUP FAILURE: Error closing Redis client: {str(e)}")
                # Don't re-raise since we're in cleanup
        
        logger.info("System cleanup completed")

# Example usage
async def main():
    """Example of using the orchestration system"""
    print("=== Multi-Agent Orchestration System ===\n")
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(use_memory=True)
    await orchestrator.initialize()
    
    # Get system status
    status = await orchestrator.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")
    
    print("\n✅ Orchestration system ready!")
    print("\nNext steps:")
    print("1. Implement specific agents (planner, executor, web_search, code_gen)")
    print("2. Add real LLM integration")
    print("3. Implement tool connections")
    print("4. Add monitoring and logging")
    
    await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())