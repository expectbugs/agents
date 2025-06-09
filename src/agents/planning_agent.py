#!/usr/bin/env python3
"""
Planning Agent - Breaks down complex tasks into actionable steps
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
import asyncio
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.llms import LlamaCpp
try:
    from ..orchestrator import BaseAgent, MultiAgentState
    from ..config import Config
    from ..logging_config import get_logger
except ImportError:
    from orchestrator import BaseAgent, MultiAgentState
    from config import Config
    from logging_config import get_logger

logger = get_logger(__name__)

class PlanningAgent(BaseAgent):
    """Agent responsible for breaking down tasks into actionable plans"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("planner")
        self.model_path = model_path or Config.MODEL_PATH
        self.llm = None
        self.system_prompt = """You are a planning assistant that breaks down complex tasks into clear, actionable steps.
When given a task, you should:
1. Analyze what needs to be done
2. Break it into 3-5 concrete steps
3. Identify which agents or tools might be needed
4. Consider potential challenges

Format your response as a structured plan with numbered steps."""
    
    async def initialize(self):
        """Initialize the planning agent"""
        await super().initialize()
        
        # Initialize LLM with comprehensive error handling
        await self._initialize_llm()
    
    async def _initialize_llm(self):
        """Initialize LLM with proper error handling and validation"""
        if not os.path.exists(self.model_path):
            logger.info(f"Model not found at {self.model_path}. Using mock planning logic.")
            self.llm = None
            return
        
        try:
            # Check if model file is readable
            if not os.access(self.model_path, os.R_OK):
                raise PermissionError(f"No read permission for model file: {self.model_path}")
            
            # Get model configuration
            model_config = Config.get_model_config()
            model_config.update({
                'max_tokens': 512,
                'n_batch': 512
            })
            
            # Initialize LLM with timeout
            def init_llm():
                return LlamaCpp(**model_config)
            
            # Run LLM initialization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.llm = await asyncio.wait_for(
                loop.run_in_executor(None, init_llm),
                timeout=60  # 60 second timeout
            )
            
            logger.info(f"Planning agent LLM loaded successfully from {self.model_path}")
            
        except asyncio.TimeoutError:
            logger.error(f"LLM initialization timed out for {self.model_path}")
            self.llm = None
        except PermissionError as e:
            logger.error(f"Permission error loading LLM: {e}")
            self.llm = None
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            self.llm = None
        except ImportError as e:
            logger.error(f"Missing dependencies for LLM: {e}")
            self.llm = None
        except Exception as e:
            logger.error(f"Unexpected error loading LLM: {e}")
            self.llm = None
        
        if self.llm is None:
            logger.error(f"PLANNING AGENT INITIALIZATION FAILED: Cannot load LLM from {self.model_path}. Planning agent will not work!")
    
    async def process(self, state: MultiAgentState) -> Dict[str, Any]:
        """Process the planning request"""
        logger.info("Planning agent processing...")
        
        # Check if we already have a plan for this request
        existing_tasks = state.get("task_queue", [])
        context = state.get("context", {})
        
        # If we already have tasks and a current plan, don't regenerate
        if existing_tasks and context.get("current_plan"):
            logger.info(f"Plan already exists with {len(existing_tasks)} tasks, skipping planning")
            return {
                "messages": [AIMessage(content="📋 **Plan Already Exists**\n\nUsing existing plan with current tasks.")],
                "current_agent": "executor" if any(t.get("status") == "pending" for t in existing_tasks) else "end",
                "task_queue": [],  # Don't add duplicate tasks
                "context": context
            }
        
        # Get the task from messages
        task = self._extract_task(state.get("messages", []))
        if not task:
            return {
                "messages": [AIMessage(content="No clear task found to plan.")],
                "current_agent": "orchestrator",
                "error_count": state.get("error_count", 0) + 1
            }
        
        # Generate plan
        plan = await self._generate_plan(task)
        
        # Parse plan into task queue
        task_queue = self._parse_plan_to_tasks(plan)
        
        # Save planning state
        await self.save_state({
            "last_task": task,
            "generated_plan": plan,
            "task_count": len(task_queue),
            "timestamp": datetime.now().isoformat()
        })
        
        # Create response
        response_message = AIMessage(
            content=f"📋 **Planning Complete**\n\n{plan}\n\nIdentified {len(task_queue)} subtasks."
        )
        
        # Update state
        updates = {
            "messages": [response_message],
            "current_agent": "executor" if task_queue else "end",
            "task_queue": state.get("task_queue", []) + task_queue,
            "context": {
                **state.get("context", {}),
                "current_plan": plan,
                "planning_timestamp": datetime.now().isoformat()
            }
        }
        
        return updates
    
    def _extract_task(self, messages: List) -> Optional[str]:
        """Extract the main task from messages"""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'content') and msg.content):
                content = msg.content if hasattr(msg, 'content') else str(msg)
                if len(content) > 10:  # Minimal task length
                    return content
        return None
    
    async def _generate_plan(self, task: str) -> str:
        """Generate a plan for the given task - FAILS LOUDLY if LLM not available"""
        if self.llm is None:
            raise RuntimeError(f"PLANNING AGENT FAILURE: LLM is not available! Model path: {self.model_path}. Cannot generate plan for task: {task}")
        
        try:
            prompt = f"{self.system_prompt}\n\nTask: {task}\n\nPlan:"
            
            # Add timeout to LLM invocation
            plan = await asyncio.wait_for(
                self.llm.ainvoke(prompt),
                timeout=Config.ASYNC_TIMEOUT
            )
            
            # Validate plan output - FAIL LOUDLY if empty
            if not plan or len(plan.strip()) < 10:
                raise RuntimeError(f"PLANNING AGENT FAILURE: LLM generated empty or invalid plan for task: {task}. Plan output: '{plan}'")
                
            return plan
            
        except asyncio.TimeoutError:
            raise RuntimeError(f"PLANNING AGENT FAILURE: LLM plan generation timed out after {Config.ASYNC_TIMEOUT} seconds for task: {task}")
        except Exception as e:
            raise RuntimeError(f"PLANNING AGENT FAILURE: LLM error during plan generation for task '{task}': {str(e)}")
    
    
    def _parse_plan_to_tasks(self, plan: str) -> List[Dict[str, Any]]:
        """Parse the plan text into structured tasks"""
        tasks = []
        lines = plan.split('\n')
        
        current_task = None
        task_number = 0
        
        for line in lines:
            line = line.strip()
            
            # Look for numbered items (1., 2., etc.)
            if line and line[0].isdigit() and '.' in line[:3]:
                if current_task:
                    tasks.append(current_task)
                
                task_number += 1
                # Extract task title
                title = line.split('.', 1)[1].strip()
                if '**' in title:
                    title = title.replace('**', '').strip()
                
                current_task = {
                    "id": f"task_{task_number}",
                    "title": title,
                    "agent": None,
                    "description": "",
                    "status": "pending",
                    "created_at": datetime.now().isoformat()
                }
            
            # Look for agent assignment
            elif current_task and "agent:" in line.lower():
                agent_name = line.split(':', 1)[1].strip()
                current_task["agent"] = agent_name
            
            # Add to description
            elif current_task and line and not line.startswith('*'):
                current_task["description"] += line + " "
        
        # Add last task
        if current_task:
            tasks.append(current_task)
        
        return tasks

# Example standalone test
async def test_planning_agent():
    """Test the planning agent"""
    import asyncio
    
    agent = PlanningAgent()
    await agent.initialize()
    
    # Test state
    test_state = MultiAgentState(
        messages=[HumanMessage(content="Create a Python web scraper for news articles")],
        current_agent="planner",
        current_model="",
        gpu_memory={},
        context={},
        task_queue=[],
        agent_states={},
        error_count=0,
        metadata={}
    )
    
    result = await agent.process(test_state)
    
    print("Planning Result:")
    print(f"Generated {len(result.get('task_queue', []))} tasks")
    print(f"\nPlan:\n{result['messages'][0].content}")
    
    await agent.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_planning_agent())