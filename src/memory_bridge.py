#!/usr/bin/env python3
"""
Bridge between LangGraph agents and Buddy's memory system
Integrates the LangGraph research workflow with persistent memory storage
"""

import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

# Add Buddy system to path
sys.path.append('/home/user/buddy')

# Import Buddy's memory components
try:
    from enhanced_memory import EnhancedMemory
    from context_logger import ContextLogger
    from context_bridge import ContextBridge
    from context_expander import ContextExpander
    from mem0 import Memory
    import yaml
    BUDDY_AVAILABLE = True
except ImportError as e:
    print(f"MEMORY BRIDGE WARNING: Buddy memory system not available: {e}")
    BUDDY_AVAILABLE = False

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger

logger = get_logger(__name__)

class MemoryBridge:
    """Bridge LangGraph results to Buddy's memory system"""
    
    def __init__(self):
        if not BUDDY_AVAILABLE:
            raise RuntimeError("MEMORY BRIDGE FAILURE: Buddy memory system dependencies not available. Cannot initialize memory bridge.")
        
        try:
            # Initialize Buddy's memory components
            self.context_logger = ContextLogger()
            self.context_bridge = ContextBridge(self.context_logger)
            self.context_expander = ContextExpander(self.context_bridge)
            
            # Load mem0 configuration
            config_path = "/home/user/buddy/mem0_config.yaml"
            if not os.path.exists(config_path):
                raise RuntimeError(f"MEMORY BRIDGE FAILURE: mem0 config not found at {config_path}")
            
            with open(config_path, 'r') as f:
                mem0_config = yaml.safe_load(f)
            
            # Create base mem0 instance
            base_memory = Memory.from_config(config_dict=mem0_config)
            
            # Wrap with enhanced memory
            self.memory = EnhancedMemory(base_memory, self.context_logger, self.context_expander)
            
            logger.info("Memory bridge initialized successfully with Buddy's memory system")
            
        except Exception as e:
            raise RuntimeError(f"MEMORY BRIDGE FAILURE: Failed to initialize memory bridge: {str(e)}")
    
    async def store_workflow_results(self, workflow_id: str, results: Dict[str, Any], user_id: str = "langgraph_system") -> Dict[str, Any]:
        """Store LangGraph workflow results in Buddy's memory"""
        try:
            # Extract meaningful content from results
            content_to_store = self._extract_meaningful_content(results)
            
            # Create conversation format for mem0
            conversation = [
                {"role": "system", "content": f"LangGraph workflow '{workflow_id}' completed"},
                {"role": "assistant", "content": content_to_store}
            ]
            
            # Store in memory with comprehensive metadata
            result = self.memory.add(
                conversation,
                user_id=user_id,
                metadata={
                    "workflow_id": workflow_id,
                    "source": "langgraph",
                    "agent_type": "research_workflow",
                    "timestamp": datetime.now().isoformat(),
                    "workflow_type": "research",
                    "system": "multi_agent_orchestrator"
                }
            )
            
            logger.info(f"Stored workflow {workflow_id} results in memory: {result}")
            return result
            
        except Exception as e:
            raise RuntimeError(f"MEMORY BRIDGE FAILURE: Failed to store workflow results for {workflow_id}: {str(e)}")
    
    def _extract_meaningful_content(self, results: Dict[str, Any]) -> str:
        """Extract meaningful content from workflow results for storage"""
        content_parts = []
        
        # Extract from messages
        messages = results.get('messages', [])
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
                # Skip system/routing messages, focus on actual results
                if not any(skip_phrase in content.lower() for skip_phrase in [
                    'routing to', 'processing', 'mode', 'searching for'
                ]):
                    content_parts.append(content)
        
        # Extract from context if available
        context = results.get('context', {})
        if 'search_results' in context:
            search_results = context['search_results']
            if search_results:
                content_parts.append(f"Found {len(search_results)} search results")
                # Add key findings
                for i, result in enumerate(search_results[:3]):  # Top 3 results
                    title = result.get('title', 'No title')
                    snippet = result.get('snippet', '')[:200]
                    content_parts.append(f"Result {i+1}: {title} - {snippet}")
        
        # Join all content
        combined_content = "\n\n".join(content_parts)
        
        if not combined_content.strip():
            return "LangGraph workflow completed with no extractable content"
        
        return combined_content
    
    async def retrieve_related_memories(self, query: str, user_id: str = "langgraph_system", limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories related to a query"""
        try:
            search_result = self.memory.search(
                query=query,
                user_id=user_id,
                limit=limit
            )
            
            memories = search_result.get('results', [])
            logger.info(f"Retrieved {len(memories)} related memories for query: {query}")
            return memories
            
        except Exception as e:
            raise RuntimeError(f"MEMORY BRIDGE FAILURE: Failed to retrieve memories for query '{query}': {str(e)}")
    
    async def get_workflow_history(self, workflow_type: str = "research", user_id: str = "langgraph_system", limit: int = 10) -> List[Dict[str, Any]]:
        """Get history of previous workflows"""
        try:
            # Search for workflow-related memories
            search_result = self.memory.search(
                query=f"LangGraph workflow {workflow_type}",
                user_id=user_id,
                limit=limit
            )
            
            workflow_memories = search_result.get('results', [])
            
            # Filter to only workflow results
            filtered_memories = []
            for memory in workflow_memories:
                metadata = memory.get('metadata', {})
                if metadata.get('source') == 'langgraph' and metadata.get('workflow_type') == workflow_type:
                    filtered_memories.append(memory)
            
            logger.info(f"Retrieved {len(filtered_memories)} {workflow_type} workflow memories")
            return filtered_memories
            
        except Exception as e:
            raise RuntimeError(f"MEMORY BRIDGE FAILURE: Failed to get workflow history: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if memory bridge and underlying systems are healthy"""
        try:
            health_status = {
                "memory_bridge": True,
                "buddy_available": BUDDY_AVAILABLE,
                "mem0_available": False,
                "context_logging": False,
                "enhanced_memory": False
            }
            
            if BUDDY_AVAILABLE:
                # Test mem0 connection
                try:
                    test_result = self.memory.search("test", user_id="health_check", limit=1)
                    health_status["mem0_available"] = True
                except Exception as e:
                    logger.error(f"MEMORY BRIDGE HEALTH CHECK FAILURE: mem0 connection test failed: {str(e)}")
                    health_status["mem0_available"] = False
                    health_status["mem0_error"] = str(e)
                
                # Test context logging
                try:
                    if self.context_logger:
                        health_status["context_logging"] = True
                except Exception as e:
                    logger.error(f"MEMORY BRIDGE HEALTH CHECK FAILURE: context logging test failed: {str(e)}")
                    health_status["context_logging"] = False
                    health_status["context_logging_error"] = str(e)
                
                # Test enhanced memory
                try:
                    if self.memory:
                        health_status["enhanced_memory"] = True
                except Exception as e:
                    logger.error(f"MEMORY BRIDGE HEALTH CHECK FAILURE: enhanced memory test failed: {str(e)}")
                    health_status["enhanced_memory"] = False
                    health_status["enhanced_memory_error"] = str(e)
            
            logger.info(f"Memory bridge health check: {health_status}")
            return health_status
            
        except Exception as e:
            raise RuntimeError(f"MEMORY BRIDGE FAILURE: Health check failed: {str(e)}")


# Test function for development
async def test_memory_bridge():
    """Test the memory bridge functionality"""
    try:
        print("Testing Memory Bridge...")
        
        # Initialize bridge
        bridge = MemoryBridge()
        
        # Health check
        health = await bridge.health_check()
        print(f"Health Check: {health}")
        
        # Test storing workflow results
        test_results = {
            'messages': [
                {'content': 'Test search results for AI safety regulations'},
                {'content': 'Found 5 relevant articles about AI safety frameworks'}
            ],
            'context': {
                'search_results': [
                    {'title': 'AI Safety Framework 2024', 'snippet': 'New regulatory framework...'},
                    {'title': 'EU AI Act Implementation', 'snippet': 'European Union implements...'}
                ]
            }
        }
        
        store_result = await bridge.store_workflow_results(
            workflow_id="test_research_001",
            results=test_results
        )
        print(f"Store Result: {store_result}")
        
        # Test retrieving memories
        memories = await bridge.retrieve_related_memories("AI safety regulations")
        print(f"Retrieved {len(memories)} memories")
        
        # Test workflow history
        history = await bridge.get_workflow_history("research")
        print(f"Retrieved {len(history)} workflow history entries")
        
        print("✅ Memory Bridge test completed successfully!")
        
    except Exception as e:
        print(f"❌ Memory Bridge test failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_memory_bridge())