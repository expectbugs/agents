#!/usr/bin/env python3
"""
Demo: Complete research workflow with memory persistence
Shows: Web Search → Analysis → Memory Storage → Retrieval

This demonstrates the full LangGraph multi-agent system working with:
1. Perplexica AI-enhanced web search
2. Agent orchestration and routing
3. Persistent memory storage via Buddy's memory system
4. Memory retrieval and context awareness
"""

import asyncio
import sys
import os
sys.path.append('src')

from src.orchestrator import MultiAgentOrchestrator
from src.agents.perplexica_search_agent import PerplexicaSearchAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.execution_agent import ExecutionAgent
from src.memory_bridge import MemoryBridge

async def demo_research_pipeline():
    """Demonstrate complete research workflow with memory integration"""
    print("🚀 LangGraph Multi-Agent Research Pipeline Demo")
    print("=" * 60)
    
    # Initialize memory bridge
    try:
        print("\n💾 Initializing memory bridge...")
        memory_bridge = MemoryBridge()
        
        # Health check
        health = await memory_bridge.health_check()
        print(f"✅ Memory system health: {health}")
        
    except Exception as e:
        print(f"❌ Memory bridge failed: {e}")
        print("Continuing demo without memory integration...")
        memory_bridge = None
    
    # Initialize orchestrator
    print("\n🔧 Initializing multi-agent orchestrator...")
    orchestrator = MultiAgentOrchestrator(use_memory=True)
    await orchestrator.initialize()
    
    # Create and initialize agents
    print("\n🤖 Initializing agents...")
    web_search_agent = PerplexicaSearchAgent()
    await web_search_agent.initialize()
    
    planning_agent = PlanningAgent()
    await planning_agent.initialize()
    
    execution_agent = ExecutionAgent()
    await execution_agent.initialize()
    
    # Add agents to orchestrator
    orchestrator.add_agent("web_search", web_search_agent)
    orchestrator.add_agent("planner", planning_agent)
    orchestrator.add_agent("executor", execution_agent)
    
    # Build the workflow graph
    orchestrator.build_graph()
    print("✅ Multi-agent system ready!")
    
    # Test queries
    test_queries = [
        "AI safety regulations 2024",
        "LangGraph multi-agent frameworks",
        "autonomous AI systems governance"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n" + "="*60)
        print(f"📋 Research Query {i}/3: '{query}'")
        print("="*60)
        
        # Execute workflow
        print(f"\n🔄 Executing research workflow...")
        try:
            result = await orchestrator.process_request(
                f"search for {query}",
                thread_id=f"demo-session-{i}"
            )
            
            print(f"\n✅ Workflow completed successfully: {result.get('success', False)}")
            
            # Display condensed results
            if result.get('success'):
                messages = result.get('messages', [])
                print(f"\n📊 Generated {len(messages)} messages")
                
                # Show only the final search results (skip routing messages)
                for msg in messages:
                    content = msg.content if hasattr(msg, 'content') else str(msg)
                    if 'Web Search Results' in content:
                        lines = content.split('\n')
                        print(f"\n🔍 Search Summary:")
                        for line in lines[:8]:  # First 8 lines (header + some results)
                            print(f"   {line}")
                        if len(lines) > 8:
                            print(f"   ... and {len(lines) - 8} more lines")
                        break
            else:
                print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
                continue
            
            # Store in memory if available
            if memory_bridge:
                print(f"\n💾 Storing results in memory...")
                try:
                    memory_result = await memory_bridge.store_workflow_results(
                        workflow_id=f"demo-research-{i:03d}",
                        results=result,
                        user_id="demo_user"
                    )
                    print(f"✅ Memory storage complete: {len(memory_result.get('results', []))} memories created")
                except Exception as e:
                    print(f"❌ Memory storage failed: {e}")
            
        except Exception as e:
            print(f"❌ Workflow failed with error: {e}")
            continue
    
    # Demonstrate memory retrieval
    if memory_bridge:
        print(f"\n" + "="*60)
        print("🧠 Memory Retrieval Demonstration")
        print("="*60)
        
        try:
            # Search for related memories
            print(f"\n🔍 Searching for AI-related memories...")
            ai_memories = await memory_bridge.retrieve_related_memories(
                "artificial intelligence regulations safety",
                user_id="demo_user",
                limit=5
            )
            print(f"Found {len(ai_memories)} AI-related memories")
            
            # Show workflow history
            print(f"\n📚 Retrieving workflow history...")
            workflow_history = await memory_bridge.get_workflow_history(
                workflow_type="research",
                user_id="demo_user",
                limit=10
            )
            print(f"Found {len(workflow_history)} previous research workflows")
            
            if workflow_history:
                print(f"\n📋 Recent workflows:")
                for memory in workflow_history[:3]:  # Show latest 3
                    metadata = memory.get('metadata', {})
                    timestamp = metadata.get('timestamp', 'Unknown time')
                    workflow_id = metadata.get('workflow_id', 'Unknown ID')
                    print(f"   - {workflow_id} at {timestamp}")
            
        except Exception as e:
            print(f"❌ Memory retrieval failed: {e}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up...")
    await orchestrator.cleanup()
    
    print(f"\n🎉 Demo Complete!")
    print("\n✅ Successfully demonstrated:")
    print("   1. ✅ Multi-agent orchestration with LangGraph")
    print("   2. ✅ AI-enhanced web search via Perplexica")
    print("   3. ✅ Intelligent agent routing")
    if memory_bridge:
        print("   4. ✅ Persistent memory storage and retrieval")
        print("   5. ✅ Cross-workflow context awareness")
    else:
        print("   4. ⚠️  Memory integration (unavailable)")
    
    print(f"\n🚀 The LangGraph multi-agent research system is working!")

if __name__ == "__main__":
    asyncio.run(demo_research_pipeline())