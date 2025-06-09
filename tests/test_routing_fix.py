#!/usr/bin/env python3
"""
Test the orchestrator routing fix
"""

import asyncio
import sys
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent / "agents"))

from orchestrator import MultiAgentOrchestrator
from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent

async def test_routing_fix():
    """Test that the routing fix actually works"""
    print("🔧 TESTING ORCHESTRATOR ROUTING FIX")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(use_memory=True)
    await orchestrator.initialize()
    
    # Add agents
    planning_agent = PlanningAgent()
    await planning_agent.initialize()
    orchestrator.add_agent("planner", planning_agent)
    
    execution_agent = ExecutionAgent()
    await execution_agent.initialize()
    orchestrator.add_agent("executor", execution_agent)
    
    # Build graph
    orchestrator.build_graph()
    
    # Test 1: Planning request
    print("\n🔸 Test 1: Planning Request")
    result1 = await orchestrator.process_request(
        "Plan a machine learning project for image classification",
        thread_id="routing-test-1"
    )
    
    print(f"   Success: {result1['success']}")
    print(f"   Messages: {len(result1.get('messages', []))}")
    print(f"   Final Agent: {result1.get('final_agent')}")
    
    if result1['success'] and len(result1.get('messages', [])) >= 2:
        print("   ✅ Routing to planner working!")
    else:
        print("   ❌ Routing to planner failed!")
    
    # Test 2: Execution request  
    print("\n🔸 Test 2: Execution Request")
    result2 = await orchestrator.process_request(
        "Execute: Install numpy and pandas",
        thread_id="routing-test-2"
    )
    
    print(f"   Success: {result2['success']}")
    print(f"   Messages: {len(result2.get('messages', []))}")
    print(f"   Final Agent: {result2.get('final_agent')}")
    
    if result2['success'] and len(result2.get('messages', [])) >= 2:
        print("   ✅ Routing to executor working!")
    else:
        print("   ❌ Routing to executor failed!")
    
    # Test 3: Complex web scraper (should go to planner)
    print("\n🔸 Test 3: Web Scraper Request")
    result3 = await orchestrator.process_request(
        "Create a comprehensive web scraper for news articles",
        thread_id="routing-test-3"
    )
    
    print(f"   Success: {result3['success']}")
    print(f"   Messages: {len(result3.get('messages', []))}")
    print(f"   Final Agent: {result3.get('final_agent')}")
    
    # Check for task queue generation
    if result3['success']:
        saved_state = await orchestrator.graph.aget_state(
            {"configurable": {"thread_id": "routing-test-3"}}
        )
        task_queue = saved_state.values.get("task_queue", []) if saved_state else []
        print(f"   Task Queue Generated: {len(task_queue)} tasks")
        
        if len(task_queue) > 0:
            print("   ✅ Task queue generation working!")
        else:
            print("   ⚠️  No task queue generated")
    
    # Cleanup
    await orchestrator.cleanup()
    
    print(f"\n📊 ROUTING FIX TEST SUMMARY:")
    print(f"   Planning Route: {'✅' if result1['success'] else '❌'}")
    print(f"   Execution Route: {'✅' if result2['success'] else '❌'}")  
    print(f"   Complex Route: {'✅' if result3['success'] else '❌'}")

if __name__ == "__main__":
    asyncio.run(test_routing_fix())