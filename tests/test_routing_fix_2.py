#!/usr/bin/env python3
"""
Test the routing fix - should now route web scraper requests to planner
"""

import asyncio
import sys
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent / "agents"))

from orchestrator import MultiAgentOrchestrator
from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent

async def test_routing_fix_2():
    """Test that web scraper requests now route to planner first"""
    print("🔧 TESTING ROUTING FIX #2")
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
    
    # Test web scraper request - should go to planner first
    print("\n🔸 Test: Web Scraper Request (should route to planner)")
    print("   Request: 'Create a comprehensive web scraper for news articles'")
    
    result = await orchestrator.process_request(
        "Create a comprehensive web scraper for news articles",
        thread_id="routing-fix-test"
    )
    
    print(f"   Success: {result['success']}")
    print(f"   Messages: {len(result.get('messages', []))}")
    print(f"   Final Agent: {result.get('final_agent')}")
    
    # Check state for task queue
    saved_state = await orchestrator.graph.aget_state(
        {"configurable": {"thread_id": "routing-fix-test"}}
    )
    
    if saved_state:
        task_queue = saved_state.values.get("task_queue", [])
        print(f"   Task Queue Length: {len(task_queue)}")
        
        if len(task_queue) > 0:
            print("   ✅ Task queue generated!")
            print("   Generated tasks:")
            for i, task in enumerate(task_queue, 1):
                print(f"      {i}. {task.get('title')} (Status: {task.get('status')})")
        else:
            print("   ❌ No task queue generated")
    else:
        print("   ❌ No saved state found")
    
    await orchestrator.cleanup()
    
    return len(task_queue) if saved_state and task_queue else 0

if __name__ == "__main__":
    result = asyncio.run(test_routing_fix_2())
    print(f"\nResult: {result} tasks generated")