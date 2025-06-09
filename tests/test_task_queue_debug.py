#!/usr/bin/env python3
"""
Debug task queue generation in planning agent
"""

import asyncio
import sys
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent / "agents"))

from agents.planning_agent import PlanningAgent
from orchestrator import MultiAgentState
from langchain_core.messages import HumanMessage

async def debug_task_queue():
    """Debug task queue generation"""
    print("🔍 DEBUGGING TASK QUEUE GENERATION")
    print("=" * 50)
    
    # Initialize planning agent
    agent = PlanningAgent()
    await agent.initialize()
    
    # Create test state
    test_state = MultiAgentState(
        messages=[HumanMessage(content="Create a comprehensive web scraper for news articles")],
        current_agent="planner",
        current_model="",
        gpu_memory={},
        context={},
        task_queue=[],
        agent_states={},
        error_count=0,
        metadata={}
    )
    
    print("🔸 Processing planning request...")
    result = await agent.process(test_state)
    
    print(f"\n📊 RESULTS:")
    print(f"   Task Queue Length: {len(result.get('task_queue', []))}")
    print(f"   Current Agent: {result.get('current_agent')}")
    print(f"   Messages Generated: {len(result.get('messages', []))}")
    
    # Debug the task queue contents
    task_queue = result.get('task_queue', [])
    if task_queue:
        print(f"\n📋 TASK QUEUE DETAILS:")
        for i, task in enumerate(task_queue, 1):
            print(f"   Task {i}:")
            print(f"      ID: {task.get('id')}")
            print(f"      Title: {task.get('title')}")
            print(f"      Agent: {task.get('agent')}")
            print(f"      Status: {task.get('status')}")
            print(f"      Description: {task.get('description', '')[:100]}...")
    else:
        print(f"\n⚠️  NO TASKS GENERATED!")
        
        # Debug the plan generation
        plan = await agent._generate_plan("Create a comprehensive web scraper for news articles")
        print(f"\n📝 GENERATED PLAN:")
        print(plan)
        
        # Debug the task parsing
        tasks = agent._parse_plan_to_tasks(plan)
        print(f"\n🔧 PARSED TASKS: {len(tasks)}")
        for i, task in enumerate(tasks, 1):
            print(f"   Task {i}: {task.get('title')} (Agent: {task.get('agent')})")
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(debug_task_queue())