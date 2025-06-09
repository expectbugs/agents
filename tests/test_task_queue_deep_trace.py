#!/usr/bin/env python3
"""
DEEP TRACE of task queue generation and persistence
Track exactly what happens to task queues through the entire workflow
"""

import asyncio
import sys
import json
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent / "agents"))

from orchestrator import MultiAgentOrchestrator, MultiAgentState
from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent
from langchain_core.messages import HumanMessage

async def deep_trace_task_queue():
    """Deep trace of task queue through entire workflow"""
    print("🔬 DEEP TASK QUEUE TRACE")
    print("=" * 80)
    
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
    
    print("🔸 STEP 1: Initial State")
    initial_state = MultiAgentState(
        messages=[HumanMessage(content="Create a comprehensive web scraper for news articles")],
        current_agent="orchestrator",
        current_model="",
        gpu_memory={},
        context={},
        task_queue=[],
        agent_states={},
        error_count=0,
        metadata={}
    )
    print(f"   Task Queue Length: {len(initial_state['task_queue'])}")
    
    print("\n🔸 STEP 2: Direct Planning Agent Test")
    # Test planning agent directly
    planner_result = await planning_agent.process(initial_state)
    print(f"   Planning Agent Generated Tasks: {len(planner_result.get('task_queue', []))}")
    print(f"   Next Agent: {planner_result.get('current_agent')}")
    
    # Show the actual tasks
    if planner_result.get('task_queue'):
        print("   Generated Tasks:")
        for i, task in enumerate(planner_result['task_queue'], 1):
            print(f"      {i}. {task.get('title')} (Agent: {task.get('agent')}, Status: {task.get('status')})")
    
    print("\n🔸 STEP 3: State Merging Test")
    # Test how LangGraph merges state
    merged_state = {**initial_state, **planner_result}
    print(f"   Merged State Task Queue Length: {len(merged_state.get('task_queue', []))}")
    
    # Check for state annotation issues
    print(f"   State annotation working: {len(merged_state['task_queue']) > 0}")
    
    print("\n🔸 STEP 4: Full Workflow Test with Tracing")
    
    # Create a custom workflow tracer
    class WorkflowTracer:
        def __init__(self):
            self.step = 0
            
        async def trace_invoke(self, original_invoke, *args, **kwargs):
            self.step += 1
            print(f"\n   Workflow Step {self.step}: Invoking...")
            
            # Get state before
            config = kwargs.get('config', {})
            thread_id = config.get('configurable', {}).get('thread_id')
            
            if thread_id:
                try:
                    before_state = await orchestrator.graph.aget_state(config)
                    before_tasks = len(before_state.values.get('task_queue', [])) if before_state else 0
                    print(f"      Before: {before_tasks} tasks in queue")
                except:
                    print("      Before: Could not get state")
            
            # Execute
            result = await original_invoke(*args, **kwargs)
            
            # Get state after
            if thread_id:
                try:
                    after_state = await orchestrator.graph.aget_state(config)
                    after_tasks = len(after_state.values.get('task_queue', [])) if after_state else 0
                    print(f"      After: {after_tasks} tasks in queue")
                    
                    # Show task details if any
                    if after_state and after_state.values.get('task_queue'):
                        tasks = after_state.values['task_queue']
                        pending = sum(1 for t in tasks if t.get('status') == 'pending')
                        completed = sum(1 for t in tasks if t.get('status') == 'completed')
                        print(f"      Task Status: {pending} pending, {completed} completed")
                except:
                    print("      After: Could not get state")
            
            return result
    
    # Monkey patch for tracing
    tracer = WorkflowTracer()
    original_invoke = orchestrator.graph.ainvoke
    orchestrator.graph.ainvoke = lambda *args, **kwargs: tracer.trace_invoke(original_invoke, *args, **kwargs)
    
    # Run full workflow
    result = await orchestrator.process_request(
        "Create a comprehensive web scraper for news articles",
        thread_id="deep-trace-test"
    )
    
    print(f"\n🔸 STEP 5: Final Results")
    print(f"   Workflow Success: {result['success']}")
    print(f"   Final Agent: {result.get('final_agent')}")
    print(f"   Messages Generated: {len(result.get('messages', []))}")
    
    # Check final state
    final_state = await orchestrator.graph.aget_state(
        {"configurable": {"thread_id": "deep-trace-test"}}
    )
    
    if final_state:
        final_tasks = final_state.values.get('task_queue', [])
        print(f"   Final Task Queue Length: {len(final_tasks)}")
        
        if final_tasks:
            print("   Final Task Queue Contents:")
            for i, task in enumerate(final_tasks, 1):
                print(f"      {i}. {task.get('title')} (Status: {task.get('status')})")
        else:
            print("   ❌ NO TASKS IN FINAL QUEUE!")
            
            # Let's check what's in the final state
            print(f"   Final State Keys: {list(final_state.values.keys())}")
            print(f"   Messages Count: {len(final_state.values.get('messages', []))}")
            print(f"   Current Agent: {final_state.values.get('current_agent')}")
            print(f"   Metadata: {final_state.values.get('metadata', {})}")
    else:
        print("   ❌ NO FINAL STATE FOUND!")
    
    print("\n🔸 STEP 6: State Annotation Verification")
    # Test the state annotation directly
    test_state1 = MultiAgentState(
        messages=[],
        current_agent="test",
        current_model="",
        gpu_memory={},
        context={},
        task_queue=[{"id": "test1", "title": "Task 1"}],
        agent_states={},
        error_count=0,
        metadata={}
    )
    
    test_state2 = {
        "task_queue": [{"id": "test2", "title": "Task 2"}]
    }
    
    # Test merging
    merged = {**test_state1, **test_state2}
    print(f"   Direct merge test: {len(merged['task_queue'])} tasks")
    print(f"   Merge result: {[t['title'] for t in merged['task_queue']]}")
    
    # This should show us if the annotation is working
    if len(merged['task_queue']) == 1:
        print("   ❌ ANNOTATION NOT WORKING - Tasks being overwritten!")
    else:
        print("   ✅ Annotation working - Tasks being merged")
    
    await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(deep_trace_task_queue())