#!/usr/bin/env python3
"""
Integration Test for Multi-Agent System
Tests the complete workflow from user input to task completion
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add agents directory to path
sys.path.append(str(Path(__file__).parent / "agents"))

from orchestrator import MultiAgentOrchestrator
from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent
from langchain_core.messages import HumanMessage

async def test_complete_workflow():
    """Test the complete multi-agent workflow"""
    print("🚀 Starting Multi-Agent Integration Test")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(use_memory=True)
    await orchestrator.initialize()
    
    # Initialize and add agents
    print("\n📋 Initializing agents...")
    
    planning_agent = PlanningAgent()
    await planning_agent.initialize()
    orchestrator.add_agent("planner", planning_agent)
    
    execution_agent = ExecutionAgent()
    await execution_agent.initialize()
    orchestrator.add_agent("executor", execution_agent)
    
    # Build the workflow graph
    print("🔗 Building workflow graph...")
    orchestrator.build_graph()
    
    # Check system status
    status = await orchestrator.get_system_status()
    print(f"\n📊 System Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("🎯 TEST 1: Complete Planning and Execution Workflow")
    print("="*60)
    
    # Test 1: Complete workflow
    user_request = "Create a Python web scraper for extracting news articles from websites"
    print(f"\n👤 User Request: {user_request}")
    
    result = await orchestrator.process_request(user_request, thread_id="test-session-1")
    
    print(f"\n📤 Workflow Result:")
    print(f"   Success: {result['success']}")
    print(f"   Final Agent: {result.get('final_agent', 'N/A')}")
    print(f"   Messages: {len(result.get('messages', []))}")
    
    # Print message flow
    print(f"\n💬 Message Flow:")
    for i, msg in enumerate(result.get('messages', []), 1):
        content = msg.content if hasattr(msg, 'content') else str(msg)
        print(f"   {i}. {content[:100]}...")
    
    print("\n" + "="*60)
    print("🎯 TEST 2: Planning-Only Request")
    print("="*60)
    
    # Test 2: Planning-only request
    planning_request = "Plan how to analyze stock market data using Python"
    print(f"\n👤 User Request: {planning_request}")
    
    result2 = await orchestrator.process_request(planning_request, thread_id="test-session-2")
    
    print(f"\n📤 Planning Result:")
    print(f"   Success: {result2['success']}")
    print(f"   Messages: {len(result2.get('messages', []))}")
    
    print("\n" + "="*60)
    print("🎯 TEST 3: Direct Execution Request")
    print("="*60)
    
    # Test 3: Direct execution
    execution_request = "Execute: Install pandas and matplotlib libraries"
    print(f"\n👤 User Request: {execution_request}")
    
    result3 = await orchestrator.process_request(execution_request, thread_id="test-session-3")
    
    print(f"\n📤 Execution Result:")
    print(f"   Success: {result3['success']}")
    
    print("\n" + "="*60)
    print("🎯 TEST 4: Memory and State Persistence")
    print("="*60)
    
    # Test 4: Memory persistence
    print("\n🧠 Testing conversation memory...")
    
    # First message in conversation
    result4a = await orchestrator.process_request(
        "I want to build a machine learning model", 
        thread_id="memory-test"
    )
    
    # Follow-up message using same thread
    result4b = await orchestrator.process_request(
        "What was my previous request?", 
        thread_id="memory-test"
    )
    
    print(f"   First request result: {result4a['success']}")
    print(f"   Follow-up result: {result4b['success']}")
    
    print("\n" + "="*60)
    print("🎯 TEST 5: Error Handling")
    print("="*60)
    
    # Test 5: Error handling
    print("\n⚠️  Testing error handling...")
    
    result5 = await orchestrator.process_request("", thread_id="error-test")
    print(f"   Empty request handling: {result5['success']}")
    
    print("\n" + "="*60)
    print("📈 SYSTEM PERFORMANCE METRICS")
    print("="*60)
    
    # Get final system status
    final_status = await orchestrator.get_system_status()
    print(f"\n📊 Final System Status:")
    for key, value in final_status.items():
        print(f"   {key}: {value}")
    
    # Check workspace
    workspace_dir = Path("/home/user/agents/workspace")
    if workspace_dir.exists():
        workspace_contents = list(workspace_dir.glob("*"))
        print(f"\n📁 Workspace Contents: {len(workspace_contents)} items")
        for item in workspace_contents[:5]:  # Show first 5 items
            print(f"   - {item.name}")
    
    print("\n" + "="*60)
    print("✅ INTEGRATION TEST SUMMARY")
    print("="*60)
    
    # Test summary
    test_results = {
        "complete_workflow": result['success'],
        "planning_only": result2['success'],
        "direct_execution": result3['success'],
        "memory_persistence": result4a['success'] and result4b['success'],
        "error_handling": not result5['success']  # Should fail gracefully
    }
    
    print(f"\n🎯 Test Results:")
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    overall_success = all(test_results.values())
    print(f"\n🏆 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 Phase 1 Foundation Complete!")
        print("✨ The multi-agent system is ready for Phase 2 development:")
        print("   - Web search capabilities (Perplexica/SearXNG)")
        print("   - Advanced LLM integration")
        print("   - Tool connections")
        print("   - Real-time monitoring")
    
    # Cleanup
    await orchestrator.cleanup()
    await planning_agent.cleanup()
    await execution_agent.cleanup()
    
    print("\n🧹 Cleanup completed")
    return overall_success

async def main():
    """Main test function"""
    try:
        success = await test_complete_workflow()
        exit_code = 0 if success else 1
        print(f"\nExiting with code: {exit_code}")
        return exit_code
    except Exception as e:
        print(f"\n❌ Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())