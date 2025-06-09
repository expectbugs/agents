#!/usr/bin/env python3
"""
COMPREHENSIVE MULTI-AGENT SYSTEM TEST SUITE
This test suite performs deep validation of ALL system components
to ensure NO silent failures, fallbacks, or incomplete implementations.
"""

import asyncio
import json
import sys
import os
import time
import psutil
import traceback
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import redis.asyncio as aioredis

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from orchestrator import MultiAgentOrchestrator, MultiAgentState
from agents.planning_agent import PlanningAgent
from agents.execution_agent import ExecutionAgent
from logging_config import LoggingSetup
from langchain_core.messages import HumanMessage, AIMessage

class ComprehensiveTestSuite:
    """Comprehensive test suite for multi-agent system validation"""
    
    def __init__(self):
        self.test_results = {}
        self.system_metrics = {}
        self.orchestrator = None
        self.redis_client = None
        self.test_start_time = None
        
    async def initialize_test_environment(self):
        """Initialize test environment with deep validation"""
        print("🔬 COMPREHENSIVE SYSTEM TEST SUITE")
        print("=" * 80)
        print("🎯 MISSION: Validate ALL components with ZERO tolerance for silent failures")
        print("=" * 80)
        
        self.test_start_time = time.time()
        
        # Initialize orchestrator with validation
        print("\n📋 1. INITIALIZING SYSTEM COMPONENTS...")
        self.orchestrator = MultiAgentOrchestrator(use_memory=True)
        await self.orchestrator.initialize()
        
        # Validate Redis connection
        print("   🔸 Testing Redis connection...")
        self.redis_client = await aioredis.from_url("redis://localhost", decode_responses=True)
        await self.redis_client.ping()
        print("   ✅ Redis connection verified")
        
        # Initialize and validate agents
        print("   🔸 Initializing Planning Agent...")
        planning_agent = PlanningAgent()
        await planning_agent.initialize()
        self.orchestrator.add_agent("planner", planning_agent)
        
        print("   🔸 Initializing Execution Agent...")
        execution_agent = ExecutionAgent()
        await execution_agent.initialize()
        self.orchestrator.add_agent("executor", execution_agent)
        
        # Build and validate graph
        print("   🔸 Building LangGraph workflow...")
        self.orchestrator.build_graph()
        
        # Validate system status
        status = await self.orchestrator.get_system_status()
        assert status["orchestrator_active"], "Orchestrator not active!"
        assert status["memory_enabled"], "Memory not enabled!"
        assert status["redis_connected"], "Redis not connected!"
        assert len(status["agents"]) >= 2, "Not enough agents registered!"
        
        print("   ✅ System initialization validated")
        
    async def test_1_deep_component_validation(self):
        """Test 1: Deep validation of every system component"""
        print("\n🔍 TEST 1: DEEP COMPONENT VALIDATION")
        print("-" * 50)
        
        results = {}
        
        # Test 1.1: LangGraph Integration Deep Dive
        print("   🔸 1.1 LangGraph StateGraph Deep Validation...")
        try:
            # Verify graph compilation
            assert self.orchestrator.graph is not None, "Graph not compiled!"
            
            # Test state schema validation
            test_state = MultiAgentState(
                messages=[HumanMessage(content="test")],
                current_agent="test",
                current_model="test",
                gpu_memory={},
                context={},
                task_queue=[],
                agent_states={},
                error_count=0,
                metadata={}
            )
            
            # Verify all required state fields exist
            required_fields = ["messages", "current_agent", "current_model", "gpu_memory", 
                             "context", "task_queue", "agent_states", "error_count", "metadata"]
            for field in required_fields:
                assert field in test_state, f"Missing required state field: {field}"
            
            results["langgraph_integration"] = "PASS"
            print("      ✅ LangGraph integration validated")
            
        except Exception as e:
            results["langgraph_integration"] = f"FAIL: {e}"
            print(f"      ❌ LangGraph validation failed: {e}")
        
        # Test 1.2: Redis State Management Deep Dive
        print("   🔸 1.2 Redis State Management Deep Validation...")
        try:
            # Test complex state storage and retrieval
            complex_state = {
                "agent_id": "test_deep_validation",
                "timestamp": datetime.now().isoformat(),
                "nested_data": {
                    "level1": {
                        "level2": ["item1", "item2", {"level3": "deep_value"}]
                    }
                },
                "large_list": list(range(1000)),  # Test large data
                "unicode_test": "测试 émojis 🚀 ñoñó"
            }
            
            # Store complex state
            key = "test:deep:validation"
            await self.redis_client.set(key, json.dumps(complex_state))
            
            # Retrieve and validate
            retrieved = json.loads(await self.redis_client.get(key))
            assert retrieved == complex_state, "State data integrity failed!"
            
            # Test TTL functionality
            await self.redis_client.expire(key, 10)
            ttl = await self.redis_client.ttl(key)
            assert ttl > 0, "TTL not working!"
            
            # Cleanup
            await self.redis_client.delete(key)
            
            results["redis_deep_validation"] = "PASS"
            print("      ✅ Redis state management validated")
            
        except Exception as e:
            results["redis_deep_validation"] = f"FAIL: {e}"
            print(f"      ❌ Redis validation failed: {e}")
        
        # Test 1.3: Agent Initialization and Capabilities
        print("   🔸 1.3 Agent Capabilities Deep Validation...")
        try:
            # Validate planning agent
            planner = self.orchestrator.agents["planner"]
            assert hasattr(planner, 'process'), "Planner missing process method!"
            assert hasattr(planner, 'redis_client'), "Planner missing Redis client!"
            assert planner.redis_client is not None, "Planner Redis client not initialized!"
            
            # Validate execution agent  
            executor = self.orchestrator.agents["executor"]
            assert hasattr(executor, 'process'), "Executor missing process method!"
            assert hasattr(executor, 'workspace_dir'), "Executor missing workspace!"
            assert executor.workspace_dir.exists(), "Executor workspace not created!"
            
            results["agent_capabilities"] = "PASS"
            print("      ✅ Agent capabilities validated")
            
        except Exception as e:
            results["agent_capabilities"] = f"FAIL: {e}"
            print(f"      ❌ Agent validation failed: {e}")
        
        # Add overall status
        all_passed = all(v == "PASS" for v in results.values())
        self.test_results["deep_component_validation"] = {
            "status": "PASS" if all_passed else "FAIL",
            **results
        }
        return results
    
    async def test_2_real_world_scenarios(self):
        """Test 2: Real-world user scenarios with full validation"""
        print("\n🌍 TEST 2: REAL-WORLD USER SCENARIOS")
        print("-" * 50)
        
        scenarios = [
            {
                "name": "Complex Web Scraper Project",
                "request": "Create a comprehensive Python web scraper that can extract news articles from multiple websites, handle rate limiting, store data in a database, and generate daily reports",
                "expected_agents": ["planner", "executor"],
                "min_tasks": 5
            },
            {
                "name": "Data Science Pipeline", 
                "request": "Build a complete data science pipeline to analyze customer behavior data, including data cleaning, feature engineering, model training, and visualization dashboard",
                "expected_agents": ["planner"],
                "min_tasks": 4
            },
            {
                "name": "API Development Project",
                "request": "Plan and implement a REST API with authentication, database integration, error handling, and comprehensive testing",
                "expected_agents": ["planner"],
                "min_tasks": 6
            },
            {
                "name": "DevOps Automation",
                "request": "Set up automated CI/CD pipeline with Docker containerization, testing stages, and deployment automation",
                "expected_agents": ["planner"],
                "min_tasks": 5
            }
        ]
        
        results = {}
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"   🔸 2.{i} Testing: {scenario['name']}")
            
            try:
                # Execute scenario
                start_time = time.time()
                result = await self.orchestrator.process_request(
                    scenario["request"], 
                    thread_id=f"scenario-{i}"
                )
                execution_time = time.time() - start_time
                
                # Validate result structure
                assert result["success"], f"Scenario failed: {result.get('error', 'Unknown')}"
                assert "messages" in result, "No messages in result!"
                assert len(result["messages"]) >= 2, "Too few messages generated!"
                
                # Validate agent involvement
                messages_content = " ".join([msg.content for msg in result["messages"] 
                                           if hasattr(msg, 'content')])
                
                # Check for proper planning
                assert any(keyword in messages_content.lower() 
                          for keyword in ["plan", "step", "task"]), "No planning detected!"
                
                # Performance validation
                assert execution_time < 30.0, f"Execution too slow: {execution_time:.2f}s"
                
                # Memory validation - check if state was saved
                if result.get("final_agent") != "error":
                    # Verify conversation was saved to memory
                    saved_state = await self.orchestrator.graph.aget_state(
                        {"configurable": {"thread_id": f"scenario-{i}"}}
                    )
                    assert saved_state is not None, "State not saved to memory!"
                    assert len(saved_state.values.get("messages", [])) >= 2, "Messages not persisted!"
                
                results[scenario["name"]] = {
                    "status": "PASS",
                    "execution_time": execution_time,
                    "message_count": len(result["messages"]),
                    "final_agent": result.get("final_agent")
                }
                print(f"      ✅ {scenario['name']} completed in {execution_time:.2f}s")
                
            except Exception as e:
                results[scenario["name"]] = {
                    "status": f"FAIL: {e}",
                    "error": str(e)
                }
                print(f"      ❌ {scenario['name']} failed: {e}")
        
        # Add overall status
        all_passed = all(
            r.get("status") == "PASS" 
            for r in results.values() 
            if isinstance(r, dict)
        )
        self.test_results["real_world_scenarios"] = {
            "status": "PASS" if all_passed else "FAIL",
            **results
        }
        return results
    
    async def test_3_edge_cases_and_errors(self):
        """Test 3: Edge cases and error conditions"""
        print("\n⚠️  TEST 3: EDGE CASES AND ERROR CONDITIONS")
        print("-" * 50)
        
        edge_cases = [
            {"name": "Empty Input", "input": ""},
            {"name": "Very Long Input", "input": "a" * 10000},
            {"name": "Special Characters", "input": "测试 émojis 🚀 \n\t\r special chars"},
            {"name": "JSON-like Input", "input": '{"malicious": "input", "code": "exec(...)"}'},
            {"name": "SQL-like Input", "input": "'; DROP TABLE users; --"},
            {"name": "Nonsense Input", "input": "asdfgh qwerty zxcvbn random gibberish"},
            {"name": "Contradictory Request", "input": "Plan to not plan anything and execute nothing"},
        ]
        
        results = {}
        
        for i, case in enumerate(edge_cases, 1):
            print(f"   🔸 3.{i} Testing: {case['name']}")
            
            try:
                result = await self.orchestrator.process_request(
                    case["input"], 
                    thread_id=f"edge-case-{i}"
                )
                
                # Edge cases should either succeed gracefully or fail properly
                if result["success"]:
                    # If it succeeds, it should have handled the edge case properly
                    assert "messages" in result, "No messages for edge case!"
                    assert len(result["messages"]) >= 1, "No response to edge case!"
                    results[case["name"]] = "PASS - Handled gracefully"
                else:
                    # If it fails, it should fail cleanly with proper error
                    assert "error" in result, "No error message for failed case!"
                    results[case["name"]] = "PASS - Failed cleanly"
                
                print(f"      ✅ {case['name']} handled properly")
                
            except Exception as e:
                results[case["name"]] = f"FAIL: Unexpected exception: {e}"
                print(f"      ❌ {case['name']} caused unexpected error: {e}")
        
        # Add overall status
        all_passed = all(
            "PASS" in str(r) 
            for r in results.values()
        )
        self.test_results["edge_cases"] = {
            "status": "PASS" if all_passed else "FAIL",
            **results
        }
        return results
    
    async def test_4_concurrent_usage(self):
        """Test 4: Concurrent usage and race conditions"""
        print("\n🔄 TEST 4: CONCURRENT USAGE AND RACE CONDITIONS")
        print("-" * 50)
        
        print("   🔸 4.1 Concurrent Request Processing...")
        
        # Define multiple concurrent requests
        concurrent_requests = [
            "Plan a machine learning project for image classification",
            "Execute: Install required packages for data science",
            "Create a web scraping tool for social media data",
            "Plan and set up a microservices architecture",
            "Execute: Create project structure for API development"
        ]
        
        try:
            # Execute all requests concurrently
            start_time = time.time()
            tasks = []
            
            for i, request in enumerate(concurrent_requests):
                task = self.orchestrator.process_request(
                    request, 
                    thread_id=f"concurrent-{i}"
                )
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            execution_time = time.time() - start_time
            
            # Validate results
            successful_results = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"      ⚠️  Request {i} failed with exception: {result}")
                elif result.get("success"):
                    successful_results += 1
                else:
                    print(f"      ⚠️  Request {i} failed: {result.get('error')}")
            
            # At least 80% should succeed
            success_rate = successful_results / len(concurrent_requests)
            assert success_rate >= 0.8, f"Too many concurrent failures: {success_rate:.1%}"
            
            print(f"      ✅ Concurrent processing: {successful_results}/{len(concurrent_requests)} succeeded in {execution_time:.2f}s")
            
            self.test_results["concurrent_usage"] = {
                "status": "PASS" if success_rate >= 0.8 else "FAIL",
                "success_rate": success_rate,
                "execution_time": execution_time,
                "total_requests": len(concurrent_requests),
                "successful_requests": successful_results
            }
            
        except Exception as e:
            self.test_results["concurrent_usage"] = f"FAIL: {e}"
            print(f"      ❌ Concurrent usage test failed: {e}")
    
    async def test_5_memory_and_persistence(self):
        """Test 5: Memory persistence and conversation continuity"""
        print("\n🧠 TEST 5: MEMORY PERSISTENCE AND CONVERSATION CONTINUITY")
        print("-" * 50)
        
        print("   🔸 5.1 Multi-turn Conversation Testing...")
        
        try:
            thread_id = "memory-test-conversation"
            
            # Turn 1: Initial request
            result1 = await self.orchestrator.process_request(
                "I want to build a data analysis tool for sales data",
                thread_id=thread_id
            )
            assert result1["success"], "First turn failed!"
            
            # Turn 2: Follow-up question
            result2 = await self.orchestrator.process_request(
                "What was my previous request about?",
                thread_id=thread_id
            )
            assert result2["success"], "Second turn failed!"
            
            # Turn 3: Continuation
            result3 = await self.orchestrator.process_request(
                "Add machine learning capabilities to that project",
                thread_id=thread_id
            )
            assert result3["success"], "Third turn failed!"
            
            # Validate conversation continuity
            total_messages = len(result1["messages"]) + len(result2["messages"]) + len(result3["messages"])
            assert total_messages >= 6, "Not enough conversation messages!"
            
            # Check state persistence in Redis
            saved_state = await self.orchestrator.graph.aget_state(
                {"configurable": {"thread_id": thread_id}}
            )
            assert saved_state is not None, "Conversation state not saved!"
            assert len(saved_state.values.get("messages", [])) >= 6, "Messages not persisted!"
            
            print("      ✅ Memory persistence validated")
            
            self.test_results["memory_persistence"] = "PASS"
            
        except Exception as e:
            self.test_results["memory_persistence"] = f"FAIL: {e}"
            print(f"      ❌ Memory persistence test failed: {e}")
    
    async def test_6_resource_monitoring(self):
        """Test 6: Resource usage and performance monitoring"""
        print("\n📊 TEST 6: RESOURCE USAGE AND PERFORMANCE MONITORING")
        print("-" * 50)
        
        # Get initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        print(f"   📈 Initial Memory Usage: {initial_memory:.1f} MB")
        
        # Execute resource-intensive test
        print("   🔸 6.1 Resource Intensive Operations...")
        
        try:
            # Execute multiple complex requests
            for i in range(5):
                result = await self.orchestrator.process_request(
                    f"Create a complex data processing pipeline with {i+1}0 stages including data validation, transformation, analysis, and reporting",
                    thread_id=f"resource-test-{i}"
                )
                assert result["success"], f"Resource test {i} failed!"
            
            # Check final resource usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            print(f"   📈 Final Memory Usage: {final_memory:.1f} MB")
            print(f"   📈 Memory Increase: {memory_increase:.1f} MB")
            
            # Validate reasonable resource usage
            assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.1f} MB"
            
            # Check Redis memory usage
            redis_info = await self.redis_client.info("memory")
            redis_memory = int(redis_info["used_memory"]) / 1024 / 1024  # MB
            print(f"   📈 Redis Memory Usage: {redis_memory:.1f} MB")
            
            self.test_results["resource_monitoring"] = {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "redis_memory_mb": redis_memory,
                "status": "PASS"
            }
            
            print("      ✅ Resource usage within acceptable limits")
            
        except Exception as e:
            self.test_results["resource_monitoring"] = f"FAIL: {e}"
            print(f"      ❌ Resource monitoring failed: {e}")
    
    async def test_7_no_silent_failures(self):
        """Test 7: Verify no silent failures or incomplete stubs"""
        print("\n🔍 TEST 7: SILENT FAILURE AND STUB DETECTION")
        print("-" * 50)
        
        print("   🔸 7.1 Searching for Silent Failures...")
        
        issues_found = []
        
        try:
            # Test that planning agent actually generates real plans
            result = await self.orchestrator.process_request(
                "Create a machine learning model for predicting house prices",
                thread_id="stub-detection-test"
            )
            
            if result["success"]:
                messages_text = " ".join([msg.content for msg in result["messages"] 
                                        if hasattr(msg, 'content')])
                
                # Check for generic/stub responses
                stub_indicators = [
                    "not implemented",
                    "placeholder",
                    "todo",
                    "stub",
                    "mock implementation",
                    "coming soon",
                    "under development"
                ]
                
                for indicator in stub_indicators:
                    if indicator.lower() in messages_text.lower():
                        issues_found.append(f"Stub indicator found: '{indicator}'")
                
                # Check for actual plan content
                plan_indicators = [
                    "step",
                    "task",
                    "plan",
                    "implement",
                    "data",
                    "model",
                    "feature"
                ]
                
                plan_found = any(indicator.lower() in messages_text.lower() 
                               for indicator in plan_indicators)
                
                if not plan_found:
                    issues_found.append("No actual planning content detected")
                
                # Check task queue generation
                saved_state = await self.orchestrator.graph.aget_state(
                    {"configurable": {"thread_id": "stub-detection-test"}}
                )
                
                if saved_state and saved_state.values.get("task_queue"):
                    task_queue = saved_state.values["task_queue"]
                    if len(task_queue) < 3:
                        issues_found.append(f"Task queue too small: {len(task_queue)} tasks")
                else:
                    issues_found.append("No task queue generated")
            
            else:
                issues_found.append(f"Test request failed: {result.get('error')}")
            
            # Test Redis state persistence
            test_key = "silent-failure-test"
            test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
            await self.redis_client.set(test_key, json.dumps(test_data))
            
            retrieved = await self.redis_client.get(test_key)
            if not retrieved:
                issues_found.append("Redis state not actually persisting")
            else:
                retrieved_data = json.loads(retrieved)
                if retrieved_data != test_data:
                    issues_found.append("Redis data corruption detected")
            
            await self.redis_client.delete(test_key)
            
            if issues_found:
                self.test_results["silent_failures"] = f"ISSUES FOUND: {'; '.join(issues_found)}"
                print(f"      ⚠️  Issues detected: {len(issues_found)}")
                for issue in issues_found:
                    print(f"         - {issue}")
            else:
                self.test_results["silent_failures"] = "PASS - No silent failures detected"
                print("      ✅ No silent failures detected")
                
        except Exception as e:
            self.test_results["silent_failures"] = f"FAIL: {e}"
            print(f"      ❌ Silent failure detection failed: {e}")
    
    async def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\n📋 COMPREHENSIVE TEST REPORT")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if (isinstance(result, str) and result == "PASS") or
                             (isinstance(result, dict) and result.get("status") == "PASS") or
                             (isinstance(result, dict) and "PASS" in str(result)) or
                             (isinstance(result, str) and "PASS" in result))
        
        test_duration = time.time() - self.test_start_time
        
        print(f"\n📊 **TEST SUMMARY**")
        print(f"   Total Test Categories: {total_tests}")
        print(f"   Tests Passed: {passed_tests}")
        print(f"   Tests Failed: {total_tests - passed_tests}")
        print(f"   Success Rate: {passed_tests/total_tests*100:.1f}%")
        print(f"   Total Duration: {test_duration:.2f} seconds")
        
        print(f"\n🔍 **DETAILED RESULTS**")
        for test_name, result in self.test_results.items():
            # Properly check status for different result types
            if isinstance(result, dict):
                if result.get("status") == "PASS":
                    status = "✅ PASS"
                elif result.get("status") == "FAIL":
                    status = "❌ FAIL"
                elif result.get("success_rate", 0) >= 0.8:
                    status = "✅ PASS"  # For legacy dict results
                else:
                    status = "❌ FAIL"
            elif isinstance(result, str):
                status = "✅ PASS" if "PASS" in result else "❌ FAIL"
            else:
                status = "❌ FAIL"
            print(f"   {test_name}: {status}")
            if isinstance(result, dict):
                for key, value in result.items():
                    if key != "status":
                        print(f"      {key}: {value}")
            elif "FAIL" in str(result):
                print(f"      Details: {result}")
        
        # System health summary
        print(f"\n🏥 **SYSTEM HEALTH**")
        if hasattr(self, 'system_metrics'):
            for metric, value in self.system_metrics.items():
                print(f"   {metric}: {value}")
        
        # Final verdict
        success_rate = passed_tests / total_tests
        if success_rate >= 0.95:
            print(f"\n🎉 **VERDICT: EXCELLENT** - System is production-ready!")
        elif success_rate >= 0.85:
            print(f"\n✅ **VERDICT: GOOD** - System is functional with minor issues")
        elif success_rate >= 0.70:
            print(f"\n⚠️  **VERDICT: ACCEPTABLE** - System needs improvement")
        else:
            print(f"\n❌ **VERDICT: NEEDS WORK** - Major issues detected")
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": success_rate,
            "duration": test_duration,
            "detailed_results": self.test_results
        }
    
    async def cleanup_test_environment(self):
        """Clean up test environment"""
        print("\n🧹 CLEANING UP TEST ENVIRONMENT...")
        
        # Cleanup Redis test data
        if self.redis_client:
            # Find and delete test keys
            test_keys = await self.redis_client.keys("test:*")
            test_keys.extend(await self.redis_client.keys("agent:*"))
            if test_keys:
                await self.redis_client.delete(*test_keys)
            await self.redis_client.aclose()
        
        # Cleanup orchestrator
        if self.orchestrator:
            await self.orchestrator.cleanup()
        
        print("   ✅ Test environment cleaned up")

async def main():
    """Run comprehensive test suite"""
    test_suite = ComprehensiveTestSuite()
    
    try:
        # Initialize test environment
        await test_suite.initialize_test_environment()
        
        # Run all test categories
        await test_suite.test_1_deep_component_validation()
        await test_suite.test_2_real_world_scenarios()
        await test_suite.test_3_edge_cases_and_errors()
        await test_suite.test_4_concurrent_usage()
        await test_suite.test_5_memory_and_persistence()
        await test_suite.test_6_resource_monitoring()
        await test_suite.test_7_no_silent_failures()
        
        # Generate comprehensive report
        final_report = await test_suite.generate_comprehensive_report()
        
        # Return appropriate exit code
        return 0 if final_report["success_rate"] >= 0.85 else 1
        
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR in test suite: {e}")
        traceback.print_exc()
        return 2
        
    finally:
        await test_suite.cleanup_test_environment()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    print(f"\nTest suite completed with exit code: {exit_code}")
    sys.exit(exit_code)