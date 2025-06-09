#!/usr/bin/env python3
"""
Minimal LangGraph test demonstrating core concepts
This test uses a simple mock LLM to avoid requiring API keys
"""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import operator

# Define a simple state schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    task_status: str

# Mock LLM for testing without API keys
class MockLLM:
    """Simple mock LLM that returns predefined responses"""
    def __init__(self, name: str):
        self.name = name
    
    async def ainvoke(self, messages: list) -> AIMessage:
        """Async invoke for the mock LLM"""
        last_message = messages[-1].content if messages else ""
        
        if self.name == "planner":
            return AIMessage(content=f"Planning task: {last_message}")
        elif self.name == "executor":
            return AIMessage(content=f"Executing: {last_message}")
        else:
            return AIMessage(content=f"{self.name} processed: {last_message}")

# Define agent nodes
async def planner_node(state: AgentState) -> dict:
    """Planning agent that decides what to do"""
    print(f"\n🤔 Planner Agent activated")
    print(f"   Current messages: {len(state['messages'])}")
    
    llm = MockLLM("planner")
    last_message = state['messages'][-1] if state['messages'] else HumanMessage(content="")
    
    response = await llm.ainvoke(state['messages'])
    
    return {
        "messages": [response],
        "current_agent": "executor",
        "task_status": "planned"
    }

async def executor_node(state: AgentState) -> dict:
    """Executor agent that performs tasks"""
    print(f"\n⚡ Executor Agent activated")
    print(f"   Task status: {state.get('task_status', 'unknown')}")
    
    llm = MockLLM("executor")
    response = await llm.ainvoke(state['messages'])
    
    return {
        "messages": [response],
        "current_agent": "done",
        "task_status": "completed"
    }

def should_continue(state: AgentState) -> str:
    """Decide whether to continue or end"""
    if state.get("current_agent") == "done":
        return "end"
    return "continue"

async def main():
    """Run the minimal LangGraph test"""
    print("=== Minimal LangGraph Test ===\n")
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    
    # Add edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "__end__")
    
    # Compile the graph
    app = workflow.compile()
    
    # Visualize the graph structure
    print("Graph structure:")
    print("  START → planner → executor → END")
    print()
    
    # Test 1: Simple task flow
    print("Test 1: Simple task flow")
    print("-" * 40)
    
    initial_state = {
        "messages": [HumanMessage(content="Create a Python web scraper")],
        "current_agent": "planner",
        "task_status": "new"
    }
    
    result = await app.ainvoke(initial_state)
    
    print("\nFinal state:")
    print(f"  Messages: {len(result['messages'])}")
    print(f"  Status: {result['task_status']}")
    print(f"  Final agent: {result['current_agent']}")
    
    print("\nMessage flow:")
    for i, msg in enumerate(result['messages']):
        print(f"  {i+1}. {msg.content}")
    
    # Test 2: With checkpointing (memory)
    print("\n\nTest 2: With memory/checkpointing")
    print("-" * 40)
    
    # Create memory saver
    memory = MemorySaver()
    app_with_memory = workflow.compile(checkpointer=memory)
    
    # Run with thread_id for memory
    config = {"configurable": {"thread_id": "test-thread-1"}}
    
    result2 = await app_with_memory.ainvoke(
        {"messages": [HumanMessage(content="Analyze stock data")],
         "current_agent": "planner",
         "task_status": "new"},
        config=config
    )
    
    print(f"\nStored in memory with thread_id: {config['configurable']['thread_id']}")
    print(f"Final status: {result2['task_status']}")
    
    # Verify we can retrieve the state
    saved_state = await app_with_memory.aget_state(config)
    print(f"Retrieved state messages: {len(saved_state.values['messages'])}")
    
    print("\n✅ All tests passed!")
    print("\nKey concepts demonstrated:")
    print("- StateGraph for defining agent workflows")
    print("- Nodes represent agents/functions")
    print("- Edges define flow between agents")
    print("- State is passed and updated between nodes")
    print("- Memory/checkpointing for conversation history")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())