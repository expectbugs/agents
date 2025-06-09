#!/usr/bin/env python3
"""Test LangGraph installation and basic functionality"""

try:
    # Test imports
    from langgraph.graph import StateGraph, MessagesState
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.memory import MemorySaver
    print("✓ LangGraph imports successful")
    
    # Test basic graph creation
    builder = StateGraph(MessagesState)
    print("✓ StateGraph creation successful")
    
    # Test memory saver
    memory = MemorySaver()
    print("✓ MemorySaver creation successful")
    
    # Print version info using pip
    import subprocess
    try:
        result = subprocess.run(['pip', 'show', 'langgraph'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                version = line.split(': ')[1]
                print(f"\nLangGraph version: {version}")
                break
    except:
        print("\nLangGraph version: Unable to determine")
    
    # List available components
    print("\nAvailable components:")
    print("- StateGraph: Graph-based agent orchestration")
    print("- MessagesState: Built-in state for message handling")
    print("- MemorySaver: In-memory checkpoint storage")
    print("- create_react_agent: Prebuilt ReAct agent factory")
    
    print("\n✅ All basic tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()