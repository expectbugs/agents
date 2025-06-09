#!/usr/bin/env python3
"""
Simple CLI entry point for LangGraph agents
Usage: python -m agents research "your query here"
"""

import asyncio
import sys
import os

from .orchestrator import MultiAgentOrchestrator
from .agents.perplexica_search_agent import PerplexicaSearchAgent  # Use new agent!
from .agents.planning_agent import PlanningAgent
from .agents.execution_agent import ExecutionAgent

async def run_research_workflow(query: str):
    """Execute a research workflow with web search and analysis"""
    print(f"\n🔍 Starting research workflow for: '{query}'")
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(use_memory=True)
    await orchestrator.initialize()
    
    # Create and initialize agents
    web_search_agent = PerplexicaSearchAgent()
    await web_search_agent.initialize()
    
    planning_agent = PlanningAgent()
    await planning_agent.initialize()
    
    execution_agent = ExecutionAgent()
    await execution_agent.initialize()
    
    # Add agents
    orchestrator.add_agent("web_search", web_search_agent)  # New Perplexica agent
    orchestrator.add_agent("planner", planning_agent)
    orchestrator.add_agent("executor", execution_agent)
    
    # Build the graph
    orchestrator.build_graph()
    
    # Process the research request
    result = await orchestrator.process_request(
        f"search for {query}",
        thread_id="research-session"
    )
    
    print("\n✅ Research Complete!")
    print(f"Success: {result.get('success', False)}")
    
    # Display results
    if result.get('success'):
        messages = result.get('messages', [])
        for i, msg in enumerate(messages):
            content = msg.content if hasattr(msg, 'content') else str(msg)
            print(f"\n--- Message {i+1} ---")
            print(content)
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    await orchestrator.cleanup()

def main():
    if len(sys.argv) < 3 or sys.argv[1] != "research":
        print("Usage: python -m agents research 'your query'")
        print("Example: python -m agents research 'AI safety regulations 2024'")
        sys.exit(1)
    
    query = " ".join(sys.argv[2:])
    asyncio.run(run_research_workflow(query))

if __name__ == "__main__":
    main()