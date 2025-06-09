# LangGraph Multi-Agent System

A production-ready multi-agent orchestration system built with LangGraph and LangChain for complex AI workflows with persistent memory.

## Features

- 🤖 **Multi-Agent Orchestration**: LangGraph-powered agent coordination
- 🔍 **AI-Enhanced Web Search**: Perplexica integration for intelligent research
- 🧠 **Persistent Memory**: Qdrant + Neo4j integration via Buddy memory system
- 💾 **State Management**: Redis-backed distributed state handling
- 🚀 **Async Architecture**: Full async/await support for high performance
- 🛡️ **Security**: Input validation, sanitization, and secure configuration

## Quick Start

### Prerequisites
- Python 3.12+
- Redis server running
- Node.js (for Perplexica)

### Installation
```bash
git clone <repository>
cd agents

# Install Python dependencies
pip install -r requirements.txt

# Install additional memory system dependencies
pip install mem0ai[graph] qdrant-client neo4j

# Set up Perplexica (in separate terminal)
cd Perplexica
npm install && npm run dev
```

### Basic Usage
```bash
# Simple research workflow
python -m src research "AI safety regulations 2024"

# Full demo with memory integration  
python demo_research.py
```

### Programmatic Usage
```python
import asyncio
from src.orchestrator import MultiAgentOrchestrator
from src.agents.perplexica_search_agent import PerplexicaSearchAgent
from src.agents.planning_agent import PlanningAgent

async def main():
    orchestrator = MultiAgentOrchestrator(use_memory=True)
    await orchestrator.initialize()
    
    orchestrator.add_agent("web_search", PerplexicaSearchAgent())
    orchestrator.add_agent("planner", PlanningAgent())
    orchestrator.build_graph()
    
    result = await orchestrator.process_request(
        "search for quantum computing breakthroughs",
        thread_id="research-session"
    )
    
    print(result["messages"][-1].content)
    await orchestrator.cleanup()

asyncio.run(main())
```

## Architecture

### Core Components
- **Orchestrator**: Central coordinator using LangGraph StateGraph
- **Planning Agent**: Task decomposition and strategy planning
- **Perplexica Search Agent**: AI-enhanced web search and research
- **Execution Agent**: Concrete task execution and operations
- **Memory Bridge**: Integration with persistent memory systems

### Technology Stack
- **Framework**: LangGraph + LangChain 0.3.25
- **Memory**: mem0ai with Qdrant (vector) + Neo4j (graph)
- **State Store**: Redis with async connection pooling
- **Search**: Perplexica AI-powered search interface
- **Language**: Python 3.12 with full async/await

## Configuration

### Environment Variables
```bash
# Redis Configuration
export REDIS_URL="redis://localhost:6379"
export REDIS_PASSWORD="your_password"  # optional

# LLM Configuration (for planning agent)
export MODEL_PATH="/path/to/your/model.gguf"
export GPU_LAYERS=-1

# Security Settings
export MAX_INPUT_LENGTH=10000
export LOG_LEVEL=INFO
```

### Required Services
1. **Redis**: For distributed state management
2. **Perplexica**: AI search interface (http://localhost:3000)
3. **Qdrant**: Vector database (http://localhost:6333)
4. **Neo4j**: Graph database (bolt://localhost:7687)

## Project Structure
```
agents/
├── src/                     # Core system
│   ├── orchestrator.py     # Multi-agent coordinator
│   ├── agents/             # Agent implementations
│   ├── memory_bridge.py    # Memory system integration
│   └── config.py           # Configuration management
├── demo_research.py        # Complete workflow demo
├── Perplexica/            # AI search interface
└── docs/                  # Documentation
```

## Testing
```bash
# Test CLI interface
python -m src research "test query"

# Test full demo with memory
python demo_research.py

# Test memory bridge independently  
python src/memory_bridge.py
```

## Development

### Adding New Agents
```python
from src.orchestrator import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__("my_agent")
    
    async def process(self, state):
        # Your agent logic here
        return {
            "messages": [AIMessage(content="Task completed")],
            "current_agent": "orchestrator"
        }

# Register with orchestrator
orchestrator.add_agent("my_agent", MyAgent())
```

### Error Handling
The system follows strict error handling principles:
- No silent failures - all errors are logged loudly
- Comprehensive error context and troubleshooting information
- Proper resource cleanup in all scenarios

## Dependencies

### Core Requirements
- `langgraph` - Multi-agent orchestration
- `langchain>=0.3.25` - LLM framework
- `redis` - State management
- `aiohttp` - HTTP client for API calls

### Memory System
- `mem0ai[graph]>=0.1.106` - Memory management with graph support
- `qdrant-client>=1.7.0` - Vector database client
- `neo4j>=5.0.0` - Graph database driver

### Optional
- `llama-cpp-python` - Local LLM inference
- `sentence-transformers` - Text embeddings

## License

MIT License - See LICENSE file for details.

## Status

**Version**: 0.0.1  
**Status**: Production Ready  
**Test Coverage**: All core functionality tested and working  

The system provides a solid foundation for building complex multi-agent AI workflows with persistent memory and intelligent web search capabilities.