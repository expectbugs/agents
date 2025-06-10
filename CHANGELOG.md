# Changelog

All notable changes to the LangGraph Multi-Agent System will be documented in this file.

## [0.0.2] - 2025-06-10

### Fixed
- **Critical Backend Connectivity**: Fixed post-reboot service dependencies
- **SearXNG Integration**: Resolved module import and secret key configuration issues
- **Perplexica Port Detection**: Fixed automatic port detection and configuration
- **Silent Fallbacks**: Added clear error messages when SearXNG backend fails
- **Service Dependencies**: Fixed Redis, LLM server, and search backend startup sequence

### Added
- **Automated Startup Script**: `/home/user/agents/start_services.sh` for post-reboot initialization
- **Service Health Monitoring**: Comprehensive service status checking and reporting
- **Error Visibility**: Clear backend failure messages following "no silent failures" principle
- **Timeout Optimization**: Increased Perplexica timeout for AI processing (30→60s)

### Improved
- **Performance**: Reduced search response time from 60s timeouts to 1.5s success
- **Reliability**: All critical backends (Redis, SearXNG, Perplexica, LLM server) working
- **User Experience**: Clear service URLs and startup guidance in automated script
- **Error Handling**: Loud error reporting instead of silent backend failures

### Technical
- **LLM Context**: Increased from 2048 to 4096 tokens for better AI responses
- **Port Management**: Automatic detection of Perplexica port (3000/3001)
- **Configuration**: Fixed SearXNG secret key from default "ultrasecretkey"
- **Dependencies**: Proper PYTHONPATH setup for SearXNG module imports

## [0.0.1] - 2025-06-09

### Added
- Initial release of the LangGraph multi-agent orchestration system
- Core multi-agent architecture with LangGraph integration
- Perplexica-powered AI-enhanced web search agent
- Planning agent for task decomposition and strategy
- Execution agent for task completion
- Memory bridge integration with Buddy's persistent memory system (Qdrant + Neo4j)
- Professional CLI interface: `python -m src research "query"`
- Comprehensive demo script showcasing full workflow
- Redis-based state management and agent coordination
- Flexible import system supporting both package and standalone usage
- Comprehensive error handling and logging infrastructure

### Features
- 🔍 AI-enhanced web search via Perplexica integration
- 🧠 Persistent memory storage with graph database support
- 🤖 Multi-agent orchestration with intelligent routing
- 💾 Cross-workflow context awareness and memory retrieval
- 🚀 Asynchronous agent execution with proper cleanup
- 📊 Health monitoring and system status reporting

### Technical Details
- Built on LangChain 0.3.25 and LangGraph
- Supports mem0ai with graph database integration
- Python 3.12+ with full async/await support
- Redis for distributed state management
- Modular agent architecture for easy extension

### Security
- Input validation and sanitization
- Secure Redis configuration
- No hardcoded credentials
- Proper error handling without information leakage

### Known Issues
- None in this release

### Dependencies
- langchain>=0.3.25
- langgraph
- mem0ai[graph]>=0.1.106
- redis
- aiohttp
- Additional dependencies in requirements.txt

---

This is the first public release of the multi-agent system, providing a foundation for building complex AI workflows with persistent memory and web search capabilities.