# Multi-Agent LLM System - CLAUDE.md

## Project Overview
A comprehensive locally-hosted LLM multi-agent multi-tool system leveraging LangChain, LangGraph, and related tools for orchestrating AI workflows on high-performance hardware.

## System Specifications

### Hardware
- **CPU**: 13th Gen Intel Core i7-13700KF (16 cores, 24 threads, up to 5.4 GHz)
- **GPU**: NVIDIA GeForce RTX 3090 (24GB VRAM)
- **RAM**: 32GB
- **Storage**: NVMe SSD (assumed based on modern system)

### Operating System
- **OS**: Gentoo Linux 2.17
- **Kernel**: 6.12.28-gentoo-dist
- **Init System**: OpenRC
- **Shell**: Bash
- **Python**: 3.12.10

### GPU Environment
- **NVIDIA Driver**: 575.57.08
- **CUDA Support**: Available for GPU acceleration
- **VRAM**: 24GB (optimal for running large language models)

## Infrastructure Services

### Database Systems
- **PostgreSQL 17**: Relational database for structured data
- **Neo4j**: Graph database for relationship mapping
- **Qdrant**: Vector database for embeddings and similarity search

### Network & System Services
- **NetworkManager**: Network connectivity management
- **OpenVPN**: Secure VPN connectivity
- **Tailscale**: Zero-config VPN mesh networking
- **SSH**: Remote access and management

## Development Environment

### Core Technologies
- **Python 3.12.10**: Primary development language
- **LangChain**: Framework for LLM application development
- **LangGraph**: Multi-agent orchestration and workflow management
- **Vector Stores**: Qdrant integration for semantic search
- **Graph Database**: Neo4j for complex relationship modeling

### System Capabilities
- **Local LLM Hosting**: RTX 3090 enables running models up to ~13B parameters efficiently
- **Multi-Agent Architecture**: LangGraph for coordinating multiple AI agents
- **Tool Integration**: LangChain tools for external API calls and system interactions
- **Persistent Storage**: PostgreSQL for state management, Qdrant for embeddings
- **Scalability**: 24-thread CPU for parallel processing

## Project Goals

### Primary Objectives
1. **Local LLM Deployment**: Host and run language models locally without cloud dependencies
2. **Multi-Agent System**: Implement collaborative AI agents with specialized roles
3. **Tool Integration**: Connect agents to various tools and APIs
4. **Knowledge Management**: Utilize vector and graph databases for context retention
5. **Workflow Automation**: Create complex AI-driven workflows using LangGraph

### Architecture Components
- **Agent Orchestrator**: Central coordinator managing agent interactions
- **Specialized Agents**: Task-specific agents (research, code generation, analysis, etc.)
- **Tool Registry**: Centralized management of available tools and APIs
- **Memory Systems**: Short-term (conversation) and long-term (vector/graph DB) memory
- **Execution Environment**: Sandboxed environments for code execution

### Integration Points
- **LangChain**: Core framework for LLM interactions and tool usage
- **LangGraph**: State management and multi-agent coordination
- **Qdrant**: Semantic search and retrieval-augmented generation (RAG)
- **Neo4j**: Complex relationship mapping between entities
- **PostgreSQL**: Structured data storage and transaction management

## Development Guidelines

### Best Practices
- **Modular Design**: Keep agents and tools as independent modules
- **Type Safety**: Use Python type hints throughout
- **Error Handling**: Implement robust error handling and fallbacks
- **Logging**: Comprehensive logging for debugging multi-agent interactions
- **Testing**: Unit tests for individual components, integration tests for workflows

### Performance Considerations
- **GPU Utilization**: Optimize model loading and inference for RTX 3090
- **Memory Management**: Monitor VRAM usage, implement model unloading when needed
- **Concurrency**: Leverage multi-threading for parallel agent execution
- **Caching**: Implement embedding and response caching strategies

### Security
- **Local Execution**: All processing happens on-premises
- **Network Isolation**: Use VPN connections for any external communications
- **Access Control**: Implement agent-level permissions and capabilities
- **Data Privacy**: Ensure sensitive data remains within local infrastructure
- Rule 1: NEVER disable or remove a feature to fix a bug or error.
- Rule 2: NEVER fix an error or bug by hiding it.
- Rule 3: NO silent fallbacks or silent failures, all problems should be loud and proud.
- Rule 4: Always check online documentation of every packaged used and do everything the officially recommended way.
- Rule 5: Clean up your mess.  Remove any temporary and/or outdated files or scripts that were only meant to be used once and no longer serve a purpose.
