# Setup Guide for External Dependencies

This guide explains how to set up the external services required for the LangGraph Multi-Agent System to function properly.

## Required External Services

The system depends on these external services that are **NOT** included in the repository:

1. **Perplexica** - AI-enhanced search interface
2. **SearXNG** - Privacy-focused metasearch engine
3. **Redis** - State management
4. **Qdrant** - Vector database
5. **Neo4j** - Graph database

## 1. Perplexica Setup

Perplexica provides AI-enhanced web search capabilities.

### Installation
```bash
# Clone Perplexica
git clone https://github.com/ItzCrazyKns/Perplexica.git
cd Perplexica

# Install dependencies
npm install

# Copy our working configuration
cp ../config/perplexica/config.toml .
cp ../config/perplexica/sample.config.toml .
cp -r ../config/perplexica/searxng .

# Set up environment
cp sample.config.toml config.toml
```

### Configuration
Edit `config.toml` to match your setup:

```toml
[GENERAL]
SIMILARITY_MEASURE = "cosine"
KEEP_ALIVE = "5m"

[MODELS.CUSTOM_OPENAI]
API_KEY = "not-needed"
API_URL = "http://127.0.0.1:8000/v1"  # Your llama.cpp server
MODEL_NAME = "hermes-2-pro-mistral"   # Your model name

[API_ENDPOINTS]
SEARXNG = "http://localhost:8888"     # SearXNG URL
```

### Start Perplexica
```bash
npm run dev
# Should be available at http://localhost:3000
```

## 2. SearXNG Setup

SearXNG provides privacy-focused metasearch capabilities.

### Installation
```bash
# Clone SearXNG
git clone https://github.com/searxng/searxng.git
cd searxng

# Copy our working configuration
cp ../config/searxng/settings.yml searx/
cp ../config/searxng/limiter.toml searx/

# Install dependencies
pip install -r requirements.txt
```

### Start SearXNG
```bash
python searx/webapp.py
# Should be available at http://localhost:8888
```

## 3. Redis Setup

### Installation (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### Installation (macOS)
```bash
brew install redis
brew services start redis
```

### Verify Redis
```bash
redis-cli ping
# Should return: PONG
```

## 4. Qdrant Setup

### Using Docker (Recommended)
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Using Binary
```bash
# Download and install from https://github.com/qdrant/qdrant/releases
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
./qdrant
```

### Verify Qdrant
```bash
curl http://localhost:6333/collections
# Should return: {"result":[],"status":"ok","time":0.0}
```

## 5. Neo4j Setup

### Using Docker (Recommended)
```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password123 \
    neo4j:latest
```

### Using Binary
```bash
# Download from https://neo4j.com/download/
# Follow installation instructions for your platform
```

### Verify Neo4j
- Web interface: http://localhost:7474
- Username: neo4j
- Password: password123 (or whatever you set)

## 6. LLM Server Setup (Optional)

For the planning agent to work with local LLMs:

### Using llama.cpp
```bash
# Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make

# Download a model (example: Hermes-2-Pro-Mistral)
# Place in /home/user/models/

# Start server
./server -m /path/to/your/model.gguf -p 8000
```

## 7. Verification

Once all services are running, verify the system:

```bash
# Test the multi-agent system
cd agents
python -m src research "test query"

# Test full demo with memory
python demo_research.py

# Test memory bridge
python src/memory_bridge.py
```

## Service URLs Summary

After setup, these services should be available:

- **Perplexica**: http://localhost:3000
- **SearXNG**: http://localhost:8888  
- **Redis**: localhost:6379
- **Qdrant**: http://localhost:6333
- **Neo4j**: bolt://localhost:7687 (Web: http://localhost:7474)
- **LLM Server**: http://localhost:8000 (if using llama.cpp)

## Troubleshooting

### Common Issues

1. **Port Conflicts**: Ensure no other services are using the required ports
2. **Permission Issues**: Make sure Redis and other services have proper permissions
3. **Memory Issues**: Qdrant and Neo4j may need sufficient RAM allocation
4. **Network Issues**: Ensure services can communicate on localhost

### Logs
- Check service logs for error messages
- Use `docker logs <container>` for Docker-based services
- Check the multi-agent system logs in `workspace/logs/`

## Alternative Configurations

### Using External APIs
Instead of local services, you can use external APIs by modifying:
- `config/perplexica/config.toml` - Add API keys for OpenAI, Anthropic, etc.
- `src/config.py` - Update service URLs for remote instances

### Docker Compose (Advanced)
Consider creating a `docker-compose.yml` to orchestrate all services together.

---

This setup ensures the LangGraph Multi-Agent System has all required dependencies and can function as designed.