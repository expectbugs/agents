#!/bin/bash
# Multi-Agent System Service Startup Script
# Run this after a reboot to start all required services

set -e  # Exit on any error

echo "🚀 Starting Multi-Agent System Services..."
echo "========================================"

# Function to check if a service is running
check_service() {
    local name=$1
    local test_command=$2
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo "✅ $name: OK"
        return 0
    else
        echo "❌ $name: FAILED"
        return 1
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local name=$1
    local test_command=$2
    local timeout=30
    local count=0
    
    echo "⏳ Waiting for $name to start..."
    while ! eval "$test_command" > /dev/null 2>&1; do
        sleep 1
        count=$((count + 1))
        if [ $count -ge $timeout ]; then
            echo "❌ $name failed to start within $timeout seconds"
            return 1
        fi
    done
    echo "✅ $name started successfully"
    return 0
}

# Kill any existing services to avoid conflicts
echo "🧹 Cleaning up any existing services..."
pkill -f "redis-server" || true
pkill -f "llama_cpp.server" || true  
pkill -f "searx/webapp.py" || true
pkill -f "npm run dev" || true
sleep 2

# 1. Start Redis
echo ""
echo "1️⃣ Starting Redis..."
sudo /etc/init.d/redis start
wait_for_service "Redis" "redis-cli ping"

# 2. Start LLM Server
echo ""
echo "2️⃣ Starting LLM Server..."
cd /home/user/agents
python -m llama_cpp.server \
    --model /home/user/models/Hermes-2-Pro-Mistral-10.7B-Q6_K/Hermes-2-Pro-Mistral-10.7B-Q6_K.gguf \
    --host 127.0.0.1 \
    --port 8000 \
    --n_gpu_layers -1 \
    --n_ctx 4096 \
    > /tmp/llama_server.log 2>&1 &

wait_for_service "LLM Server" "curl -s http://127.0.0.1:8000/v1/models"

# 3. Start SearXNG
echo ""
echo "3️⃣ Starting SearXNG..."
cd /home/user/agents/searxng
PYTHONPATH=/home/user/agents/searxng python searx/webapp.py \
    --host 127.0.0.1 \
    --port 8888 \
    > /tmp/searxng.log 2>&1 &

wait_for_service "SearXNG" "curl -s http://localhost:8888"

# 4. Start Perplexica
echo ""
echo "4️⃣ Starting Perplexica..."
cd /home/user/agents/Perplexica
nohup npm run dev > perplexica.log 2>&1 &

# Perplexica takes longer to start, so wait longer
echo "⏳ Waiting for Perplexica to start (this takes longer)..."
sleep 10

# Check which port Perplexica is using (3000 or 3001)
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Perplexica started on port 3000"
    PERPLEXICA_PORT=3000
elif curl -s http://localhost:3001 > /dev/null 2>&1; then
    echo "✅ Perplexica started on port 3001"
    PERPLEXICA_PORT=3001
else
    echo "❌ Perplexica failed to start on either port 3000 or 3001"
    echo "Check the logs in /home/user/agents/Perplexica/perplexica.log"
    exit 1
fi

# Update search agent configuration if needed
echo ""
echo "🔧 Updating search agent configuration..."
cd /home/user/agents
if [ "$PERPLEXICA_PORT" = "3000" ]; then
    sed -i 's|http://localhost:3001|http://localhost:3000|g' src/agents/perplexica_search_agent.py
elif [ "$PERPLEXICA_PORT" = "3001" ]; then
    sed -i 's|http://localhost:3000|http://localhost:3001|g' src/agents/perplexica_search_agent.py
fi

# 5. Final System Test
echo ""
echo "🧪 Running System Health Check..."
echo "================================="

failed_services=0

check_service "Redis" "redis-cli ping" || failed_services=$((failed_services + 1))
check_service "LLM Server" "curl -s http://127.0.0.1:8000/v1/models" || failed_services=$((failed_services + 1))
check_service "SearXNG" "curl -s http://localhost:8888" || failed_services=$((failed_services + 1))
check_service "Perplexica" "curl -s http://localhost:$PERPLEXICA_PORT" || failed_services=$((failed_services + 1))

echo ""
if [ $failed_services -eq 0 ]; then
    echo "🎉 ALL SERVICES STARTED SUCCESSFULLY!"
    echo "✅ System is ready for use"
    echo ""
    echo "Test the system with:"
    echo "cd /home/user/agents && python -m src research 'test query'"
    echo ""
    echo "Service URLs:"
    echo "  - Perplexica: http://localhost:$PERPLEXICA_PORT"
    echo "  - SearXNG: http://localhost:8888"
    echo "  - LLM Server: http://localhost:8000"
    echo "  - Redis: localhost:6379"
else
    echo "❌ $failed_services service(s) failed to start"
    echo "Check the logs:"
    echo "  - LLM Server: /tmp/llama_server.log"
    echo "  - SearXNG: /tmp/searxng.log"
    echo "  - Perplexica: /home/user/agents/Perplexica/perplexica.log"
    exit 1
fi