#!/usr/bin/env python3
"""Test Redis connection and basic operations"""

import redis
import redis.asyncio as aioredis
import asyncio
import json
from datetime import datetime

def test_sync_redis():
    """Test synchronous Redis operations"""
    print("Testing synchronous Redis connection...")
    
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Test basic operations
        r.set('test_key', 'Hello from sync Redis!')
        value = r.get('test_key')
        print(f"✓ Set/Get test: {value}")
        
        # Test JSON storage (for agent states)
        agent_state = {
            'agent_id': 'test_agent',
            'timestamp': datetime.now().isoformat(),
            'status': 'active',
            'memory': {'last_action': 'testing'}
        }
        r.set('agent:test_agent:state', json.dumps(agent_state))
        retrieved_state = json.loads(r.get('agent:test_agent:state'))
        print(f"✓ JSON state storage: {retrieved_state['agent_id']}")
        
        # Test hash operations (for structured data)
        r.hset('agent:config', mapping={
            'model': 'llama-2-7b',
            'temperature': '0.7',
            'max_tokens': '512'
        })
        config = r.hgetall('agent:config')
        print(f"✓ Hash operations: {config}")
        
        # Test pub/sub (for inter-agent communication)
        # Note: Full pub/sub test requires separate process
        
        # Cleanup
        r.delete('test_key', 'agent:test_agent:state', 'agent:config')
        print("✓ Cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Sync Redis error: {e}")
        return False

async def test_async_redis():
    """Test asynchronous Redis operations"""
    print("\nTesting asynchronous Redis connection...")
    
    try:
        # Connect to Redis
        redis_client = await aioredis.from_url("redis://localhost", decode_responses=True)
        
        # Test basic async operations
        await redis_client.set('async_test_key', 'Hello from async Redis!')
        value = await redis_client.get('async_test_key')
        print(f"✓ Async Set/Get test: {value}")
        
        # Test pipeline for bulk operations
        async with redis_client.pipeline(transaction=True) as pipe:
            pipe.set('agent:1:status', 'planning')
            pipe.set('agent:2:status', 'executing')
            pipe.set('agent:3:status', 'idle')
            results = await pipe.execute()
        print(f"✓ Pipeline operations: {len(results)} commands executed")
        
        # Test expiring keys (for temporary states)
        await redis_client.setex('temp_state', 60, 'This expires in 60 seconds')
        ttl = await redis_client.ttl('temp_state')
        print(f"✓ Expiring key test: TTL = {ttl} seconds")
        
        # Cleanup
        await redis_client.delete('async_test_key', 'agent:1:status', 
                          'agent:2:status', 'agent:3:status', 'temp_state')
        print("✓ Async cleanup successful")
        
        await redis_client.close()
        return True
        
    except Exception as e:
        print(f"❌ Async Redis error: {e}")
        return False

def main():
    """Run all Redis tests"""
    print("=== Redis Connection Tests ===\n")
    
    # Check Redis server info
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        info = r.info()
        print(f"Redis Server Version: {info['redis_version']}")
        print(f"Memory Used: {info['used_memory_human']}")
        print(f"Connected Clients: {info['connected_clients']}")
        print()
    except Exception as e:
        print(f"❌ Cannot connect to Redis: {e}")
        print("Make sure Redis is running: sudo rc-service redis start")
        return
    
    # Run sync tests
    sync_success = test_sync_redis()
    
    # Run async tests
    async_success = asyncio.run(test_async_redis())
    
    # Summary
    print("\n=== Test Summary ===")
    if sync_success and async_success:
        print("✅ All Redis tests passed!")
        print("\nRedis is ready for multi-agent state management:")
        print("- Fast in-memory storage for agent states")
        print("- Pub/Sub for inter-agent communication")
        print("- Atomic operations for concurrent access")
        print("- Expiring keys for temporary states")
    else:
        print("❌ Some tests failed. Please check Redis configuration.")

if __name__ == "__main__":
    main()