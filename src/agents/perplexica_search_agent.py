#!/usr/bin/env python3
"""
Perplexica Search Agent - Uses the working Perplexica API for AI-enhanced web search
This replaces the unnecessary WebSearchAgent workaround
"""

from typing import Dict, Any, List
import aiohttp
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from datetime import datetime

try:
    from ..orchestrator import BaseAgent, MultiAgentState
    from ..logging_config import get_logger
except ImportError:
    from orchestrator import BaseAgent, MultiAgentState
    from logging_config import get_logger

logger = get_logger(__name__)

class PerplexicaSearchAgent(BaseAgent):
    """Agent that uses Perplexica for AI-enhanced web search"""
    
    def __init__(self):
        super().__init__("perplexica_search")
        self.perplexica_url = "http://localhost:3000"
        
    async def process(self, state: MultiAgentState) -> Dict[str, Any]:
        """Process search request using Perplexica"""
        logger.info("Perplexica search agent processing...")
        
        # Extract query from messages
        query = self._extract_query(state.get("messages", []))
        if not query:
            return {
                "messages": [AIMessage(content="❌ **Search Error**\n\nNo search query found in request.")],
                "current_agent": "orchestrator",
                "error_count": state.get("error_count", 0) + 1
            }
        
        # Call Perplexica
        results = await self._search_via_perplexica(query)
        
        # Format response
        if results:
            response = self._format_results(query, results)
            
            # Save search state
            await self.save_state({
                "last_query": query,
                "results_count": len(results),
                "timestamp": datetime.now().isoformat(),
                "backend": "perplexica"
            })
            
            return {
                "messages": [AIMessage(content=response)],
                "current_agent": "orchestrator",
                "context": {
                    **state.get("context", {}),
                    "search_results": results,
                    "last_search": {
                        "query": query,
                        "results": results[:5],  # Store top 5 results
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }
        else:
            return {
                "messages": [AIMessage(content=f"🔍 **Search Complete**\n\nNo results found for: `{query}`\n\nTry rephrasing your search query.")],
                "current_agent": "orchestrator"
            }
    
    def _extract_query(self, messages: List) -> str:
        """Extract search query from messages"""
        for msg in reversed(messages):
            content = msg.content if hasattr(msg, 'content') else str(msg)
            
            # Look for search patterns
            if "search for" in content.lower():
                return content.lower().split("search for", 1)[1].strip()
            elif "research" in content.lower():
                # Extract everything after "research"
                parts = content.lower().split("research", 1)
                if len(parts) > 1:
                    return parts[1].strip()
            elif "find information about" in content.lower():
                return content.lower().split("find information about", 1)[1].strip()
            elif "look up" in content.lower():
                return content.lower().split("look up", 1)[1].strip()
            
            # If it's a short message without action words, treat as query
            if len(content) < 100 and not any(word in content.lower() for word in ["create", "build", "implement", "execute"]):
                return content.strip()
        
        return ""
    
    async def _search_via_perplexica(self, query: str) -> List[Dict[str, Any]]:
        """Search using Perplexica API which provides AI-enhanced results"""
        search_url = f"{self.perplexica_url}/api/search"
        payload = {
            "chatModel": {
                "provider": "custom_openai",
                "name": "hermes-2-pro-mistral"
            },
            "focusMode": "webSearch",
            "query": query,
            "optimizationMode": "speed"
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(search_url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Perplexica returned response for query: {query}")
                        
                        # Perplexica returns: {"message": "AI response", "sources": [...]}
                        results = []
                        
                        # Add the AI-generated summary as first result
                        if data.get("message"):
                            results.append({
                                "title": "AI Summary",
                                "url": "",
                                "snippet": data["message"][:500],
                                "source": "perplexica_ai"
                            })
                        
                        # Add individual sources
                        for source in data.get("sources", []):
                            metadata = source.get("metadata", {})
                            results.append({
                                "title": metadata.get("title", "Unknown Title"),
                                "url": metadata.get("url", ""),
                                "snippet": source.get("pageContent", "")[:300],
                                "source": "perplexica"
                            })
                        
                        return results
                    else:
                        logger.error(f"Perplexica API returned status {response.status}")
                        return []
                        
        except asyncio.TimeoutError:
            logger.error(f"Perplexica search timed out for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Perplexica search failed: {e}")
            return []
    
    def _format_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Format search results for display"""
        response = f"🔍 **Web Search Results** (Perplexica AI)\n\n"
        response += f"**Query:** `{query}`\n"
        response += f"**Found:** {len(results)} enhanced results\n\n"
        
        for i, result in enumerate(results[:7], 1):  # Show top 7 results
            title = result.get("title", "No title")[:60]
            url = result.get("url", "")
            snippet = result.get("snippet", "No description available")[:200]
            source_type = result.get("source", "unknown")
            
            # Different formatting for AI summary vs sources
            if source_type == "perplexica_ai":
                response += f"🤖 **AI Analysis:**\n{snippet}...\n\n"
            else:
                response += f"**{i-1}. {title}**\n"
                if url:
                    response += f"🔗 {url}\n"
                if snippet:
                    response += f"📄 {snippet}...\n"
                response += "\n"
        
        response += "💡 *This search used AI to analyze and summarize web results for better insights.*"
        
        return response


# Test function for development
async def test_perplexica_agent():
    """Test the Perplexica search agent"""
    agent = PerplexicaSearchAgent()
    await agent.initialize()
    
    # Test search
    test_state = MultiAgentState(
        messages=[HumanMessage(content="search for AI safety regulations 2024")],
        current_agent="perplexica_search",
        current_model="",
        gpu_memory={},
        context={},
        task_queue=[],
        agent_states={},
        error_count=0,
        metadata={}
    )
    
    result = await agent.process(test_state)
    
    print("Perplexica Search Result:")
    print(result['messages'][0].content)
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(test_perplexica_agent())