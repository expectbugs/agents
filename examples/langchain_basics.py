#!/usr/bin/env python3
"""
LangChain Basics - Core Concepts Demonstration
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
# Deprecated: from langchain.memory import ConversationBufferMemory
# Deprecated: from langchain.chains import ConversationChain
from langchain.tools import Tool
# Deprecated: from langchain.agents import initialize_agent, AgentType

# Note: Memory and agent classes have been deprecated in LangChain v0.3
# Use LangGraph for state management and agent orchestration instead

# 1. BASIC PROMPT TEMPLATE
print("=== 1. Basic Prompt Template ===")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("user", "{input}")
])

# Show how templates work
formatted = prompt.format_messages(input="What is LangChain?")
print(f"Formatted prompt: {formatted[0].content}")
print(f"User message: {formatted[1].content}\n")

# 2. CHAINS - Composing components
print("=== 2. Chains (LCEL - LangChain Expression Language) ===")
# Note: You'll need a local LLM like Ollama installed
# For now, we'll show the chain structure

# Example chain structure (requires local LLM):
# llm = ChatOllama(model="llama3.2")  # or any model you have
# chain = prompt | llm | StrOutputParser()
# response = chain.invoke({"input": "Explain chains in one sentence"})

print("Chain structure: prompt | llm | output_parser")
print("This creates a pipeline that processes input through each component\n")

# 3. MEMORY - Conversation State (DEPRECATED in LangChain v0.3)
print("=== 3. Memory - Conversation History (DEPRECATED) ===")
print("Note: ConversationBufferMemory is deprecated in LangChain v0.3")
print("Use LangGraph MessagesState for conversation memory instead")
print("Example conversation memory would be stored in MessagesState:")
print("  messages = [")
print("    HumanMessage(content='Hi, I\'m learning LangChain'),")
print("    AIMessage(content='Hello! LangChain is great for building LLM apps.')")
print("  ]")
print()

# 4. TOOLS - External capabilities
print("=== 4. Tools - External Functions ===")

def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

# Create a tool from the function
word_length_tool = Tool(
    name="WordLength",
    func=get_word_length,
    description="Get the length of a word"
)

# Test the tool
result = word_length_tool.run("LangChain")
print(f"Testing tool - Length of 'LangChain': {result}")
print()

# 5. DOCUMENT LOADERS AND VECTOR STORES
print("=== 5. Document Processing (RAG) ===")
print("Key components for RAG (Retrieval Augmented Generation):")
print("- Document Loaders: Load data from various sources")
print("- Text Splitters: Chunk documents for processing")
print("- Embeddings: Convert text to vectors")
print("- Vector Stores: Store and search embeddings")
print("- Retrievers: Find relevant documents\n")

# 6. AGENTS - Dynamic Decision Making (DEPRECATED in LangChain v0.3)
print("=== 6. Agents - Dynamic Decision Making (DEPRECATED) ===")
print("Note: initialize_agent and AgentType are deprecated in LangChain v0.3")
print("Use LangGraph for building multi-agent systems instead")
print("LangGraph agents can:")
print("- Decide which tools to use")
print("- Plan multi-step solutions")
print("- Handle complex queries")
print("- Reason about problems")
print("- Maintain conversation state")
print("- Coordinate with other agents\n")
print("Example LangGraph agent structure:")
print("from langgraph.prebuilt import create_react_agent")
print("agent = create_react_agent(llm, tools=[word_length_tool])")

print("=== Key Concepts Summary ===")
print("""
1. **Prompts**: Templates for structured LLM input
2. **Models**: LLM interfaces (OpenAI, Ollama, etc.)
3. **Chains**: Compose components into pipelines
4. **Memory**: Maintain conversation state
5. **Tools**: Give LLMs access to external functions
6. **Agents**: LLMs that can use tools dynamically
7. **RAG**: Retrieve context from documents
8. **LCEL**: LangChain Expression Language for composing

To run with a real LLM:
1. Install Ollama: https://ollama.ai
2. Pull a model: ollama pull llama3.2
3. Use LangGraph for modern agent systems

Migration Notes:
- Replace ConversationBufferMemory with LangGraph MessagesState
- Replace initialize_agent with LangGraph agent patterns
- Use StateGraph for workflow orchestration
""")