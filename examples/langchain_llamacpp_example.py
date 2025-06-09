#!/usr/bin/env python3
"""
LangChain with llama.cpp Example
Using local models with GPU acceleration
"""

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseOutputParser
import json

# Callback for streaming output
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Initialize LlamaCpp
# You'll need to download a GGUF model file first
# Example: wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

def create_llama_model(model_path: str):
    """Create a LlamaCpp model with GPU acceleration"""
    return LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,  # Use all GPU layers
        n_ctx=4096,       # Context window
        n_batch=512,      # Batch size for prompt processing
        temperature=0.7,
        max_tokens=512,
        top_p=0.95,
        callback_manager=callback_manager,
        verbose=True,     # Print stats
        # GPU-specific settings
        f16_kv=True,      # Use half precision for key/value cache
        use_mlock=True,   # Lock model in RAM
    )

# Example 1: Basic completion
def basic_example(llm):
    print("\n=== Basic Completion Example ===")
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in simple terms:"
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    # Uncomment to run:
    # result = chain.invoke({"topic": "neural networks"})
    print("Chain created for basic completions")

# Example 2: Conversation with memory
def conversation_example(llm):
    print("\n=== Conversation with Memory ===")
    template = """The following is a conversation between a human and an AI assistant.

Current conversation:
{history}
Human: {input}
AI:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    memory = ConversationBufferMemory()
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )
    
    # Example conversation flow:
    # response1 = conversation.invoke({"input": "Hello, what can you help with?"})
    # response2 = conversation.invoke({"input": "Can you remember what I just asked?"})
    print("Conversation chain with memory created")

# Example 3: Structured output with JSON
class JSONOutputParser(BaseOutputParser):
    """Parse LLM output as JSON"""
    
    def parse(self, text: str) -> dict:
        # Extract JSON from the response
        try:
            # Find JSON between markers or parse directly
            json_str = text.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            return json.loads(json_str)
        except:
            return {"error": "Failed to parse JSON", "raw": text}

def structured_output_example(llm):
    print("\n=== Structured Output Example ===")
    prompt = PromptTemplate(
        input_variables=["task"],
        template="""Generate a JSON object for the following task.
Return only valid JSON, no other text.

Task: {task}

JSON format:
{{
    "task_name": "string",
    "priority": "high/medium/low",
    "steps": ["step1", "step2", ...],
    "estimated_time": "string"
}}

JSON:"""
    )
    
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=JSONOutputParser()
    )
    
    # Example usage:
    # result = chain.invoke({"task": "Set up a Python web server"})
    # print(json.dumps(result, indent=2))
    print("Structured output chain created")

# Example 4: RAG-ready chain
def rag_example(llm):
    print("\n=== RAG-Ready Chain ===")
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""Use the following context to answer the question.
If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # In a real RAG system, context would come from vector store
    # Example:
    # context = retriever.get_relevant_documents(question)
    # result = chain.invoke({"context": context, "question": question})
    print("RAG chain template created")

# Model recommendations for RTX 3090 (24GB VRAM)
print("=== llama.cpp with LangChain Setup ===")
print("\nRecommended models for RTX 3090:")
print("1. Llama 2 13B (Q4_K_M): ~7GB - Fast and capable")
print("2. Mixtral 8x7B (Q4_K_M): ~25GB - Requires careful memory management")
print("3. Llama 3 8B (Q6_K): ~6GB - Latest architecture")
print("4. Mistral 7B (Q8): ~7GB - High quality")

print("\nTo download a model:")
print("wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf")

print("\nUsage example:")
print("""
# Create model
llm = create_llama_model("./llama-2-13b-chat.Q4_K_M.gguf")

# Use in chains
basic_example(llm)
conversation_example(llm)
structured_output_example(llm)
rag_example(llm)
""")

print("\nAdvantages over Ollama:")
print("- Direct control over GPU layers and memory")
print("- Better performance tuning options")
print("- No separate server process needed")
print("- Full integration with LangChain streaming")
print("- Support for custom quantization formats")