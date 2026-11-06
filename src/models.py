"""
Embeddings and LLM models for the RAG application.
"""


from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_ollama import ChatOllama, OllamaEmbeddings


# Load environment variables and nVIDIA models
load_dotenv()

nvidia_embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b",
    truncate="END",
)

nvidia_llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0.2,
    max_completion_tokens=2048,
)

# Load Ollama models
ollama_embeddings = OllamaEmbeddings(
    model="bge-m3:567m",
)

ollama_rewriter_llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0.2,
    num_predict=512,
    reasoning=False,
)

ollama_llm = ChatOllama(
    model="qwen3.5:9b",
    temperature=0.2,
    num_predict=2048,
    reasoning=False,
)