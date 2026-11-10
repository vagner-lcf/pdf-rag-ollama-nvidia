"""
Embeddings and LLM models for the RAG application and evaluation.
"""


from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings


# Load environment variables.
load_dotenv()

# Load nVidia models
def get_nvidia_models():
    """Initialize NVIDIA embeddings and LLM only when requested."""
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

    nvidia_embeddings = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b",
        truncate="END",
    )

    nvidia_llm = ChatNVIDIA(
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        temperature=0.2,
        max_completion_tokens=2048,
        model_kwargs={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
    )

    return nvidia_embeddings, nvidia_llm

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

ollama_llm_evaluator = ChatOllama(
    model="qwen3.5:9b",
    temperature=0,
    num_predict=512,
    reasoning=False,
    format="json",
)
