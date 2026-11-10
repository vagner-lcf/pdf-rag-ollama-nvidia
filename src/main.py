from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from doc_creator import read_pdfs_and_convert_to_documents
from question import answer_questions
from models import get_nvidia_models, ollama_embeddings, ollama_llm, ollama_rewriter_llm
from hash_manifest import needs_reindex, save_manifest


def build_pipeline(provider: str):
    """
    Build the document retrieval and question-answering pipeline based on the selected provider.
    """
    # ---------------------------------------------------------
    # Create the data directory if it doesn't exist and define the paths for the vector store persistence
    # ---------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    
    data_dir = project_root / "data" / "documents"
    data_dir.mkdir(parents=True, exist_ok=True)

    nvidia_persist_directory = str(project_root / "data" / "chroma_nvidia")
    ollama_persist_directory = str(project_root / "data" / "chroma_ollama")

    # ---------------------------------------------------------
    # Select embeddings and persist directory based on the provider
    # ---------------------------------------------------------
    if provider == "nvidia":
        nvidia_embeddings, nvidia_llm = get_nvidia_models()
        embeddings = nvidia_embeddings
        persist_directory = nvidia_persist_directory

    elif provider == "ollama":
        embeddings = ollama_embeddings
        persist_directory = ollama_persist_directory

    else:
        raise ValueError(f"Provider '{provider}' não suportado.")

    # ---------------------------------------------------------
    # Initialize vector store
    # ---------------------------------------------------------
    vectorstore = Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # ---------------------------------------------------------
    # Check whether indexing is required
    # ---------------------------------------------------------
    index_needed = needs_reindex(data_dir, persist_directory)

    # ---------------------------------------------------------
    # Also check if the vector store is empty (no documents indexed)
    # ---------------------------------------------------------
    index_empty = not vectorstore.get(limit=1)["ids"]

    if index_needed or index_empty:
        print("Indexando documentos...")

        # ---------------------------------------------------------
        # Read PDF files and convert them into documents
        # ---------------------------------------------------------
        pdf_files = sorted(data_dir.glob("*.pdf"))
        documents = read_pdfs_and_convert_to_documents(pdf_files)

        # ---------------------------------------------------------
        # Split documents into smaller chunks for better retrieval
        # ---------------------------------------------------------
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100,
        )
        parts = splitter.split_documents(documents)

        if not parts:
            raise ValueError("Nenhum texto extraível encontrado nos PDFs.")

        # Invalidate the manifest before rebuilding the collection
        manifest_path = Path(persist_directory) / "hash_manifest.json"
        manifest_path.unlink(missing_ok=True)

        # Remove old collection
        vectorstore.delete_collection()

        # Create a fresh collection
        vectorstore = Chroma(
            collection_name="documents",
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

        vectorstore.add_documents(parts)

        # Save only after successful indexing
        save_manifest(data_dir, persist_directory)

        print("Indexação concluída.")

    # ---------------------------------------------------------
    # Create a retriever for semantic search
    # ---------------------------------------------------------
    retriever = vectorstore.as_retriever(search_kwargs={"k":4})

    # ---------------------------------------------------------
    # Prompt template for rewriting user queries for semantic search
    # ---------------------------------------------------------
    REWRITER_PROMPT_TEMPLATE = """
    Você é um especialista em reformulação de consultas para busca semântica
    em documentos armazenados em um banco vetorial.

    Sua tarefa é reescrever a pergunta do usuário para facilitar a recuperação
    de trechos relevantes dos documentos.

    Regras:
    - Preserve a intenção e o significado original da pergunta.
    - Mantenha nomes próprios, datas, números e termos técnicos relevantes.
    - Melhore a clareza e a especificidade quando necessário.
    - Não invente informações nem acrescente suposições.
    - Retorne uma única consulta em texto simples, sem responder à pergunta, reproduzir instruções ou acrescentar comentários.
    - Não responda à pergunta.
    - Não amplie o escopo da pergunta nem acrescente novas
    solicitações, mesmo que estejam relacionadas ao assunto.
    - Se a pergunta já estiver clara e adequada à busca
    semântica, retorne-a sem alterações.
    - Retorne somente a consulta reformulada, sem aspas ou explicações.

    Consulta reformulada:
    """

    rewritten_prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITER_PROMPT_TEMPLATE),
        ("human", "{query}")
    ])

    # ---------------------------------------------------------
    # Chain execution for rewriting queries based on the selected provider
    # ---------------------------------------------------------
    if provider == "nvidia":
        rewriter_chain = rewritten_prompt | nvidia_llm | StrOutputParser()

    if provider == "ollama":
        rewriter_chain = rewritten_prompt | ollama_rewriter_llm | StrOutputParser()

    # ---------------------------------------------------------
    # Prompt template for answering questions based on retrieved documents
    # ---------------------------------------------------------

    PROMPT_TEMPLATE = """Você é um assistente especializado em responder perguntas
    sobre o conteúdo de documentos fornecidos.

    Sua função é ajudar o usuário a consultar e compreender as informações
    presentes nesses documentos, independentemente do assunto abordado.

    Responda exclusivamente com base no contexto fornecido.

    Regras:
    - Não utilize conhecimentos externos ao contexto.
    - Não invente informações, fatos ou conclusões que não estejam
    sustentados pelo contexto.
    - Responda a todas as partes da pergunta, desde que existam
    informações suficientes no contexto.
    - Quando o contexto apresentar informações sobre categorias,
    processos ou tipos de documentos diferentes, mantenha suas
    características e instruções separadas. Não atribua informações
    de uma categoria a outra sem evidência explícita.
    - Não presuma que os documentos contenham informações sobre
    assuntos que não foram abordados.
    - A ausência de informação não deve ser interpretada como
    confirmação de que algo não aconteceu ou não existe.
    - Se o contexto permitir apenas uma resposta parcial, apresente
    o que estiver fundamentado e indique o que não foi possível
    determinar.
    - Responda de forma clara, objetiva e no idioma da pergunta.

    Contexto:
    {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", PROMPT_TEMPLATE),
        ("human", "{query}")
    ])

    # ---------------------------------------------------------
    # Chain execution based on the selected provider
    # ---------------------------------------------------------
    if provider == "nvidia":
        chain = prompt | nvidia_llm | StrOutputParser()

    if provider == "ollama":
        chain = prompt | ollama_llm | StrOutputParser()

    return retriever, rewriter_chain, chain


def main():
    """
    Main function to execute the document indexing and question-answering process.
    """
    # ---------------------------------------------------------
    # Select the provider for embeddings and LLMs
    # ---------------------------------------------------------
    PROVIDER = "ollama"  # Options: "nvidia" or "ollama"

    retriever, rewriter_chain, chain = build_pipeline(PROVIDER)

    # ---------------------------------------------------------
    # Print the question-answering loop
    # ---------------------------------------------------------
    print(answer_questions(PROVIDER, retriever, rewriter_chain, chain))


if __name__ == "__main__":
    main()
