from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from time import perf_counter


def consult_documents(
    query: str,
    provider: str,
    retriever: BaseRetriever,
    rewriter_chain: Runnable,
    chain: Runnable
) -> dict:
    """Consult source documents based on the query, provider, retriever, rewriter chain, and chain.

    Args:
        query (str): The user's query or question.
        provider (str): The provider for embeddings and LLMs.
        retriever (BaseRetriever): The retriever for fetching similar documents.
        rewriter_chain (Runnable): The chain for rewriting the query with LLM.
        chain (Runnable): The chain for generating the final response.

    Raises:
        ValueError: If the rewritten query is empty.
        LookupError: If no relevant documents are found.

    Returns:
        dict: A dictionary containing the query, rewritten query, response, sources, and provider.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Digite uma pergunta válida.")
    
    start = perf_counter()
    rewritten_query = rewriter_chain.invoke({"query": query})
    rewrite_end = perf_counter()

    if not isinstance(rewritten_query, str) or not rewritten_query.strip():
        raise ValueError("Rewriter retornou uma consulta vazia.")

    rewritten_query = rewritten_query.strip()

    similar_chunks = retriever.invoke(rewritten_query)
    retrieve_end = perf_counter()

    if not similar_chunks:
        raise LookupError("Nenhum documento relevante encontrado para a consulta fornecida.")

    context = "\n\n".join(
        f"Documento: {chunk.metadata.get('source', 'não informado')}\n"
        f"{chunk.page_content}"
        for chunk in similar_chunks
    )

    response = chain.invoke({
        "query": query,
        "context": context
    })

    if not isinstance(response, str) or not response.strip():
        raise ValueError("O modelo retornou uma resposta vazia.")
    
    read_end = perf_counter()

    return {
        "query": query,
        "rewritten_query": rewritten_query,
        "response": response,
        "sources": [
            {
                "metadata": chunk.metadata,
                "content": chunk.page_content
            }
            for chunk in similar_chunks
        ],
        "provider": provider,
        "timings_seconds": {
            "rewrite": round(rewrite_end - start, 3),
            "retrieve": round(retrieve_end - rewrite_end, 3),
            "read": round(read_end - retrieve_end, 3),
            "total": round(read_end - start, 3),
        },
    }
                      
def answer_questions(
    provider: str,
    retriever: BaseRetriever,
    rewriter_chain: Runnable,
    chain: Runnable
) -> str:
    """Answers user questions by consulting source documents.

    Args:
        provider (str): The provider for embeddings and LLMs.
        retriever (BaseRetriever): The retriever for fetching similar documents.
        rewriter_chain (Runnable): The chain for rewriting the query with LLM.
        chain (Runnable): The chain for generating the final response.

    Returns:
        str: A message indicating the program has finished.
    """
    while True:
        query = input("\nDigite sua pergunta (ou 'sair'):\n> ")

        if query.lower() == "sair":
            break

        if not query.strip():
            print("Por favor, digite uma pergunta válida.\n")
            continue
        
        try:
            print("⏳ Processando consulta...\n", flush=True)
            
            result = consult_documents(
                query=query,
                provider=provider,
                retriever=retriever,
                rewriter_chain=rewriter_chain,
                chain=chain
            )

            print(result["response"])

        except (ValueError, LookupError) as e:
            print(f"Erro: {e}\n")

        except Exception as e:
            print(f"Falha na consulta ({type(e).__name__}): {e}\n")

    return "Programa encerrando. Obrigado por utilizar o assistente de consulta de documentos.\n"
