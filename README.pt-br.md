# RAG de PDFs com Ollama e NVIDIA

Aplicação em Python para consultar documentos PDF utilizando **RAG (Retrieval-Augmented Generation)**, com recuperação semântica de informações e geração de respostas fundamentadas nos trechos encontrados.

O projeto permite comparar duas abordagens: **Ollama (execução local)** e **NVIDIA API (execução em nuvem)**.

## Como funciona

1. Os documentos PDF são divididos em trechos e indexados em um banco vetorial Chroma.
2. A pergunta do usuário é reformulada para melhorar a busca semântica.
3. Os trechos mais relevantes são recuperados.
4. O modelo gera uma resposta utilizando o contexto recuperado.

**Fluxo:** Rewrite → Retrieve → Read.

## Tecnologias

Python, LangChain, ChromaDB, Ollama e NVIDIA API.

## Estado do projeto

O protótipo funcional utiliza uma interface de terminal em português e retorna resultados padronizados, incluindo pergunta original, consulta reformulada, resposta e fontes recuperadas.

**Próximas etapas:** avaliação de qualidade, tratamento de erros, documentação de instalação e interface com Streamlit.

## Documentos

Os arquivos PDF e os bancos vetoriais não são distribuídos no repositório. A preparação automática do ambiente para uma instalação nova ainda está em desenvolvimento.