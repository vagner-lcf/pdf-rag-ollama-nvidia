# RAG de PDFs com Ollama e NVIDIA

🌐 *Leia isto em [English](README.md).*

Aplicação em Python para consultar **documentos PDF de qualquer tema** com **RAG (Retrieval-Augmented Generation)**. Oferece dois modos de execução — **Ollama (local)** e **NVIDIA API (nuvem)** — com busca semântica e respostas baseadas nos trechos recuperados.

## Como funciona

**Pergunta → Rewrite → Retrieve → Read → Resposta**

1. O texto dos PDFs é fragmentado e indexado no **ChromaDB**, com índices separados por provedor.
2. Um LLM reformula a pergunta para a busca semântica; os **4 trechos mais relevantes** são recuperados.
3. O Reader recebe a **pergunta original** e os trechos recuperados, identificados por documento.
4. Um manifesto **SHA-256** permite reutilizar o índice e reconstruí-lo quando os PDFs mudam.

| Modo | LLM | Embeddings |
| --- | --- | --- |
| Ollama (local) | Qwen 3.5 9B | BGE-M3 |
| NVIDIA API | Nemotron 3.5 Lightning 30B A3B | Nemotron 3 Embed 1B |

**Tecnologias:** Python 3.12, LangChain, ChromaDB, pypdf, Ollama e NVIDIA API.

**Custo:** o modo Ollama não cobra por consulta, mas depende dos recursos do computador. A NVIDIA oferece **acesso gratuito para desenvolvimento e testes** pelo [NVIDIA Build](https://build.nvidia.com/), sujeito a limites e termos de uso; esse acesso não se destina à produção.

## Instalação e uso

Na raiz do projeto, com **Python 3.12** instalado:

```bash
python -m venv .venv
```

Ative o ambiente virtual: no **Windows (PowerShell)**, `.venv\Scripts\Activate.ps1`; no **Linux/macOS**, `source .venv/bin/activate`. Depois:

```bash
python -m pip install -r requirements.txt
```

1. Copie `.env.example` para `.env`. No modo NVIDIA, obtenha uma chave no [NVIDIA Build](https://build.nvidia.com/) e preencha `NVIDIA_API_KEY`. O Ollama dispensa essa chave; LangSmith é opcional e vem desativado.
2. Coloque os PDFs em `data/documents/`. Documentos, índices e credenciais não são versionados.
3. Ajuste `PROVIDER` em `src/main.py` para `"ollama"` ou `"nvidia"`.
4. Para utilizar o modo local ou executar o avaliador, instale e inicie o [Ollama](https://ollama.com/) e baixe os modelos:

```bash
ollama pull qwen3.5:9b
ollama pull bge-m3:567m
```

Execute o CLI na raiz do projeto:

```bash
python src/main.py
```

Digite perguntas no terminal e use `sair` para encerrar. Na primeira execução, os PDFs são indexados; nas seguintes, o índice pode ser reutilizado.

## Avaliação experimental

A qualidade foi investigada com um **benchmark de 15 perguntas sobre dois manuais públicos da Vale**, em três baterias por provedor. Os scripts `src/evaluate.py` e `src/evaluator.py` registram respostas, consultas reformuladas, fontes recuperadas, tempos por etapa e avaliação por LLM local (**0–2 pontos por pergunta**).

Para avaliar outros PDFs, utilize `evaluation/benchmark.example.json` como referência e prepare um gabarito próprio com perguntas, respostas esperadas, pontos essenciais e evidências. Os caminhos dos arquivos são informados por argumentos de linha de comando. O julgamento das respostas utiliza um LLM local via **Ollama**, inclusive quando o RAG avaliado utiliza NVIDIA.

As notas abaixo foram obtidas com o **mesmo prompt avaliador**; as respostas das duas primeiras baterias foram reavaliadas sem executar novamente o RAG.

| Bateria | NVIDIA | Ollama |
| --- | ---: | ---: |
| 1 — inicial | 26/30 (86,7%) | 20/30 (66,7%) |
| 2 — revisão | 21/30 (70,0%) | 16/30 (53,3%) |
| 3 — ajustes adicionais | 25/30 (83,3%) | 19/30 (63,3%) |

**Interpretação:** pontuações automatizadas, **não taxas de acerto verificadas manualmente**. Foram observadas inconsistências do avaliador, falhas de recuperação e confusão entre procedimentos. Os modos diferem em LLM e embeddings; portanto, a comparação é entre **pipelines completos**. As variações não podem ser atribuídas apenas aos prompts. Os PDFs usados nos testes não são distribuídos.

### Como executar a avaliação

O arquivo `evaluation/benchmark.example.json` demonstra a estrutura esperada. Para avaliar seus próprios documentos, crie um benchmark com base nas informações dos PDFs.

Execute as consultas:

```bash
python src/evaluate.py --provider nvidia --benchmark evaluation/benchmark.local.json
```

Avalie as respostas geradas:

```bash
python src/evaluator.py --benchmark evaluation/benchmark.local.json --results evaluation/results/ARQUIVO_GERADO.json
```

Os resultados são salvos em JSON para análise posterior. O avaliador atribui notas de **0 a 2** conforme os critérios do gabarito.

## Limitações e próximos passos

O projeto trabalha com **PDFs que contenham texto extraível** (sem OCR). A recuperação de quatro trechos pode omitir evidências, e respostas importantes exigem conferência. No modo NVIDIA, dados usados nas consultas são enviados ao provedor. A versão atual é um **CLI experimental**, não uma solução pronta para produção.

**Próxima etapa:** interface simples em **Streamlit** no mesmo repositório, mantendo o CLI, com seleção de provedor, perguntas e respostas com fontes.
