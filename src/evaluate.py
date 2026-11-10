import json
from datetime import datetime
from pathlib import Path
from argparse import ArgumentParser

from main import build_pipeline
from question import consult_documents


def main():
    """Main function to run the document retrieval and question-answering from the benchmark file.
    It retrieves documents based on the provided queries and then uses the retrieved documents to answer questions.
    The results are saved in a JSON file.
    """
    parser = ArgumentParser()
    parser.add_argument("--provider", choices=["ollama", "nvidia"], required=True)
    parser.add_argument("--benchmark", type=Path, required=True)

    args = parser.parse_args()

    provider = args.provider
    benchmark_path = args.benchmark.resolve()

    project_root = Path(__file__).resolve().parent.parent
    
    results_dir = project_root / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"{provider}_{timestamp}.json"

    cases = json.loads(benchmark_path.read_text(encoding="utf-8"))
    retriever, rewriter_chain, chain = build_pipeline(provider)

    results = []

    for case in cases:
        print(f"\nExecutando {case['id']}...")

        try:
            result = consult_documents(
                query=case["question"],
                provider=provider,
                retriever=retriever,
                rewriter_chain=rewriter_chain,
                chain=chain
            )

            record = {
                "id": case["id"],
                **result,
                "error": None,
            }

            print(f"{case['id']} concluído.")

        except Exception as exc:
            record = {
                "id": case["id"],
                "query": case["question"],
                "provider": provider,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        
            print(f"{case['id']} falhou ({type(exc).__name__}): {exc}")

        results.append(record)

        output = {
            "provider": provider,
            "benchmark": benchmark_path.name,
            "results": results,
        }

        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"\nResultados salvos em: {output_path}")


if __name__ == "__main__":
    main()
