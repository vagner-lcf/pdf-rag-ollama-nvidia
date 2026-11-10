import json
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from models import ollama_llm_evaluator

ROOT = Path(__file__).resolve().parent.parent

parser = ArgumentParser()
parser.add_argument("--benchmark", type=Path, required=True)
parser.add_argument("--results", type=Path, required=True)

args = parser.parse_args()

benchmark_path = args.benchmark.resolve()
results_path = args.results.resolve()

benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
results = json.loads(results_path.read_text(encoding="utf-8"))["results"]

results_by_id = {item["id"]: item for item in results}


def evaluate_case(case):
    """Evaluate a case based on the provided data from the benchmark file and the results from the evaluation file. """
    result = results_by_id.get(case["id"])

    if (
        result is None
        or result.get("error")
        or not isinstance(result.get("response"), str)
        or not result["response"].strip()
    ):
        return {
            "id": case["id"],
            "score": None,
            "reason": "Não avaliado: resultado ausente, com erro ou resposta vazia.",
        }

    evaluation_data = {
        "question": case["question"],
        "expected_answer": case["expected_answer"],
        "required_points": case["required_points"],
        "evidence": case["evidence"],
        "model_response": result["response"],
    }

    prompt = f"""
    Você é um avaliador de qualidade de respostas de um sistema RAG.

    Compare a resposta do modelo com a resposta esperada,
    os pontos obrigatórios e as evidências documentais.

    Atribua uma nota conforme os critérios:

    0 = Erro factual relevante, invenção importante
    ou ausência substancial de resposta.

    1 = Resposta parcialmente correta, com omissão
    relevante ou cobertura insuficiente.

    2 = Resposta correta e suficientemente completa
    para a pergunta, sem contradições relevantes.

    Regras:
    - Uma contradição factual relevante implica nota 0,
    mesmo que outras partes estejam corretas.
    - Diferencie omissão de contradição.
    - Avalie o conteúdo, não o estilo.
    - Não exija palavras idênticas à resposta esperada.
    - Quando a informação não estiver disponível,
    reconhecer corretamente a limitação vale nota 2.
    - Não responda à pergunta original.
    - Retorne somente JSON com "score" e "reason".
    - "score" deve ser um inteiro de 0 a 2.
    - "reason" deve ser uma justificativa curta.

    Dados para avaliação:

    {json.dumps(evaluation_data, ensure_ascii=False, indent=2)}
    """

    response = ollama_llm_evaluator.invoke(prompt)
    evaluation = json.loads(response.content)

    score = evaluation["score"]

    if type(score) is not int or score not in (0, 1, 2):
        raise ValueError(f"Nota inválida no caso {case['id']}")

    return {
        "id": case["id"],
        "score": score,
        "reason": evaluation["reason"],
    }


output_path = (
    ROOT / "evaluation" / "analysis"
    / f"{results_path.stem}_evaluated_{datetime.now():%Y%m%d_%H%M%S}.json"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

evaluations = []

for case in benchmark:
    print(f"Avaliando {case['id']}...", flush=True)

    evaluation = evaluate_case(case)
    evaluations.append(evaluation)

    output_path.write_text(
        json.dumps(evaluations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Nota: {evaluation['score']}")

valid_scores = [
    item["score"]
    for item in evaluations
    if item["score"] is not None
]

total = sum(valid_scores)
skipped = len(evaluations) - len(valid_scores)

print(
    f"\nResultado: {total}/{len(valid_scores) * 2} pontos"
    f" ({skipped} casos não avaliados)"
)
print(f"Avaliações salvas em: {output_path}")
