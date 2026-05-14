"""
Evaluation pipeline using RAGAS metrics.
Scores: faithfulness, answer_relevancy, context_precision, context_recall.
Designed to be run in CI — exits with code 1 if any metric drops below threshold.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from src.rag_chain import RAGChain


# Thresholds — failing any of these gates the CI pipeline
THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.65,
}

EVAL_DATASET_PATH = Path("ci/eval_questions.json")


def load_eval_questions() -> List[Dict]:
    """Load ground-truth QA pairs for evaluation."""
    if not EVAL_DATASET_PATH.exists():
        # Fallback sample — replace with your domain questions
        return [
            {
                "question": "What is the main topic of the document?",
                "ground_truth": "The document covers AI engineering best practices.",
            }
        ]
    with open(EVAL_DATASET_PATH) as f:
        return json.load(f)


def build_eval_dataset(rag: RAGChain, questions: List[Dict]) -> Dataset:
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in questions:
        try:
            result = rag.run(item["question"], enforce_citations=False)
            rows["question"].append(item["question"])
            rows["answer"].append(result["answer"])
            rows["contexts"].append([s["snippet"] for s in result["sources"]])
            rows["ground_truth"].append(item.get("ground_truth", ""))
        except Exception as e:
            print(f"[Eval] Skipping question due to error: {e}")
    return Dataset.from_dict(rows)


def run_evaluation() -> bool:
    """Run RAGAS evaluation. Returns True if all thresholds pass."""
    rag = RAGChain()
    questions = load_eval_questions()
    print(f"[Eval] Running evaluation on {len(questions)} questions...")

    dataset = build_eval_dataset(rag, questions)
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n=== Evaluation Results ===")
    passed = True
    for metric, threshold in THRESHOLDS.items():
        score = results[metric]
        status = "✅ PASS" if score >= threshold else "❌ FAIL"
        if score < threshold:
            passed = False
        print(f"  {metric:25s}: {score:.3f}  (threshold: {threshold})  {status}")

    # Save results for CI artifact
    output = Path("ci/eval_results.json")
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump(dict(results), f, indent=2)
    print(f"\n[Eval] Results saved to {output}")
    return passed


if __name__ == "__main__":
    passed = run_evaluation()
    sys.exit(0 if passed else 1)
