"""DeepEval evaluation. Owner: Person 4.

Reads tests/eval_cases.jsonl, runs each case through src.graph.run_question,
builds LLMTestCase objects, computes 5 DeepEval metrics + routing accuracy
+ citation check, and writes outputs/eval_report.json.

Usage (from the repo root):
    python -m tests.evaluate                 # full run
    python -m tests.evaluate --limit 2       # quick test on 2 cases
    python -m tests.evaluate --skip-metrics  # routing + citations only (no judge calls)
"""

import argparse
import json
import re
import time
import unicodedata
from uuid import uuid4

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.graph import run_question
from src.settings import EVAL_CASES_PATH, JUDGE_MODEL, OUTPUTS_DIR

load_dotenv()

REPORT_PATH = OUTPUTS_DIR / "eval_report.json"
METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "hallucination",
    "contextual_precision",
    "contextual_recall",
]
CONTEXT_METRICS = {
    "faithfulness",
    "hallucination",
    "contextual_precision",
    "contextual_recall",
}
PASS_THRESHOLD = 0.7
HALLUCINATION_THRESHOLD = 0.5  # lower is better for hallucination
PAUSE_SECONDS = 2  # small pause between judge calls (rate limits)
MAX_RETRIES = 3

TOOL_ERROR_PREFIXES = (
    "VALIDATION_ERROR:",
    "RETRIEVAL_ERROR:",
    "Tool execution failed",
    "Unknown tool requested",
    "Tool-call limit reached",
    "This tool has already failed",
)


# --------------------------------------------------------------------------
# Judge model (Gemini wrapped for DeepEval)
# --------------------------------------------------------------------------
def _to_text(content) -> str:
    """Convert a LangChain message content into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip() if content is not None else ""


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply (handles ``` fences)."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end != -1 else text


class GeminiJudge(DeepEvalBaseLLM):
    """Lets DeepEval use the Gemini model from settings.py as the judge."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            try:
                return self.model.with_structured_output(schema).invoke(prompt)
            except Exception:
                reply = _to_text(self.model.invoke(prompt).content)
                return schema(**json.loads(_extract_json(reply)))
        return _to_text(self.model.invoke(prompt).content)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model_name


def build_metrics(judge: GeminiJudge) -> dict:
    """Create the 5 DeepEval metrics."""
    common = {"model": judge, "async_mode": False, "include_reason": True}
    return {
        "faithfulness": FaithfulnessMetric(threshold=PASS_THRESHOLD, **common),
        "answer_relevancy": AnswerRelevancyMetric(threshold=PASS_THRESHOLD, **common),
        "hallucination": HallucinationMetric(
            threshold=HALLUCINATION_THRESHOLD, **common
        ),
        "contextual_precision": ContextualPrecisionMetric(
            threshold=PASS_THRESHOLD, **common
        ),
        "contextual_recall": ContextualRecallMetric(threshold=PASS_THRESHOLD, **common),
    }


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_cases(limit: int | None = None) -> list[dict]:
    cases = []
    with open(EVAL_CASES_PATH, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def _norm(name: str) -> str:
    """Normalize filenames so Arabic names compare reliably."""
    return unicodedata.normalize("NFC", name).strip()


def _get(obj, field, default=""):
    if isinstance(obj, dict):
        return obj.get(field, default)
    return getattr(obj, field, default)


def build_contexts(state: dict) -> list[str]:
    """Retrieved chunk texts + successful tool outputs = retrieval context."""
    contexts = [_get(chunk, "text") for chunk in state.get("retrieved_chunks", [])]
    for call in state.get("tool_calls", []):
        output = str(_get(call, "output"))
        if output and not output.startswith(TOOL_ERROR_PREFIXES):
            contexts.append(output)
    return [text for text in contexts if text]


def check_citations(expected: list[str], returned: list[str]) -> dict:
    expected_set = {_norm(name) for name in expected}
    returned_set = {_norm(name) for name in returned}
    found = expected_set & returned_set
    recall = len(found) / len(expected_set) if expected_set else 1.0
    return {
        "expected": sorted(expected_set),
        "returned": sorted(returned_set),
        "missing": sorted(expected_set - returned_set),
        "recall": round(recall, 3),
        "full_match": recall == 1.0,
    }


def measure(metric, test_case) -> dict:
    """Run one DeepEval metric with retries; never crash the whole run."""
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            metric.measure(test_case)
            time.sleep(PAUSE_SECONDS)
            return {
                "score": round(float(metric.score), 3),
                "passed": bool(metric.is_successful()),
                "reason": getattr(metric, "reason", None),
            }
        except Exception as exc:  # rate limit, bad JSON, etc.
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(10 * attempt)
    return {"score": None, "passed": None, "reason": f"ERROR: {last_error}"}


def score_metrics(test_case, metrics: dict, has_context: bool) -> dict:
    results = {}
    for name, metric in metrics.items():
        if name in CONTEXT_METRICS and not has_context:
            results[name] = {
                "score": None,
                "passed": None,
                "reason": "No retrieved context to evaluate.",
            }
        else:
            results[name] = measure(metric, test_case)
    return results


# --------------------------------------------------------------------------
# Main evaluation
# --------------------------------------------------------------------------
ERROR_MARKER = "the language model returned an error"
CASE_PAUSE = 10  # seconds between questions (rate limits)


def ask_with_retry(
    question: str, session_id: str, retries: int = 4, wait: int = 30
) -> dict:
    """Call the graph; if Gemini returns an error (e.g. rate limit), wait and retry.

    Args:
        question: The user question.
        session_id: Base session id; the attempt number is appended.
        retries: Maximum number of attempts.
        wait: Base waiting time in seconds (multiplied by the attempt number).

    Returns:
        The final graph state of the last attempt.
    """
    state = {}
    for attempt in range(1, retries + 1):
        state = run_question(question, session_id=f"{session_id}-{attempt}")
        if ERROR_MARKER not in state.get("answer", ""):
            return state
        delay = wait * attempt
        print(f"   LLM error (attempt {attempt}/{retries}); waiting {delay}s ...")
        time.sleep(wait * attempt)
    return state


def evaluate_case(case: dict, metrics: dict | None) -> dict:
    record = {
        "id": case["id"],
        "language": case.get("language"),
        "input": case["input"],
        "question": case["input"],
        "expected_domain": case["expected_domain"],
    }
    try:
        state = ask_with_retry(case["input"], f"eval-{case['id']}-{uuid4().hex[:8]}")
    except Exception as exc:
        record.update(
            error=f"{type(exc).__name__}: {exc}",
            predicted_domain=None,
            routing_correct=False,
            citations=check_citations(case.get("expected_sources", []), []),
            metrics={},
        )
        return record

    answer = state.get("answer", "")
    contexts = build_contexts(state)
    predicted = state.get("domain")

    record.update(
        predicted_domain=predicted,
        routing_correct=predicted == case["expected_domain"],
        actual_output=answer,
        answer=answer,
        llm_error=ERROR_MARKER in answer,
        expected_output=case["expected_output"],
        tools_used=[_get(call, "name") for call in state.get("tool_calls", [])],
        citations=check_citations(
            case.get("expected_sources", []), state.get("sources", [])
        ),
        metrics={},
    )

    if metrics and not record["llm_error"]:
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=answer,
            expected_output=case["expected_output"],
            retrieval_context=contexts,
            context=contexts,
        )
        record["metrics"] = score_metrics(test_case, metrics, bool(contexts))
    return record


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[dict]) -> dict:
    summary = {
        "cases": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "routing_accuracy": _mean([float(r["routing_correct"]) for r in results]),
        "routing_passed": sum(1 for r in results if r["routing_correct"]),
        "routing_failed": sum(1 for r in results if not r["routing_correct"]),
        "citation_recall": _mean([r["citations"]["recall"] for r in results]),
        "citation_full_match_rate": _mean(
            [float(r["citations"]["full_match"]) for r in results]
        ),
        "metrics": {},
        "by_language": {},
    }
    for name in METRIC_NAMES:
        entries = [
            r["metrics"][name]
            for r in results
            if name in r.get("metrics", {}) and r["metrics"][name]["score"] is not None
        ]
        if entries:
            summary["metrics"][name] = {
                "average_score": _mean([e["score"] for e in entries]),
                "pass_rate": _mean([float(e["passed"]) for e in entries]),
                "passed": sum(1 for e in entries if e["passed"]),
                "failed": sum(1 for e in entries if not e["passed"]),
                "evaluated": len(entries),
            }
    for lang in sorted({r["language"] for r in results if r.get("language")}):
        subset = [r for r in results if r.get("language") == lang]
        summary["by_language"][lang] = {
            "cases": len(subset),
            "routing_accuracy": _mean([float(r["routing_correct"]) for r in subset]),
            "citation_recall": _mean([r["citations"]["recall"] for r in subset]),
        }
    return summary


def print_report(results: list[dict], summary: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'ID':<10}{'Lang':<6}{'Route':<8}{'Cites':<8}", end="")
    print("".join(f"{n[:9]:<11}" for n in METRIC_NAMES))
    print("-" * 78)
    for r in results:
        row = f"{r['id']:<10}{r.get('language')!s:<6}"
        row += f"{'OK' if r['routing_correct'] else 'FAIL':<8}"
        row += f"{r['citations']['recall']:<8}"
        for name in METRIC_NAMES:
            score = r.get("metrics", {}).get(name, {}).get("score")
            row += f"{('-' if score is None else score):<11}"
        print(row)
    print("=" * 78)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.limit)
    metrics = None if args.skip_metrics else build_metrics(GeminiJudge(JUDGE_MODEL))

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}: {case['input'][:60]}")
        results.append(evaluate_case(case, metrics))
        time.sleep(CASE_PAUSE)

    summary = summarize(results)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {"summary": summary, "results": results},
            file,
            indent=2,
            ensure_ascii=False,
        )
    print_report(results, summary)
    print(f"\nSaved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
