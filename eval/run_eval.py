#!/usr/bin/env python3
"""
SHADE Evaluation Harness
========================
Runs the full golden dataset against the live SHADE API.

Usage:
  uv run eval/run_eval.py --url http://localhost:8000
  uv run eval/run_eval.py --url https://your-gcp-url.run.app
"""

import json
import re
import sys
import time
import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import httpx
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Gemini judge setup (Vertex AI) ─────────────────────────────────────────────
GCP_PROJECT = os.getenv("GCP_PROJECT", "id")
client = genai.Client(vertexai=True, project=GCP_PROJECT, location="us-central1")
JUDGE_MODEL = "gemini-2.0-flash"


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class TestResult:
    test_id: str
    category: str
    subcategory: str
    input_text: str
    bot_response: str
    deterministic_pass: Optional[bool] = None
    deterministic_reason: str = ""
    maaj_golden_pass: Optional[bool] = None
    maaj_golden_score: float = 0.0
    maaj_golden_reason: str = ""
    maaj_rubric_pass: Optional[bool] = None
    maaj_rubric_score: float = 0.0
    maaj_rubric_reason: str = ""
    overall_pass: bool = False
    error: str = ""


# ── Deterministic checks ───────────────────────────────────────────────────────
def run_deterministic_check(test_case: dict, response: str) -> tuple[bool, str]:
    """Verify must_contain_any / must_not_contain keyword rules on the response."""
    check = test_case.get("deterministic_check")
    if not check:
        return True, "no_deterministic_check"

    response_lower = response.lower()

    for phrase in check.get("must_not_contain", []):
        if phrase.lower() in response_lower:
            return False, f"FAIL: response contains forbidden phrase '{phrase}'"

    must_contain = check.get("must_contain_any", [])
    if must_contain:
        found = any(p.lower() in response_lower for p in must_contain)
        if not found:
            return False, f"FAIL: response missing required terms (one of: {must_contain})"

    return True, "deterministic_check_passed"


# ── MaaJ: Golden reference judge ──────────────────────────────────────────────
GOLDEN_JUDGE_PROMPT = """You are an expert evaluator for a Psychological Horror Film chatbot named SHADE.

Compare the bot's actual response to a reference expected answer and score how well the bot answered.

Question asked: {question}

Reference expected answer: {expected}

Bot's actual response: {response}

Score 0-10 where:
- 10: Covers all key points, demonstrates deep expertise, no errors
- 7-9: Covers most key points, minor gaps, accurate
- 4-6: Partial coverage, some key points missing
- 1-3: Major gaps or inaccuracies
- 0: Completely wrong or off-topic

Respond in this exact JSON format with no markdown:
{{"score": <0-10>, "pass": <true if score >= 6>, "reasoning": "brief explanation"}}"""


def maaj_golden_eval(test_case: dict, response: str) -> tuple[bool, float, str]:
    """LLM-as-judge: score response against a golden reference answer (0-10)."""
    prompt = GOLDEN_JUDGE_PROMPT.format(
        question=test_case["input"],
        expected=test_case["expected_answer"],
        response=response
    )
    try:
        result = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
        )
        raw = result.text.strip()
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
        data = json.loads(raw)
        score = float(data.get("score", 0))
        passed = bool(data.get("pass", score >= 6))
        reason = data.get("reasoning", "")
        return passed, score, reason
    except Exception as e:
        return False, 0.0, f"Judge error: {str(e)}"


# ── MaaJ: Rubric judge ────────────────────────────────────────────────────────
RUBRIC_JUDGE_PROMPT = """You are an expert evaluator for a Psychological Horror Film chatbot named SHADE.

Grade the bot's response against this rubric.

Question asked: {question}

Rubric: {rubric}

Bot's actual response: {response}

Determine what percentage of rubric criteria are met. Pass if >= 70%.

Respond in this exact JSON format with no markdown:
{{"percentage": <0-100>, "pass": <true if percentage >= 70>, "overall_reasoning": "brief summary"}}"""


def maaj_rubric_eval(test_case: dict, response: str) -> tuple[bool, float, str]:
    """LLM-as-judge: score response against a rubric checklist (0-100%, pass >= 70%)."""
    prompt = RUBRIC_JUDGE_PROMPT.format(
        question=test_case["input"],
        rubric=test_case.get("rubric", "Assess overall quality and relevance"),
        response=response
    )
    try:
        result = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
        )
        raw = result.text.strip()
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
        data = json.loads(raw)
        percentage = float(data.get("percentage", 0))
        passed = bool(data.get("pass", percentage >= 70))
        reason = data.get("overall_reasoning", "")
        return passed, percentage, reason
    except Exception as e:
        return False, 0.0, f"Judge error: {str(e)}"


# ── Bot query ──────────────────────────────────────────────────────────────────
def query_bot(url: str, message: str, timeout: int = 30) -> tuple[str, bool]:
    """POST a message to the SHADE /chat endpoint; return (reply, flagged)."""
    try:
        resp = httpx.post(
            f"{url}/chat",
            json={"message": message, "history": []},
            timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("reply", ""), data.get("flagged", False)
    except httpx.TimeoutException:
        return "[TIMEOUT]", False
    except Exception as e:
        return f"[ERROR: {str(e)}]", False


# ── Print helpers ──────────────────────────────────────────────────────────────
def print_separator(char="─", width=70):
    print(char * width)


def print_result(result: TestResult, verbose: bool = False):
    """Print pass/fail status and per-metric breakdown for a single test."""
    status = "✅ PASS" if result.overall_pass else "❌ FAIL"
    print(f"\n{status}  [{result.test_id}] {result.category}/{result.subcategory}")
    print(f"   Q: {result.input_text[:75]}{'...' if len(result.input_text) > 75 else ''}")

    if result.error:
        print(f"   ⚠️  Error: {result.error}")
        return

    if result.deterministic_pass is not None:
        d = "✅" if result.deterministic_pass else "❌"
        print(f"   {d} Deterministic: {result.deterministic_reason}")

    if result.maaj_golden_pass is not None:
        g = "✅" if result.maaj_golden_pass else "❌"
        print(f"   {g} MaaJ Golden (score {result.maaj_golden_score:.1f}/10): {result.maaj_golden_reason[:100]}")

    if result.maaj_rubric_pass is not None:
        r = "✅" if result.maaj_rubric_pass else "❌"
        print(f"   {r} MaaJ Rubric ({result.maaj_rubric_score:.0f}%): {result.maaj_rubric_reason[:100]}")

    if verbose:
        print(f"\n   Bot response: {result.bot_response[:400]}")


def print_summary(results: list[TestResult]):
    """Print aggregate stats: overall pass rate, by-category breakdown, and failed test IDs."""
    print_separator("═")
    print("EVALUATION SUMMARY")
    print_separator("═")

    total = len(results)
    passed = sum(1 for r in results if r.overall_pass)
    print(f"\nOverall: {passed}/{total} passed ({100*passed/total:.1f}%)")

    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"pass": 0, "total": 0}
        categories[cat]["total"] += 1
        if r.overall_pass:
            categories[cat]["pass"] += 1

    print("\nBy Category:")
    for cat, counts in sorted(categories.items()):
        p = counts["pass"]
        t = counts["total"]
        pct = 100 * p / t
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        status = "✅" if pct >= 70 else "⚠️ " if pct >= 50 else "❌"
        print(f"  {status} {cat:<20} {bar} {p}/{t} ({pct:.0f}%)")

    print("\nBy Metric:")
    det = [r for r in results if r.deterministic_pass is not None]
    if det:
        print(f"  Deterministic:  {sum(1 for r in det if r.deterministic_pass)}/{len(det)} passed")

    golden = [r for r in results if r.maaj_golden_pass is not None]
    if golden:
        avg = sum(r.maaj_golden_score for r in golden) / len(golden)
        print(f"  MaaJ Golden:    {sum(1 for r in golden if r.maaj_golden_pass)}/{len(golden)} passed (avg {avg:.1f}/10)")

    rubric = [r for r in results if r.maaj_rubric_pass is not None]
    if rubric:
        avg = sum(r.maaj_rubric_score for r in rubric) / len(rubric)
        print(f"  MaaJ Rubric:    {sum(1 for r in rubric if r.maaj_rubric_pass)}/{len(rubric)} passed (avg {avg:.0f}%)")

    failed = [r for r in results if not r.overall_pass]
    if failed:
        print(f"\nFailed Tests ({len(failed)}):")
        for r in failed:
            print(f"  ❌ {r.test_id}: {r.input_text[:60]}...")

    print_separator("═")


# ── Main ───────────────────────────────────────────────────────────────────────
def run_eval(url: str, dataset_path: str, verbose: bool = False, category_filter: Optional[str] = None):
    """Main loop: load dataset, query the bot for each test case, run all
    eval layers (deterministic + MaaJ), print results, and write JSON."""
    print_separator("═")
    print(f"SHADE EVALUATION HARNESS")
    print(f"Target: {url}")
    print_separator("═")

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    test_cases = dataset["test_cases"]
    if category_filter:
        test_cases = [t for t in test_cases if t["category"] == category_filter]
        print(f"Filtering to: {category_filter} ({len(test_cases)} tests)")

    # Health check
    try:
        health = httpx.get(f"{url}/health", timeout=10)
        print(f"✅ Health check: {health.json()}")
    except Exception as e:
        print(f"⚠️  Health check failed: {e}")

    print(f"\nRunning {len(test_cases)} tests...\n")
    results = []

    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['id']} — querying...", end="", flush=True)

        bot_response, flagged = query_bot(url, tc["input"])
        print(" ✓")

        result = TestResult(
            test_id=tc["id"],
            category=tc["category"],
            subcategory=tc["subcategory"],
            input_text=tc["input"],
            bot_response=bot_response,
        )

        if bot_response.startswith("[ERROR") or bot_response.startswith("[TIMEOUT"):
            result.error = bot_response
            result.overall_pass = False
            results.append(result)
            print_result(result, verbose)
            continue

        eval_type = tc.get("eval_type", "maaj_rubric")
        all_pass_flags = []

        if tc.get("deterministic_check"):
            d_pass, d_reason = run_deterministic_check(tc, bot_response)
            result.deterministic_pass = d_pass
            result.deterministic_reason = d_reason
            all_pass_flags.append(d_pass)

        if eval_type in ("maaj_golden", "maaj_both"):
            time.sleep(0.5)
            g_pass, g_score, g_reason = maaj_golden_eval(tc, bot_response)
            result.maaj_golden_pass = g_pass
            result.maaj_golden_score = g_score
            result.maaj_golden_reason = g_reason
            all_pass_flags.append(g_pass)

        if eval_type in ("maaj_rubric", "maaj_both"):
            time.sleep(0.5)
            r_pass, r_score, r_reason = maaj_rubric_eval(tc, bot_response)
            result.maaj_rubric_pass = r_pass
            result.maaj_rubric_score = r_score
            result.maaj_rubric_reason = r_reason
            all_pass_flags.append(r_pass)

        result.overall_pass = all(all_pass_flags) if all_pass_flags else False
        results.append(result)
        print_result(result, verbose)

    print_summary(results)

    output_path = Path("eval/results_latest.json")
    with open(output_path, "w") as f:
        json.dump([
            {
                "id": r.test_id,
                "category": r.category,
                "input": r.input_text,
                "bot_response": r.bot_response,
                "deterministic_pass": r.deterministic_pass,
                "maaj_golden_pass": r.maaj_golden_pass,
                "maaj_golden_score": r.maaj_golden_score,
                "maaj_rubric_pass": r.maaj_rubric_pass,
                "maaj_rubric_score": r.maaj_rubric_score,
                "overall_pass": r.overall_pass,
            }
            for r in results
        ], f, indent=2)
    print(f"\nResults saved to {output_path}")

    total = len(results)
    passed = sum(1 for r in results if r.overall_pass)
    sys.exit(0 if passed / total >= 0.7 else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHADE Eval Harness")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="eval/golden_dataset.json")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--category", choices=["in_domain", "out_of_scope", "adversarial"])
    args = parser.parse_args()
    run_eval(args.url, args.dataset, args.verbose, args.category)
