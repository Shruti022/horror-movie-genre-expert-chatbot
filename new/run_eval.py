#!/usr/bin/env python3
"""
SHADE Evaluation Harness
========================
Runs the full golden dataset against the live SHADE API.

Metrics:
  - Deterministic: regex / keyword checks (no LLM needed)
  - MaaJ golden-reference: Gemini judges response vs. expected answer
  - MaaJ rubric: Gemini grades response against a rubric

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
from dataclasses import dataclass, field
from typing import Optional

import httpx
import google.generativeai as genai
from dotenv import load_dotenv

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
judge_model = genai.GenerativeModel("gemini-1.5-flash")

console = Console() if RICH_AVAILABLE else None


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
    """
    Returns (pass, reason).
    Checks keyword presence/absence as specified in the test case.
    """
    check = test_case.get("deterministic_check")
    if not check:
        return True, "no_deterministic_check"

    check_type = check.get("type", "refusal_detection")
    response_lower = response.lower()

    # Check must_not_contain
    for phrase in check.get("must_not_contain", []):
        if phrase.lower() in response_lower:
            return False, f"FAIL: response contains forbidden phrase '{phrase}'"

    # Check must_contain_any
    must_contain = check.get("must_contain_any", [])
    if must_contain:
        found = any(p.lower() in response_lower for p in must_contain)
        if not found:
            return False, f"FAIL: response missing required terms (one of: {must_contain})"

    return True, "deterministic_check_passed"


# ── MaaJ: Golden reference judge ──────────────────────────────────────────────

GOLDEN_JUDGE_PROMPT = """You are an expert evaluator for a Psychological Horror Film chatbot named SHADE.

Your task: Compare the bot's actual response to a reference expected answer and score how well the bot answered.

Question asked: {question}

Reference expected answer: {expected}

Bot's actual response: {response}

Score the bot's response on a scale of 0-10 where:
- 10: Covers all key points from the reference, demonstrates deep expertise, no errors
- 7-9: Covers most key points, minor gaps or additions, accurate
- 4-6: Partial coverage, some key points missing, generally accurate
- 1-3: Major gaps, significant inaccuracies, or mostly off-topic  
- 0: Completely wrong or off-topic

Respond in this exact JSON format:
{{
  "score": <0-10>,
  "pass": <true if score >= 6, false otherwise>,
  "key_points_covered": ["point1", "point2"],
  "key_points_missing": ["point1"],
  "reasoning": "brief explanation"
}}"""


def maaj_golden_eval(test_case: dict, response: str) -> tuple[bool, float, str]:
    """MaaJ evaluation against golden reference answer."""
    prompt = GOLDEN_JUDGE_PROMPT.format(
        question=test_case["input"],
        expected=test_case["expected_answer"],
        response=response
    )
    try:
        result = judge_model.generate_content(prompt)
        raw = result.text.strip()
        # Strip markdown code blocks if present
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

Your task: Grade the bot's response against a detailed rubric.

Question asked: {question}

Rubric (each criterion should be met for full marks):
{rubric}

Bot's actual response:
{response}

For each criterion in the rubric, determine if it was met (true/false).
Then give an overall pass/fail (pass if ≥70% of criteria met).

Respond in this exact JSON format:
{{
  "criteria_results": [
    {{"criterion": "criterion text", "met": true, "evidence": "brief quote from response"}},
    ...
  ],
  "criteria_met_count": <number>,
  "criteria_total": <number>,
  "percentage": <0-100>,
  "pass": <true if percentage >= 70>,
  "overall_reasoning": "brief summary"
}}"""


def maaj_rubric_eval(test_case: dict, response: str) -> tuple[bool, float, str]:
    """MaaJ evaluation against rubric."""
    prompt = RUBRIC_JUDGE_PROMPT.format(
        question=test_case["input"],
        rubric=test_case.get("rubric", "Assess overall quality and relevance"),
        response=response
    )
    try:
        result = judge_model.generate_content(prompt)
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
    """Query the SHADE chatbot. Returns (response_text, flagged)."""
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


# ── Pretty printing ────────────────────────────────────────────────────────────

def print_separator(char="─", width=80):
    print(char * width)


def print_result(result: TestResult, verbose: bool = False):
    status = "✅ PASS" if result.overall_pass else "❌ FAIL"
    print(f"\n{status}  [{result.test_id}] {result.category}/{result.subcategory}")
    print(f"   Q: {result.input_text[:80]}{'...' if len(result.input_text) > 80 else ''}")

    if result.error:
        print(f"   ⚠️  Error: {result.error}")
        return

    if result.deterministic_pass is not None:
        d_status = "✅" if result.deterministic_pass else "❌"
        print(f"   {d_status} Deterministic: {result.deterministic_reason}")

    if result.maaj_golden_pass is not None:
        g_status = "✅" if result.maaj_golden_pass else "❌"
        print(f"   {g_status} MaaJ Golden (score {result.maaj_golden_score:.1f}/10): {result.maaj_golden_reason[:100]}")

    if result.maaj_rubric_pass is not None:
        r_status = "✅" if result.maaj_rubric_pass else "❌"
        print(f"   {r_status} MaaJ Rubric ({result.maaj_rubric_score:.0f}%): {result.maaj_rubric_reason[:100]}")

    if verbose:
        print(f"\n   Bot response:")
        for line in result.bot_response[:500].split('\n'):
            print(f"     {line}")
        if len(result.bot_response) > 500:
            print(f"     ... [{len(result.bot_response) - 500} more chars]")


def print_summary(results: list[TestResult]):
    print_separator("═")
    print("EVALUATION SUMMARY")
    print_separator("═")

    # Overall
    total = len(results)
    passed = sum(1 for r in results if r.overall_pass)
    print(f"\nOverall: {passed}/{total} passed ({100*passed/total:.1f}%)")

    # By category
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

    # Metric breakdown
    print("\nBy Metric Type:")
    det_results = [r for r in results if r.deterministic_pass is not None]
    if det_results:
        det_pass = sum(1 for r in det_results if r.deterministic_pass)
        print(f"  Deterministic:   {det_pass}/{len(det_results)} passed")

    golden_results = [r for r in results if r.maaj_golden_pass is not None]
    if golden_results:
        g_pass = sum(1 for r in golden_results if r.maaj_golden_pass)
        g_avg = sum(r.maaj_golden_score for r in golden_results) / len(golden_results)
        print(f"  MaaJ Golden:     {g_pass}/{len(golden_results)} passed (avg score: {g_avg:.1f}/10)")

    rubric_results = [r for r in results if r.maaj_rubric_pass is not None]
    if rubric_results:
        r_pass = sum(1 for r in rubric_results if r.maaj_rubric_pass)
        r_avg = sum(r.maaj_rubric_score for r in rubric_results) / len(rubric_results)
        print(f"  MaaJ Rubric:     {r_pass}/{len(rubric_results)} passed (avg: {r_avg:.0f}%)")

    print_separator("═")

    # Failed tests
    failed = [r for r in results if not r.overall_pass]
    if failed:
        print(f"\nFailed Tests ({len(failed)}):")
        for r in failed:
            print(f"  ❌ {r.test_id}: {r.input_text[:60]}...")

    print()


# ── Main eval runner ───────────────────────────────────────────────────────────

def run_eval(url: str, dataset_path: str, verbose: bool = False, category_filter: Optional[str] = None):
    print_separator("═")
    print(f"SHADE EVALUATION HARNESS")
    print(f"Target: {url}")
    print(f"Dataset: {dataset_path}")
    print_separator("═")

    # Load dataset
    with open(dataset_path) as f:
        dataset = json.load(f)

    test_cases = dataset["test_cases"]
    if category_filter:
        test_cases = [t for t in test_cases if t["category"] == category_filter]
        print(f"Filtering to category: {category_filter} ({len(test_cases)} tests)")

    # Health check
    try:
        health = httpx.get(f"{url}/health", timeout=10)
        health.raise_for_status()
        print(f"✅ Health check passed: {health.json()}")
    except Exception as e:
        print(f"⚠️  Health check failed: {e}. Proceeding anyway...")

    print(f"\nRunning {len(test_cases)} test cases...\n")

    results = []

    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['id']} — querying bot...", end="", flush=True)

        # Query bot
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

        # ── 1. Deterministic check ──
        if tc.get("deterministic_check"):
            d_pass, d_reason = run_deterministic_check(tc, bot_response)
            result.deterministic_pass = d_pass
            result.deterministic_reason = d_reason
            all_pass_flags.append(d_pass)

        # ── 2. MaaJ Golden reference ──
        if eval_type in ("maaj_golden", "maaj_both"):
            time.sleep(0.5)  # rate limiting
            g_pass, g_score, g_reason = maaj_golden_eval(tc, bot_response)
            result.maaj_golden_pass = g_pass
            result.maaj_golden_score = g_score
            result.maaj_golden_reason = g_reason
            all_pass_flags.append(g_pass)

        # ── 3. MaaJ Rubric ──
        if eval_type in ("maaj_rubric", "maaj_both"):
            time.sleep(0.5)  # rate limiting
            r_pass, r_score, r_reason = maaj_rubric_eval(tc, bot_response)
            result.maaj_rubric_pass = r_pass
            result.maaj_rubric_score = r_score
            result.maaj_rubric_reason = r_reason
            all_pass_flags.append(r_pass)

        # Overall: pass if all configured checks pass
        result.overall_pass = all(all_pass_flags) if all_pass_flags else False
        results.append(result)
        print_result(result, verbose)

    # Summary
    print_summary(results)

    # Save results to JSON
    output_path = Path("eval/results_latest.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([
            {
                "id": r.test_id,
                "category": r.category,
                "subcategory": r.subcategory,
                "input": r.input_text,
                "bot_response": r.bot_response,
                "deterministic_pass": r.deterministic_pass,
                "maaj_golden_pass": r.maaj_golden_pass,
                "maaj_golden_score": r.maaj_golden_score,
                "maaj_rubric_pass": r.maaj_rubric_pass,
                "maaj_rubric_score": r.maaj_rubric_score,
                "overall_pass": r.overall_pass,
                "error": r.error,
            }
            for r in results
        ], f, indent=2)
    print(f"Results saved to {output_path}")

    # Exit with appropriate code
    total = len(results)
    passed = sum(1 for r in results if r.overall_pass)
    pass_rate = passed / total if total > 0 else 0
    sys.exit(0 if pass_rate >= 0.7 else 1)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHADE Eval Harness")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the SHADE API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--dataset",
        default="eval/golden_dataset.json",
        help="Path to golden dataset JSON"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full bot responses"
    )
    parser.add_argument(
        "--category",
        choices=["in_domain", "out_of_scope", "adversarial"],
        help="Filter to a specific category"
    )

    args = parser.parse_args()
    run_eval(args.url, args.dataset, args.verbose, args.category)
