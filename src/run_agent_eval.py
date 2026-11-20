"""Agent Evaluation Runner (Phase 3A).

Executes scripted multi-turn conversation scenarios from tests/agent_scenarios.json,
programmatically verifies memory persistence, tool call behavior, constraint retention,
and citation validity (zero LLM judge), and exports reports/agent_eval.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import JobSearchAgent, AgentTurnResult

SCENARIOS_PATH = Path("tests/agent_scenarios.json")
REPORT_PATH = Path("reports/agent_eval.json")
JOBS_DATA_PATH = Path("data/jobs_clean.parquet")


def load_valid_job_ids() -> Set[str]:
    """Load valid job IDs from the clean dataset to verify citations."""
    if not JOBS_DATA_PATH.exists():
        return set()
    df = pd.read_parquet(JOBS_DATA_PATH)
    return set(df["id"].astype(str).tolist())


def extract_job_citations(text: str) -> List[str]:
    """Extract job IDs cited in model text (matches 10-digit IDs or job_id patterns)."""
    citations = re.findall(r"\b(\d{10})\b", text)
    return list(set(citations))


def verify_turn_checks(
    turn_result: AgentTurnResult,
    checks: Dict[str, Any],
    valid_job_ids: Set[str],
) -> Tuple[bool, List[str]]:
    """Evaluate programmatic checks for a single turn.

    Args:
        turn_result: The result object from the agent turn.
        checks: The dictionary of assertions defined in the scenario.
        valid_job_ids: Set of real job IDs in the dataset.

    Returns:
        Tuple of (passed: bool, failure_reasons: List[str]).
    """
    reasons = []

    # 1. must_not_crash
    if checks.get("must_not_crash", True):
        if turn_result.error:
            reasons.append(f"Turn crashed with error: {turn_result.error}")
        if not turn_result.reply:
            reasons.append("Agent produced an empty response.")

    # 2. require_tools
    req_tools = checks.get("require_tools", [])
    executed_tools = [tc.tool_name for tc in turn_result.tool_calls]
    for rt in req_tools:
        if rt not in executed_tools:
            reasons.append(f"Required tool '{rt}' was not called (called: {executed_tools}).")

    # 3. forbidden_tools
    forb_tools = checks.get("forbidden_tools", [])
    for ft in forb_tools:
        if ft in executed_tools:
            reasons.append(f"Forbidden tool '{ft}' was called.")

    # 4. memory_contains (scalar key/value equality or substring)
    mem_contains = checks.get("memory_contains", {})
    for key, expected_val in mem_contains.items():
        actual_val = turn_result.session_memory.get(key)
        if actual_val is None:
            reasons.append(f"Memory key '{key}' missing (expected '{expected_val}').")
        elif str(expected_val).lower() not in str(actual_val).lower():
            reasons.append(
                f"Memory key '{key}' value '{actual_val}' does not match expected '{expected_val}'."
            )

    # 5. memory_list_contains (list inclusion)
    mem_list_contains = checks.get("memory_list_contains", {})
    for list_key, expected_items in mem_list_contains.items():
        actual_list = turn_result.session_memory.get(list_key, [])
        if not isinstance(actual_list, list):
            reasons.append(f"Memory list '{list_key}' is not a list (got {type(actual_list)}).")
        else:
            actual_items_lower = [str(x).lower() for x in actual_list]
            for exp_item in expected_items:
                if not any(exp_item.lower() in act for act in actual_items_lower):
                    reasons.append(
                        f"Memory list '{list_key}' missing expected item '{exp_item}' (actual: {actual_list})."
                    )

    # 6. tool_arg_contains
    tool_arg_contains = checks.get("tool_arg_contains", {})
    for t_name, expected_args in tool_arg_contains.items():
        matching_calls = [tc for tc in turn_result.tool_calls if tc.tool_name == t_name]
        if not matching_calls:
            # If the tool wasn't called, require_tools would already flag it
            pass
        else:
            for exp_k, exp_v in expected_args.items():
                arg_matched = False
                for call in matching_calls:
                    val = call.args.get(exp_k)
                    if val is not None and str(exp_v).lower() in str(val).lower():
                        arg_matched = True
                        break
                if not arg_matched:
                    reasons.append(
                        f"Tool '{t_name}' was not called with argument '{exp_k}' containing '{exp_v}'."
                    )

    # 7. disallowed_in_result_titles
    disallowed = checks.get("disallowed_in_result_titles", [])
    for tc in turn_result.tool_calls:
        if tc.tool_name == "search_jobs" and isinstance(tc.result, list):
            for job in tc.result:
                title = str(job.get("title", "")).lower()
                for bad_word in disallowed:
                    if bad_word.lower() in title:
                        reasons.append(
                            f"Result contains disallowed term '{bad_word}' in title '{job.get('title')}'."
                        )

    # 8. require_valid_job_citations
    if checks.get("require_valid_job_citations", False) and valid_job_ids:
        # Check text citations
        cited_ids = extract_job_citations(turn_result.reply)
        for jid in cited_ids:
            if jid not in valid_job_ids:
                reasons.append(f"Hallucinated / non-existent job ID cited in text: {jid}")

        # Check get_job_details calls (except if scenario tests invalid ID)
        for tc in turn_result.tool_calls:
            if tc.tool_name == "get_job_details":
                jid_arg = str(tc.args.get("job_id", ""))
                # If scenario explicitly tests invalid ID, skip dataset existence check
                if "invalid" not in jid_arg.lower() and jid_arg not in valid_job_ids:
                    reasons.append(f"get_job_details called with unknown job ID: {jid_arg}")

    # 9. response_contains_phrase
    phrases = checks.get("response_contains_phrase", [])
    if phrases:
        reply_lower = turn_result.reply.lower()
        if not any(p.lower() in reply_lower for p in phrases):
            reasons.append(f"Response did not contain any of required phrases: {phrases}")

    passed = len(reasons) == 0
    return passed, reasons


def run_evaluation(
    smoke_test: bool = False,
    scenario_limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute evaluation over scripted conversation scenarios.

    Args:
        smoke_test: If True, executes only the first 2 scenarios.
        scenario_limit: Optional max number of scenarios to execute.

    Returns:
        Dictionary report matching reports/agent_eval.json schema.
    """
    if not SCENARIOS_PATH.exists():
        raise FileNotFoundError(f"Scenarios file not found at {SCENARIOS_PATH}")

    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    scenarios = payload.get("scenarios", [])
    if smoke_test:
        scenarios = scenarios[:2]
    elif scenario_limit:
        scenarios = scenarios[:scenario_limit]

    valid_job_ids = load_valid_job_ids()
    agent = JobSearchAgent()

    results_per_scenario = []
    total_turns = 0
    total_tool_calls = 0
    successful_tool_calls = 0
    memory_checks_total = 0
    memory_checks_passed = 0
    total_steps = 0
    total_latency_ms = 0.0
    live_calls = 0
    cache_hits = 0

    print("=" * 75)
    print(f"RUNNING AGENT SCENARIOS ({'SMOKE TEST (2 scenarios)' if smoke_test else f'{len(scenarios)} SCENARIOS'})")
    print(f"Model: {agent.model_name}")
    print("=" * 75)

    for sc_idx, sc in enumerate(scenarios, 1):
        sc_id = sc["id"]
        sc_name = sc["name"]
        session_id = f"eval_{sc_id}_{int(time.time())}"
        sc_passed = True
        sc_reasons: List[str] = []
        turn_evals = []

        print(f"\n[{sc_idx}/{len(scenarios)}] Scenario: {sc_name} ({sc_id})")

        for turn_data in sc["turns"]:
            t_num = turn_data["turn"]
            u_msg = turn_data["user_message"]
            checks = turn_data.get("checks", {})

            print(f"  Turn {t_num} User: \"{u_msg}\"")
            turn_res = agent.chat(session_id=session_id, message=u_msg)

            total_turns += 1
            total_steps += turn_res.steps
            total_latency_ms += turn_res.latency_ms

            if turn_res.cache_hit:
                cache_hits += 1
            else:
                live_calls += 1

            for tc in turn_res.tool_calls:
                total_tool_calls += 1
                if not tc.error:
                    successful_tool_calls += 1

            # Check memory checks
            if "memory_contains" in checks or "memory_list_contains" in checks:
                memory_checks_total += 1

            turn_ok, turn_reasons = verify_turn_checks(turn_res, checks, valid_job_ids)

            if "memory_contains" in checks or "memory_list_contains" in checks:
                if not any("Memory" in r for r in turn_reasons):
                    memory_checks_passed += 1

            if not turn_ok:
                sc_passed = False
                sc_reasons.extend([f"Turn {t_num}: {r}" for r in turn_reasons])

            executed_tool_names = [tc.tool_name for tc in turn_res.tool_calls]
            status_tag = "PASS" if turn_ok else "FAIL"
            print(f"  Turn {t_num} Agent: [{status_tag}] ({turn_res.steps} steps, {turn_res.latency_ms:.0f}ms) -> Tools: {executed_tool_names}")
            if turn_reasons:
                for r in turn_reasons:
                    print(f"      [!] {r}")

            turn_evals.append({
                "turn": t_num,
                "user_message": u_msg,
                "passed": turn_ok,
                "reasons": turn_reasons,
                "steps": turn_res.steps,
                "latency_ms": round(turn_res.latency_ms, 2),
                "tool_calls": [
                    {
                        "tool": tc.tool_name,
                        "args": tc.args,
                        "has_error": bool(tc.error),
                    }
                    for tc in turn_res.tool_calls
                ],
                "memory_state": turn_res.session_memory,
            })

        results_per_scenario.append({
            "scenario_id": sc_id,
            "scenario_name": sc_name,
            "passed": sc_passed,
            "failure_reasons": sc_reasons,
            "turns_count": len(turn_evals),
            "turn_details": turn_evals,
        })

    # Summary metrics
    n_scenarios = len(results_per_scenario)
    scenarios_passed = sum(1 for r in results_per_scenario if r["passed"])
    scenario_pass_rate = scenarios_passed / n_scenarios if n_scenarios > 0 else 0.0

    tool_call_success_rate = (
        successful_tool_calls / total_tool_calls if total_tool_calls > 0 else 1.0
    )
    memory_retention_pass_rate = (
        memory_checks_passed / memory_checks_total if memory_checks_total > 0 else 1.0
    )
    mean_steps = total_steps / total_turns if total_turns > 0 else 0.0
    mean_lat = total_latency_ms / total_turns if total_turns > 0 else 0.0
    all_turn_latencies = [t["latency_ms"] for s in results_per_scenario for t in s.get("turn_details", [])]
    max_lat = max(all_turn_latencies) if all_turn_latencies else 0.0

    report = {
        "caption": "Programmatic evaluation of agent multi-turn scenarios; no human or LLM judge",
        "gemini_model": agent.model_name,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_scenarios": n_scenarios,
            "scenarios_passed": scenarios_passed,
            "scenarios_failed": n_scenarios - scenarios_passed,
            "scenario_pass_rate": round(scenario_pass_rate, 4),
            "total_turns": total_turns,
            "mean_steps_per_turn": round(mean_steps, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "tool_call_success_rate": round(tool_call_success_rate, 4),
            "total_tool_calls": total_tool_calls,
            "memory_retention_pass_rate": round(memory_retention_pass_rate, 4),
            "memory_checks_count": memory_checks_total,
            "live_calls": live_calls,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(cache_hits / (live_calls + cache_hits), 4) if (live_calls + cache_hits) > 0 else 0.0,
        },
        "scenarios": results_per_scenario,
    }

    if not smoke_test:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved agent evaluation report to {REPORT_PATH}")

    print("\n" + "=" * 75)
    print("AGENT SCENARIOS EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Scenarios Passed     : {scenarios_passed}/{n_scenarios} ({scenario_pass_rate * 100:.1f}%)")
    print(f"Tool Call Success    : {successful_tool_calls}/{total_tool_calls} ({tool_call_success_rate * 100:.1f}%)")
    print(f"Memory Retention     : {memory_checks_passed}/{memory_checks_total} ({memory_retention_pass_rate * 100:.1f}%)")
    print(f"Mean Steps per Turn  : {mean_steps:.2f} (cap: 5)")
    print(f"Mean Turn Latency    : {mean_lat:.1f} ms")
    print(f"Max Turn Latency     : {max_lat:.1f} ms")
    print(f"Live Calls / Cache   : {live_calls} live / {cache_hits} cached")
    print("=" * 75 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multi-turn agent scenarios.")
    parser.add_argument("--smoke-test", action="store_true", default=False, help="Run 2-scenario smoke test.")
    parser.add_argument("--full", action="store_true", default=False, help="Run all 10 scenarios.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenarios.")
    args = parser.parse_args()

    smoke = args.smoke_test or (not args.full and args.limit is None)
    if args.full:
        smoke = False

    run_evaluation(smoke_test=smoke, scenario_limit=args.limit)
