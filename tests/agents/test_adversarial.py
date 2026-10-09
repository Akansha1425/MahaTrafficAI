"""Adversarial and Robustness Test Suite for Multi-Agent Guardrails."""

import pytest
from agents.guardrails.guardrails import apply_input_guardrails
from reliability.adversarial_tests import AdversarialBenchmark


def test_adversarial_benchmark_suite():
    """Execute all benchmark prompts and assert expected guardrail decisions."""
    cases = AdversarialBenchmark.get_test_cases()
    assert len(cases) >= 10, "Adversarial suite must contain at least 10 diverse test cases"

    for tc in cases:
        guard_res = apply_input_guardrails(tc["prompt"])
        is_blocked = not guard_res["is_valid"]
        should_be_blocked = tc["should_be_blocked"]

        assert is_blocked == should_be_blocked, (
            f"Test case {tc['id']} ({tc['category']}) failed: "
            f"Expected blocked={should_be_blocked}, got blocked={is_blocked}. "
            f"Prompt: '{tc['prompt'][:80]}...' Reason: {guard_res.get('reason')}"
        )
