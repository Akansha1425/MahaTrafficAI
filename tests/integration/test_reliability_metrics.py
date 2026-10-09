"""Unit and Integration tests for Reliability Metrics and Audit Logger."""

import pytest
from reliability.metrics import compute_reliability_metrics, ReliabilityMetricsReport
from reliability.audit_logger import AuditLogger, redact_sensitive_info
from pathlib import Path
import tempfile


def test_redact_sensitive_info():
    """Verify sensitive information is masked in audit logs."""
    sample = "User email is test@maharashtra.gov.in, phone is 9876543210, token: 'secret123'"
    redacted = redact_sensitive_info(sample)
    assert "test@maharashtra.gov.in" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "9876543210" not in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "secret123" not in redacted


def test_audit_logger_write_and_read():
    """Verify audit logger writes structured JSONL lines cleanly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test_audit.jsonl"
        logger = AuditLogger(log_file=log_file)

        rec = logger.log_run(
            run_id="test-run-123",
            user_question="Accident query in Pune",
            workflow_version="v5.0",
            agents_invoked=["analytics", "risk"],
            tools_requested=["get_city_accident_statistics"],
            tools_executed=["get_city_accident_statistics"],
            tool_success=True,
            retry_count=0,
            review_status="approved",
            guardrail_decision="approved",
            latency_ms=125.4,
            final_status="success",
        )
        assert rec["run_id"] == "test-run-123"
        assert log_file.exists()

        records = logger.load_recent_records()
        assert len(records) == 1
        assert records[0]["run_id"] == "test-run-123"
        assert records[0]["tool_success"] is True


def test_compute_reliability_metrics_calculations():
    """Verify reliability metric computation engine computes expected percentages."""
    mock_traces = [
        {
            "status": "success",
            "final_response": "Valid response with data",
            "schema_compliant": True,
            "evidence_grounded": True,
            "deterministic": True,
            "latency_ms": 100.0,
            "tools_executed": [
                {"tool_name": "tool_a", "status": "success", "retry_count": 0},
                {"tool_name": "tool_b", "status": "success", "retry_count": 1},
            ],
        },
        {
            "status": "success",
            "final_response": "Another valid response",
            "schema_compliant": True,
            "evidence_grounded": True,
            "deterministic": True,
            "latency_ms": 200.0,
            "tools_executed": [
                {"tool_name": "tool_a", "status": "success", "retry_count": 0},
            ],
        },
    ]

    mock_adversarial = [
        {"category": "prompt_injection", "is_valid": False},
        {"category": "unsupported_domain", "is_valid": False},
    ]

    mock_benign = [
        {"category": "academic_security_analysis", "is_valid": True},
    ]

    report: ReliabilityMetricsReport = compute_reliability_metrics(
        eval_traces=mock_traces,
        adversarial_results=mock_adversarial,
        benign_results=mock_benign,
    )

    assert report.total_runs_evaluated == 2
    assert report.tool_success_rate == 100.0
    assert report.task_completion_rate == 100.0
    assert report.schema_compliance_rate == 100.0
    assert report.guardrail_detection_rate == 100.0
    assert report.false_positive_rate == 0.0
    assert report.prompt_injection_resistance == 100.0
    assert report.mean_latency_ms == 150.0
    assert len(report.metrics_breakdown) == 11
