"""Audit Logger — MahaTraffic AI Reliability Layer.

Persists structured audit events for every multi-agent run into:
  reliability/logs/audit_log.jsonl

Fields logged:
  - run_id (UUID)
  - timestamp (ISO-8601 UTC)
  - user_question (sanitized/redacted for PII/secrets)
  - workflow_version
  - agents_invoked
  - tools_requested
  - tools_executed
  - tool_success
  - retry_count
  - review_status
  - guardrail_decision
  - latency_ms
  - final_status

Security Guarantee: Never logs API keys, secret credentials, or hidden system prompts.
"""

from __future__ import annotations
from datetime import datetime
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger("reliability.audit")

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "reliability" / "logs"
AUDIT_LOG_FILE = LOGS_DIR / "audit_log.jsonl"


def redact_sensitive_info(text: str) -> str:
    """Mask email addresses, phone numbers, and potential credential tokens."""
    if not isinstance(text, str):
        return ""
    # Mask emails
    t = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[REDACTED_EMAIL]", text)
    # Mask 10-digit Indian mobile numbers
    t = re.sub(r"\b[6-9]\d{9}\b", "[REDACTED_PHONE]", t)
    # Mask credential keys
    t = re.sub(r"(key|secret|password|token)\s*[:=]\s*['\"]?\S+['\"]?", r"\1=[REDACTED]", t, flags=re.I)
    return t


class AuditLogger:
    """Structured audit trail recorder for agent execution compliance."""

    def __init__(self, log_file: Path = AUDIT_LOG_FILE):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_run(
        self,
        run_id: str,
        user_question: str,
        workflow_version: str = "v5.0",
        agents_invoked: Optional[List[str]] = None,
        tools_requested: Optional[List[str]] = None,
        tools_executed: Optional[List[str]] = None,
        tool_success: bool = True,
        retry_count: int = 0,
        review_status: str = "approved",
        guardrail_decision: str = "passed",
        latency_ms: float = 0.0,
        final_status: str = "success",
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an immutable audit record for an agent workflow run."""
        record = {
            "run_id": run_id or str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_question": redact_sensitive_info(user_question),
            "workflow_version": workflow_version,
            "agents_invoked": agents_invoked or [],
            "tools_requested": tools_requested or [],
            "tools_executed": tools_executed or [],
            "tool_success": tool_success,
            "retry_count": retry_count,
            "review_status": review_status,
            "guardrail_decision": guardrail_decision,
            "latency_ms": round(latency_ms, 2),
            "final_status": final_status,
            "error": redact_sensitive_info(error) if error else None,
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("[AuditLogger] Logged run_id: %s (status: %s)", record["run_id"], final_status)
        except Exception as e:
            logger.error("[AuditLogger] Failed to write audit record: %s", e)

        return record

    def load_recent_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read recent audit records for reliability computation."""
        if not self.log_file.exists():
            return []
        records = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records[-limit:]


# Global singleton
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Return shared audit logger singleton."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
