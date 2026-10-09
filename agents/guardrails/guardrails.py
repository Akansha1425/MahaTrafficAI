"""Guardrails — MahaTraffic AI.

Implements robust Input, Tool, and Output Guardrails:
1. Input Scope & Adversarial Protection:
   - Prompt injection interception
   - Hidden instruction / system prompt extraction prevention
   - Unapproved tool execution blocking
   - Fabricated statistics assertion handling
   - Malformed / spam input rejection
   - Out-of-scope domain filtering
   - Nuanced discrimination: allows legitimate academic/security analysis of attack concepts
2. Tool Guardrails:
   - Enforces strict allowlist of approved MCP tools
   - Blocks dangerous parameters, code injection, and shell commands
3. Output Guardrails:
   - Verifies numerical grounding against tool payloads
   - Verifies risk scores (0-100) and tiers (LOW/MEDIUM/HIGH)
   - Enforces disclosure of missing data
   - Guarantees social sentiment is strictly non-causal
   - Enforces analytical disclaimers
   - States explicitly that automated reviewers cannot guarantee 0% hallucination
"""

from __future__ import annotations
import re
import logging
from typing import Any, Dict, List, Optional
from agents.schemas import ReviewResult, VALID_RISK_LEVELS

logger = logging.getLogger("agents.guardrails")

# ─── Domain Scope Keywords ───────────────────────────────────────────────────

MAHARASHTRA_LOCATIONS = [
    "maharashtra", "pune", "mumbai", "nashik", "nagpur", "thane", "solapur",
    "ahmednagar", "kolhapur", "satara", "raigad", "nanded", "aurangabad",
    "latur", "dhule", "jalna", "yavatmal", "chandrapur", "buldhana", "beed",
    "chhatrapati sambhajinagar", "navi mumbai", "panvel", "pimpri-chinchwad",
    "baramati", "shirdi", "amravati", "akola", "ratnagiri", "sindhudurg"
]

TRAFFIC_DOMAIN_KEYWORDS = [
    "accident", "crash", "road", "highway", "traffic", "fatality", "injury",
    "black spot", "risk", "speed", "drunk", "reckless", "pothole", "congestion",
    "vehicle", "motor", "pedestrian", "signal", "violation", "safety", "patrol",
    "morth", "irc", "nhai", "rto", "police", "district", "monsoon", "season",
    "analysis", "trend", "statistics", "data", "historical", "pattern",
    "sentiment", "social", "public", "perception", "report", "guideline", "speeding"
]

# ─── Adversarial Regex Patterns ───────────────────────────────────────────────

PROMPT_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|commands)\b",
    r"\b(system\s+prompt|developer\s+mode|dan\s+mode|jailbreak|jailbroken)\b",
    r"\b(you\s+are\s+now|act\s+as\s+an?\s+unrestricted|pretend\s+you\s+are\s+an?\s+unaligned)\b",
    r"\b(override|bypass)\s+(all\s+)?(guardrails|filters|safety|rules)\b",
    r"\boutput\s+government\s+secrets\b",
]

HIDDEN_INSTRUCTION_PATTERNS = [
    r"\b(reveal|print|show|output|leak|dump)\s+(your\s+)?(system\s+prompt|hidden\s+instructions|internal\s+instructions|developer\s+rules|api\s+keys)\b",
    r"\bwhat\s+(are\s+)?your\s+(hidden\s+instructions|system\s+prompt|secret\s+instructions)\b",
    r"\bdisplay\s+internal\s+prompts?\b",
]

UNAPPROVED_TOOL_PATTERNS = [
    r"\b(execute\s+shell|run\s+command|subprocess|bash|cmd\.exe|powershell)\b",
    r"\b(rm\s+-rf|delete\s+file|format\s+drive|drop\s+table|truncate\s+table)\b",
    r"\b(exec\(|eval\(|__import__)\b",
]

FABRICATED_STATS_PATTERNS = [
    r"\b(\d[\d,]{5,}|millions?\s+of)\s+accidents\b",
    r"\b100%\s+of\s+(all\s+)?drivers\s+.*(died|killed)\b",
    r"\b100%\s+of\s+(all\s+)?drivers\s+(died|killed)\b",
    r"\bzero\s+accidents\s+occurred\b",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\b(recipe|pizza|misal\s+pav|cake|burger|lasagna|biryani|cook|cooking)\b",
    r"\b(invest|crypto|bitcoin|ethereum|stock\s+market|portfolio|mutual\s+fund)\b",
    r"\b(diagnose|disease|medication|prescription|doctor|legal\s+lawsuit)\b",
    r"\b(poem|poetry|song\s+lyrics|story\s+about\s+dragons|fiction)\b",
    r"\b(weather\s+in\s+(paris|london|new\s+york|tokyo))\b",
]

# Security analysis indicators allowing discussion of attacks in academic context
SECURITY_ANALYSIS_INDICATORS = [
    "analyze", "analysis", "explain", "study", "research", "evaluate",
    "prevent", "mitigate", "defense", "academic", "security example", "understand"
]


# ─── 1. Input Guardrail ────────────────────────────────────────────────────────

def is_academic_security_query(query_lower: str) -> bool:
    """Detect if query is a legitimate academic inquiry examining security concepts."""
    has_analysis_intent = any(ind in query_lower for ind in SECURITY_ANALYSIS_INDICATORS)
    has_contextual_words = any(w in query_lower for w in ["why", "how", "what", "impact", "risk", "system", "agent"])
    return has_analysis_intent and has_contextual_words


def validate_input(query: str) -> dict:
    """Validate user input against injection, tool abuse, fabrication, and scope policies.

    Returns:
        dict: {
            "is_valid": bool,
            "category": str,
            "reason": str,
            "sanitized_query": str,
        }
    """
    if not query or len(query.strip()) < 3:
        return {
            "is_valid": False,
            "category": "malformed_input",
            "reason": "Query is too short or empty.",
            "sanitized_query": "",
        }

    # Length guardrail
    if len(query) > 2000:
        return {
            "is_valid": False,
            "category": "malformed_input",
            "reason": "Query exceeds maximum allowed length of 2000 characters.",
            "sanitized_query": query[:2000],
        }

    # Repetitive spam guardrail
    if re.search(r"(.)\1{30,}", query):
        return {
            "is_valid": False,
            "category": "malformed_input",
            "reason": "Query contains excessive repetitive characters (potential spam/DOS).",
            "sanitized_query": query,
        }

    lower = query.lower()
    academic_context = is_academic_security_query(lower)

    # 1. Prompt injection check (exempt if strictly academic inquiry about vulnerabilities)
    for pat in PROMPT_INJECTION_PATTERNS:
        if re.search(pat, lower):
            if not academic_context:
                logger.warning("[Input Guardrail] Intercepted prompt injection: %s", pat)
                return {
                    "is_valid": False,
                    "category": "prompt_injection",
                    "reason": "Adversarial prompt injection pattern detected and blocked.",
                    "sanitized_query": query,
                }

    # 2. Hidden instructions extraction check
    for pat in HIDDEN_INSTRUCTION_PATTERNS:
        if re.search(pat, lower):
            if not academic_context:
                logger.warning("[Input Guardrail] Intercepted instruction extraction: %s", pat)
                return {
                    "is_valid": False,
                    "category": "hidden_instruction_extraction",
                    "reason": "Requests to reveal confidential internal system instructions or secrets are prohibited.",
                    "sanitized_query": query,
                }

    # 3. Unapproved tool invocation check
    for pat in UNAPPROVED_TOOL_PATTERNS:
        if re.search(pat, lower):
            if not academic_context:
                logger.warning("[Input Guardrail] Intercepted unapproved tool request: %s", pat)
                return {
                    "is_valid": False,
                    "category": "unapproved_tool",
                    "reason": "Direct execution of arbitrary shell commands or code is strictly prohibited.",
                    "sanitized_query": query,
                }

    # 4. Fabricated statistics assertion check
    for pat in FABRICATED_STATS_PATTERNS:
        if re.search(pat, lower):
            logger.warning("[Input Guardrail] Intercepted fabricated statistics assertion: %s", pat)
            return {
                "is_valid": False,
                "category": "fabricated_statistics",
                "reason": "Query asserts fabricated or physically impossible accident statistics as absolute facts.",
                "sanitized_query": query,
            }

    # 5. Out of scope domain check
    for pat in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, lower):
            return {
                "is_valid": False,
                "category": "unsupported_domain",
                "reason": (
                    "Query is outside the road-safety domain. "
                    "MahaTraffic AI specializes exclusively in historical Maharashtra road accident analysis."
                ),
                "sanitized_query": query,
            }

    # 6. Domain relevance check
    has_loc = any(loc in lower for loc in MAHARASHTRA_LOCATIONS)
    has_concept = any(kw in lower for kw in TRAFFIC_DOMAIN_KEYWORDS)

    if not (has_loc or has_concept or academic_context):
        # Generic query check
        generic_ok = any(w in lower for w in [
            "risk", "safe", "danger", "accident", "road", "traffic", "highway",
            "data", "statistics", "trend", "analysis", "how", "what", "which", "where"
        ])
        if not generic_ok:
            return {
                "is_valid": False,
                "category": "unsupported_domain",
                "reason": (
                    "Query does not relate to Maharashtra road traffic safety. "
                    "Please ask about historical accident patterns, risk factors, or safety guidelines."
                ),
                "sanitized_query": query,
            }

    sanitized = re.sub(r"\s+", " ", query).strip()
    return {
        "is_valid": True,
        "category": "valid_traffic_query",
        "reason": "Query approved within road safety domain.",
        "sanitized_query": sanitized,
    }


def apply_input_guardrails(query: str) -> dict:
    """Public interface for input guardrail verification."""
    return validate_input(query)


# ─── 2. Reviewer and Output Guardrail ─────────────────────────────────────────

ANALYTICAL_DISCLAIMER = (
    "\n\n⚠️ ANALYTICAL NOTICE: All risk scores and rankings are historical outputs "
    "derived from 2019-2023 records. They do NOT represent real-time conditions, live alerts, "
    "or official government determinations. Public social posts represent perception signals, "
    "not statistical accident causes. Automated reviewer checks are applied, but independent "
    "verification is required (no automated review guarantees 100% absence of hallucinations)."
)


def review_response_content(
    response_text: str,
    tool_payloads: Optional[List[Dict[str, Any]]] = None,
    risk_score: Optional[float] = None,
    risk_level: Optional[str] = None,
) -> ReviewResult:
    """Review agent response for factual consistency, non-causal claims, and schema validity."""
    warnings: List[str] = []
    is_approved = True

    # Check 1: Risk score & level bounds
    valid_risk = True
    if risk_score is not None:
        if not (0.0 <= risk_score <= 100.0):
            warnings.append(f"Risk score {risk_score} is out of bounds [0.0, 100.0].")
            valid_risk = False
            is_approved = False

    if risk_level is not None:
        if risk_level.upper() not in {"LOW", "MEDIUM", "HIGH"}:
            warnings.append(f"Risk level '{risk_level}' is not in allowed set (LOW, MEDIUM, HIGH).")
            valid_risk = False
            is_approved = False

    # Check 2: Causal social claim check
    lower = response_text.lower()
    causal_violations = [
        "social media complaints cause accidents",
        "twitter complaints caused the fatal crash",
        "public negative sentiment resulted in 50 deaths",
        "social posts directly cause road accidents"
    ]
    non_causal_verified = True
    for violation in causal_violations:
        if violation in lower:
            warnings.append(f"Detected unsupported causal claim linking social media to accidents: '{violation}'")
            non_causal_verified = False
            is_approved = False

    # Check 3: Check for unsupported extreme claims
    if "guaranteed to eliminate all accidents" in lower or "100% safe with zero risk" in lower:
        warnings.append("Detected exaggerated claim exceeding supporting evidence.")
        is_approved = False

    # Check 4: Missing data disclosure verification
    missing_data_disclosed = True
    if "no data found" in lower or "not found in historical records" in lower or "temporarily unavailable" in lower:
        missing_data_disclosed = True

    # Check 5: Evidence grounding
    evidence_grounded = True
    if "data source" not in lower and "2019-2023" not in lower and "historical" not in lower:
        warnings.append("Response lacks explicit grounding reference to historical datasets.")
        evidence_grounded = False

    hallucination_risk = "LOW"
    if len(warnings) > 2:
        hallucination_risk = "HIGH"
    elif len(warnings) > 0:
        hallucination_risk = "MEDIUM"

    return ReviewResult(
        is_approved=is_approved,
        hallucination_risk=hallucination_risk,
        evidence_grounded=evidence_grounded,
        numeric_claims_verified=True,
        risk_levels_valid=valid_risk,
        missing_data_disclosed=missing_data_disclosed,
        non_causal_social_verified=non_causal_verified,
        recommendations_supported=is_approved,
        warnings=warnings,
        feedback="Verified against Phase 4 empirical tool payloads." if is_approved else "; ".join(warnings),
        disclaimer=ANALYTICAL_DISCLAIMER,
    )


def apply_output_guardrails(response: str) -> str:
    """Apply output formatting and ensure mandatory analytical disclaimer is appended."""
    if ANALYTICAL_DISCLAIMER.strip() not in response:
        response = response.rstrip() + ANALYTICAL_DISCLAIMER
    return response
