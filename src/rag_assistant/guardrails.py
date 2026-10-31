"""Input/output guardrails: PII redaction, prompt-injection screening, citation checks.

These are deliberately simple, transparent rules that run in microseconds and are
easy to audit. In production they are a first line of defence in front of a model-
based classifier (e.g. NVIDIA NeMo Guardrails or Azure AI Content Safety Prompt
Shields), not a replacement for one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)",
    r"disregard (all |any |the )?(previous|prior|above|your) (instructions|rules|guidelines)",
    r"(reveal|print|show|repeat|leak) (me )?(your|the) (system|hidden|initial) "
    r"(prompt|instructions)",
    r"you are now (in )?(developer|dan|jailbreak|unrestricted)",
    r"\b(jailbreak|do anything now)\b",
    r"pretend (that )?you (have no|are not bound by) (rules|restrictions|guidelines)",
    r"override (your|the) (safety|security) (rules|policies|settings)",
]
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.I)

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d /-]{7,}\d\b")


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    total = sum(
        d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits)
    )
    return len(digits) >= 13 and total % 10 == 0


@dataclass
class Redaction:
    text: str
    found: dict[str, int] = field(default_factory=dict)


def redact_pii(text: str) -> Redaction:
    """Replace e-mail addresses, IBANs, payment cards and phone numbers with placeholders."""
    found: dict[str, int] = {}

    def sub(pattern: re.Pattern, label: str, s: str, check=None) -> str:
        def repl(m: re.Match) -> str:
            if check and not check(m.group(0)):
                return m.group(0)
            found[label] = found.get(label, 0) + 1
            return f"[{label}]"

        return pattern.sub(repl, s)

    text = sub(_EMAIL, "EMAIL", text)
    text = sub(_IBAN, "IBAN", text)
    text = sub(_CARD, "CARD", text, _luhn_ok)
    text = sub(_PHONE, "PHONE", text)
    return Redaction(text, found)


def is_prompt_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


_CITATION = re.compile(r"\[(\d+)\]")


def check_citations(answer: str, n_sources: int) -> tuple[str, list[int]]:
    """Drop citations that point to non-existent sources; return the valid ones."""
    valid: list[int] = []

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_sources:
            if n not in valid:
                valid.append(n)
            return m.group(0)
        return ""

    cleaned = _CITATION.sub(repl, answer)
    return re.sub(r" {2,}", " ", cleaned).strip(), valid
