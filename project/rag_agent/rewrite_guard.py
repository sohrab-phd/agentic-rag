"""Post-generation guardrails for query rewrite fidelity."""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

_QUESTION_TYPES: list[tuple[str, re.Pattern[str]]] = [
    ("کجاست", re.compile(r"کجاست|کجا\s")),
    ("چرا", re.compile(r"چرا")),
    ("چه زمانی", re.compile(r"چه\s*زمانی|کی\s")),
    ("چگونه", re.compile(r"چگونه|چطور")),
    ("چیست", re.compile(r"چیست|چه\s*است")),
]

_INTENT_COLLAPSE_PATTERNS = [
    re.compile(r"رابطه\s+در\s+زمینه"),
    re.compile(r"در\s+زمینه\s+مورد\s+بحث"),
    re.compile(r"موضوع\s+مورد\s+بحث"),
]

_PROCESS_MARKERS = re.compile(r"مراحل|فرآیند|گام\s*ها|چگونه\s*انجام")
_LOCATION_MARKERS = re.compile(r"کجاست|کجا\s|مسیر|مکان|بخش|منوی|از\s+کدام")

_EN_STOPWORDS = re.compile(
    r"\b(transfer learning|pipeline approach|approach)\b", re.IGNORECASE
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"[\s\u200c]+", " ", text).strip()
    return text


def _compact(text: str) -> str:
    return re.sub(r"[\s\u200c]+", "", _normalize(text)).lower()


def _tokens(text: str, min_len: int = 3) -> list[str]:
    text = _normalize(text).lower()
    raw = re.split(r"[^\w\u0600-\u06ff]+", text)
    out = []
    for t in raw:
        t = t.strip("؟?،,.")
        if len(t) >= min_len:
            out.append(t)
    return out


def _detect_question_types(text: str) -> set[str]:
    found: set[str] = set()
    for label, pattern in _QUESTION_TYPES:
        if pattern.search(_normalize(text)):
            found.add(label)
    return found


def _term_present(term: str, text_compact: str, text_tokens: set[str]) -> bool:
    term_compact = _compact(term)
    if term_compact and term_compact in text_compact:
        return True
    term_toks = set(_tokens(term, min_len=3))
    if term_toks and term_toks.issubset(text_tokens):
        return True
    return False


def _extract_preserved_terms(original: str) -> list[str]:
    text = _normalize(original)
    terms: list[str] = []

    # Multi-word domain phrases (4+ chars per part), excluding full-question spans
    for match in re.finditer(
        r"[\u0600-\u06ff]{3,}(?:\s+[\u0600-\u06ff]{3,}){1,3}", text
    ):
        phrase = match.group(0).strip()
        if len(phrase) >= 7 and len(phrase) < len(text) * 0.85 and phrase not in terms:
            terms.append(phrase)

    # Single significant tokens (skip bare question-type markers)
    qtype_tokens = {"چیست", "کجاست", "چرا", "چگونه", "چطور", "کی"}
    for tok in _tokens(text, min_len=4):
        if tok not in qtype_tokens and tok not in terms:
            terms.append(tok)

    # Latin/domain tokens inside parentheses
    for match in re.finditer(r"\(([^)]+)\)", text):
        inner = match.group(1).strip()
        if inner and inner not in terms:
            terms.append(inner)

    terms.sort(key=len, reverse=True)
    unique: list[str] = []
    seen_compact: set[str] = set()
    for term in terms:
        key = _compact(term)
        if key and key not in seen_compact:
            unique.append(term)
            seen_compact.add(key)
    return unique[:12]


def _count_intents(original: str) -> int:
    text = _normalize(original)
    if "؟" in text or "?" in text:
        parts = re.split(r"[؟?]", text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return len(parts)
    if re.search(r"\s+و\s+", text):
        return 2
    return 1


def _validate_single_rewrite(original: str, rewritten: str) -> list[str]:
    reasons: list[str] = []
    orig = _normalize(original)
    rew = _normalize(rewritten)

    for pattern in _INTENT_COLLAPSE_PATTERNS:
        if pattern.search(rew) and not pattern.search(orig):
            reasons.append("intent_collapse")

    orig_types = _detect_question_types(orig)
    rew_types = _detect_question_types(rew)
    for qtype in orig_types:
        if qtype not in rew_types:
            reasons.append(f"question_type_drift:{qtype}")

    if _LOCATION_MARKERS.search(orig) and not _LOCATION_MARKERS.search(rew):
        if _PROCESS_MARKERS.search(rew):
            reasons.append("location_to_process_drift")

    preserved = _extract_preserved_terms(orig)
    rew_compact = _compact(rew)
    rew_tokens = set(_tokens(rew, min_len=3))
    missing_entities = []
    missing_keywords = []
    for term in preserved:
        if not _term_present(term, rew_compact, rew_tokens):
            if len(term.split()) > 1 or len(term) >= 6:
                missing_entities.append(term)
            else:
                missing_keywords.append(term)

    if missing_entities:
        reasons.append(f"missing_entities:{','.join(missing_entities[:5])}")
    if missing_keywords:
        reasons.append(f"missing_keywords:{','.join(missing_keywords[:5])}")

    return reasons


def validate_rewrites(original: str, proposed: list[str]) -> dict[str, Any]:
    original = _normalize(original)
    proposed = [_normalize(q) for q in proposed if q and q.strip()]

    if not proposed:
        return {
            "valid": False,
            "reasons": ["empty_rewrite"],
            "issues": ["empty_rewrite"],
        }

    all_reasons: list[str] = []
    all_issues: list[str] = []

    expected_intents = _count_intents(original)
    if len(proposed) < expected_intents:
        all_reasons.append("multi_intent_loss")
        all_issues.append("multi_intent_loss")

    orig_terms = _extract_preserved_terms(original)
    union_rewrites = " ".join(proposed)
    union_compact = _compact(union_rewrites)
    union_tokens = set(_tokens(union_rewrites, min_len=3))
    for term in orig_terms:
        if not _term_present(term, union_compact, union_tokens):
            if len(term.split()) > 1 or len(term) >= 6:
                all_issues.append(f"missing_entities:{term}")
            else:
                all_issues.append(f"missing_keywords:{term}")

    orig_types = _detect_question_types(original)
    union_types = _detect_question_types(union_rewrites)
    for qtype in orig_types:
        if qtype not in union_types:
            all_reasons.append(f"question_type_drift:{qtype}")

    for rew in proposed:
        for pattern in _INTENT_COLLAPSE_PATTERNS:
            if pattern.search(rew) and not pattern.search(original):
                all_reasons.append("intent_collapse")

    if len(proposed) == 1:
        all_reasons.extend(_validate_single_rewrite(original, proposed[0]))
    elif _LOCATION_MARKERS.search(original) and _PROCESS_MARKERS.search(union_rewrites):
        if not _LOCATION_MARKERS.search(union_rewrites):
            all_reasons.append("location_to_process_drift")

    # Deduplicate
    all_reasons = list(dict.fromkeys(all_reasons))
    all_issues = list(dict.fromkeys(all_issues + all_reasons))

    return {
        "valid": len(all_reasons) == 0 and len(all_issues) == 0,
        "reasons": all_reasons,
        "issues": all_issues,
        "expected_intents": expected_intents,
        "proposed_count": len(proposed),
    }


def apply_rewrite_guard(original: str, proposed: list[str]) -> tuple[list[str], dict[str, Any]]:
    validation = validate_rewrites(original, proposed)
    fallback = not validation["valid"]
    final = [original] if fallback else proposed

    audit = {
        "original_query": original,
        "proposed_rewrites": proposed,
        "final_rewrites": final,
        "validation_passed": validation["valid"],
        "fallback_triggered": fallback,
        "failure_reasons": validation.get("reasons", []),
        "issues": validation.get("issues", []),
        "expected_intents": validation.get("expected_intents"),
        "proposed_count": validation.get("proposed_count"),
    }

    if fallback:
        logger.warning(
            "rewrite_guard fallback | original=%r proposed=%r reasons=%s",
            original,
            proposed,
            validation.get("reasons"),
        )
    else:
        logger.info(
            "rewrite_guard passed | original=%r final=%r",
            original,
            final,
        )

    return final, audit
