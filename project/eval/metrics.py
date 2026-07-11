"""Automated metrics for document-grounded RAG evaluation."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any


def normalize_fa(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"[\s\u200c]+", "", text)
    text = text.replace("‌", "")
    return text.lower()


def source_stem(name: str) -> str:
    return Path(name).stem if name else ""


def doc_names_match(expected: str, retrieved: str) -> bool:
    if not expected or not retrieved:
        return False
    exp = normalize_fa(source_stem(expected))
    got = normalize_fa(source_stem(retrieved))
    return exp == got or exp in got or got in exp


def parse_retrieved_chunks(tool_output: str) -> list[dict[str, Any]]:
    if not tool_output or tool_output in {
        "NO_RELEVANT_CHUNKS",
        "NO_PARENT_DOCUMENT",
        "NO_PARENT_DOCUMENTS",
    } or tool_output.startswith("RETRIEVAL_ERROR"):
        return []

    chunks: list[dict[str, Any]] = []
    for block in tool_output.split("\n\n"):
        lines = block.strip().splitlines()
        if not lines:
            continue
        meta: dict[str, str] = {}
        content_lines: list[str] = []
        for line in lines:
            if line.startswith("Parent ID:"):
                meta["parent_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("File Name:"):
                meta["source"] = line.split(":", 1)[1].strip()
            elif line.startswith("Content:"):
                content_lines.append(line.split(":", 1)[1].strip())
            else:
                content_lines.append(line.strip())
        if meta or content_lines:
            chunks.append({**meta, "text": "\n".join(content_lines).strip()})
    return chunks


def token_set(text: str) -> set[str]:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("ي", "ی").replace("ك", "ک").lower()
    return {t for t in re.split(r"[^\w\u0600-\u06ff]+", text) if len(t) >= 3}


def overlap_ratio(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def passage_recall(gold_passages: list[str], chunks: list[dict]) -> bool:
    if not gold_passages or not chunks:
        return False
    for gold in gold_passages:
        for ch in chunks:
            if overlap_ratio(gold, ch.get("text", "")) >= 0.25:
                return True
            if overlap_ratio(gold, ch.get("text", "")) >= 0.15 and len(normalize_fa(gold)) >= 20:
                # short gold snippets: check substring
                ng = normalize_fa(gold)
                nt = normalize_fa(ch.get("text", ""))
                if ng and ng[: min(30, len(ng))] in nt:
                    return True
    return False


def source_hit(expected_doc: str, chunks: list[dict]) -> bool:
    if not expected_doc:
        return True
    return any(doc_names_match(expected_doc, ch.get("source", "")) for ch in chunks)


def must_contain_pass(answer: str, terms: list[str]) -> tuple[bool, list[str]]:
    missing = []
    na = normalize_fa(answer)
    for term in terms:
        if normalize_fa(term) not in na:
            missing.append(term)
    return len(missing) == 0, missing


def forbidden_violation(answer: str, forbidden: list[str], chunk_text: str) -> tuple[bool, list[str]]:
    hits = []
    na = normalize_fa(answer)
    nc = normalize_fa(chunk_text)
    for term in forbidden:
        nt = normalize_fa(term)
        if nt in na and nt not in nc:
            hits.append(term)
    return len(hits) > 0, hits


def abstention_like(answer: str) -> bool:
    markers = [
        "اطلاعات کافی",
        "در اسناد",
        "منبع مرتبط",
        "یافت نشد",
        "موجود نیست",
        "نمی‌توان",
        "نمیتوان",
        "پاسخی ندار",
        "مرتبطی یافت",
    ]
    na = normalize_fa(answer)
    return any(normalize_fa(m) in na for m in markers) or len(answer.strip()) < 80


def classify_failure(
    item: dict,
    metrics: dict,
    answerable: bool,
) -> str:
    if not answerable:
        if metrics.get("forbidden_violation"):
            return "hallucination_on_unanswerable"
        if not metrics.get("abstention_like") and len(item.get("final_answer", "")) > 120:
            return "overconfident_unanswerable"
        return "ok_unanswerable"

    if not metrics.get("source_hit"):
        return "retrieval_wrong_source"
    if not metrics.get("passage_recall"):
        return "retrieval_miss"
    if not metrics.get("must_contain_pass"):
        if metrics.get("passage_recall"):
            return "generation_incomplete_or_wrong"
        return "retrieval_and_generation"
    if metrics.get("forbidden_violation"):
        return "hallucination"
    if metrics.get("semantic_similarity") is not None and metrics.get("semantic_similarity", 1.0) < 0.35:
        return "generation_semantic_drift"
    return "ok"


def compute_metrics(
    gold: dict,
    answer: str,
    retrieved_chunks: list[dict],
    chunk_text: str,
    semantic_similarity: float | None = None,
) -> dict:
    answerable = gold.get("answerable", True)
    must_ok, must_missing = must_contain_pass(answer, gold.get("must_contain", []))
    forb_viol, forb_hits = forbidden_violation(
        answer, gold.get("must_not_contain", []), chunk_text
    )

    m = {
        "source_hit": source_hit(gold.get("source_doc", ""), retrieved_chunks) if answerable else None,
        "passage_recall": passage_recall(gold.get("gold_passages", [gold.get("gold_answer", "")]), retrieved_chunks) if answerable else None,
        "must_contain_pass": must_ok if answerable else None,
        "must_contain_missing": must_missing if answerable else [],
        "forbidden_violation": forb_viol,
        "forbidden_hits": forb_hits,
        "abstention_like": abstention_like(answer) if not answerable else None,
        "semantic_similarity": semantic_similarity,
        "retrieved_count": len(retrieved_chunks),
    }

    if answerable:
        m["passed"] = (
            m["source_hit"]
            and m["passage_recall"]
            and must_ok
            and not forb_viol
            and (semantic_similarity is None or semantic_similarity >= 0.30)
        )
    else:
        m["passed"] = not forb_viol and (m["abstention_like"] or len(answer.strip()) < 150)

    m["failure_class"] = classify_failure(gold, m, answerable)
    return m
