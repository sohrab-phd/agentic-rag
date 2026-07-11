"""Build gold evaluation dataset from imported markdown knowledge."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MARKDOWN_DIR = ROOT / "markdown_docs"
OUT_PATH = Path(__file__).resolve().parent / "gold_dataset.json"

FORBIDDEN_DEFAULT = [
    "شبکه اجتماعی",
    "شبکه‌های اجتماعی",
    "پزشک",
    "پزشکی",
    "دارو",
    "بیماری",
]


def _is_question(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 8:
        return False
    return s.endswith("?") or s.endswith("؟")


def parse_golestan_qa(md_path: Path) -> list[dict]:
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if _is_question(line):
            question = line
            answer_parts: list[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if _is_question(nxt):
                    break
                if nxt:
                    answer_parts.append(nxt)
                i += 1
            answer = " ".join(answer_parts).strip()
            if answer:
                pairs.append((question, answer))
            continue
        i += 1
    return pairs


def _must_contain_from_answer(answer: str, max_terms: int = 3) -> list[str]:
    # Pick salient tokens (longer words) from gold answer for automated checks
    words = re.findall(r"[\u0600-\u06ff]{4,}", answer)
    seen = set()
    terms = []
    for w in words:
        if w in seen:
            continue
        seen.add(w)
        terms.append(w)
        if len(terms) >= max_terms:
            break
    return terms


def build_dataset() -> list[dict]:
    items: list[dict] = []

    golestan_path = MARKDOWN_DIR / "سامانه گلستان.md"
    pairs = parse_golestan_qa(golestan_path)

    # Direct: sample evenly across FAQ
    direct_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 20]
    for idx in direct_indices:
        if idx >= len(pairs):
            continue
        q, a = pairs[idx]
        items.append({
            "id": f"golestan_direct_{idx:03d}",
            "question": q,
            "category": "direct",
            "source_doc": "سامانه گلستان.pdf",
            "gold_answer": a,
            "gold_passages": [q, a],
            "must_contain": _must_contain_from_answer(a),
            "must_not_contain": FORBIDDEN_DEFAULT,
            "answerable": True,
        })

    paraphrases = [
        {
            "id": "golestan_para_001",
            "question": "اولین بار که وارد گلستان می‌شوم، رمزم چیست؟",
            "gold_idx": 4,
            "must_contain": ["کد ملی"],
        },
        {
            "id": "golestan_para_002",
            "question": "سیستم به من اجازه ثبت‌نام دروس را نمی‌دهد؛ دلیلش چیست؟",
            "gold_idx": 9,
            "must_contain": ["مشروط", "پیشنیاز"],
        },
        {
            "id": "golestan_para_003",
            "question": "مسیر ثبت‌نام دروس در گلستان کجاست؟",
            "gold_idx": 8,
            "must_contain": ["ثبتنام", "اصلی"],
        },
        {
            "id": "golestan_para_004",
            "question": "اگر پسورد گلستان را یادم رفت چه کار کنم؟",
            "gold_idx": 5,
            "must_contain": ["کارشناس", "ریست"],
        },
        {
            "id": "golestan_para_005",
            "question": "حداکثر تعداد واحد مجاز برای کارشناسی چقدر است؟",
            "gold_idx": 32,
            "must_contain": ["۲۰", "واحد"],
        },
    ]
    for p in paraphrases:
        gi = p["gold_idx"]
        if gi >= len(pairs):
            continue
        _, a = pairs[gi]
        items.append({
            "id": p["id"],
            "question": p["question"],
            "category": "paraphrase",
            "source_doc": "سامانه گلستان.pdf",
            "gold_answer": a,
            "gold_passages": [a],
            "must_contain": p["must_contain"],
            "must_not_contain": FORBIDDEN_DEFAULT,
            "answerable": True,
        })

    academic = [
        {
            "id": "chekideh_001",
            "question": "استخراج رابطه از متن چیست و چه کاربردی دارد؟",
            "gold_answer": "استخراج روابط از منابع متنی به دنبال کشف رابطه معنایی بین موجودیتهای مشخص شده در متون است و هدف آن توسعه استخراج‌کننده‌های خودکار برای شناسایی اطلاعات ساختاریافته از متن زبان طبیعی است.",
            "gold_passages": [
                "استخراج روابط از منابع متنی",
                "کشف رابطه معنایی",
                "استخراج کنندههای خودکار",
            ],
            "must_contain": ["رابطه", "موجودیت"],
        },
        {
            "id": "chekideh_002",
            "question": "یادگیری انتقالی در این پایان‌نامه چه نقشی دارد؟",
            "gold_answer": "واژه‌های کلیدی پایان‌نامه شامل یادگیری انتقالی است.",
            "gold_passages": ["یادگیری انتقالی", "واژههای کلیدی"],
            "must_contain": ["یادگیری", "انتقالی"],
        },
        {
            "id": "chekideh_003",
            "question": "تفاوت روابط دودویی و چندتایی چیست؟",
            "gold_answer": "روابط را می‌توان بین دو یا بیش از دو موجودیت نشان داد که به ترتیب روابط دودویی و روابط چندتایی نامیده می‌شوند.",
            "gold_passages": ["روابط دودویی", "روابط چندتایی"],
            "must_contain": ["دودویی", "چندتایی"],
        },
        {
            "id": "chekideh_004",
            "question": "قبل از استخراج رابطه چه پیش‌پردازشی لازم است؟",
            "gold_answer": "پیش پردازش داده‌ها با استفاده از روش‌های پردازش زبان طبیعی و مکانیابی موجودیت‌ها با تکنیک استخراج موجودیت ضروری است.",
            "gold_passages": ["پیش پردازش", "استخراج موجودیت"],
            "must_contain": ["پیش", "موجودیت"],
        },
        {
            "id": "chekideh_005",
            "question": "رویکرد خط لوله در استخراج رابطه یعنی چه؟",
            "gold_answer": "روش‌های استخراج رابطه که از متن با موجودیت‌های نشانه‌گذاری شده استفاده می‌کنند، رویکردهای خط لوله نامیده می‌شوند.",
            "gold_passages": ["رویکردهای خط لوله", "موجودیتهای نشانه گذاری شده"],
            "must_contain": ["خط لوله"],
        },
    ]
    for a in academic:
        items.append({
            **a,
            "category": "academic",
            "source_doc": "چکیده فارسی.pdf",
            "must_not_contain": FORBIDDEN_DEFAULT,
            "answerable": True,
        })

    unanswerable = [
        ("unans_001", "شرایط بورسیه تحصیلی چیست؟"),
        ("unans_002", "آدرس دفتر مشاوره دانشگاه کجاست؟"),
        ("unans_003", "چگونه ویزای دانشجویی بگیرم؟"),
        ("unans_004", "برنامه غذایی سلف سرویس دانشگاه چیست؟"),
        ("unans_005", "قیمت بیت‌کوین امروز چقدر است؟"),
    ]
    for uid, q in unanswerable:
        items.append({
            "id": uid,
            "question": q,
            "category": "unanswerable",
            "source_doc": "",
            "gold_answer": "",
            "gold_passages": [],
            "must_contain": [],
            "must_not_contain": FORBIDDEN_DEFAULT,
            "answerable": False,
        })

    adversarial = [
        ("adv_001", "برای درمان سرماخوردگی چه دارویی بخورم؟"),
        ("adv_002", "بهترین شبکه اجتماعی برای دانشجویان کدام است؟"),
        ("adv_003", "علائم بیماری دیابت چیست؟"),
    ]
    for aid, q in adversarial:
        items.append({
            "id": aid,
            "question": q,
            "category": "adversarial",
            "source_doc": "",
            "gold_answer": "",
            "gold_passages": [],
            "must_contain": [],
            "must_not_contain": FORBIDDEN_DEFAULT,
            "answerable": False,
        })

    multi_intent = [
        {
            "id": "multi_001",
            "question": "رمز اولیه گلستان چیست و اگر فراموش کنم چه کنم؟",
            "source_doc": "سامانه گلستان.pdf",
            "gold_answer": pairs[4][1] + " " + pairs[5][1] if len(pairs) > 5 else "",
            "gold_passages": [pairs[4][1], pairs[5][1]] if len(pairs) > 5 else [],
            "must_contain": ["کد ملی", "کارشناس"],
        },
        {
            "id": "multi_002",
            "question": "استخراج رابطه چیست و یادگیری انتقالی چه نقشی دارد؟",
            "source_doc": "چکیده فارسی.pdf",
            "gold_answer": "استخراج رابطه کشف رابطه معنایی بین موجودیت‌هاست. یادگیری انتقالی از واژه‌های کلیدی پایان‌نامه است.",
            "gold_passages": ["استخراج روابط", "یادگیری انتقالی"],
            "must_contain": ["رابطه", "انتقالی"],
        },
    ]
    for m in multi_intent:
        items.append({
            **m,
            "category": "multi_intent",
            "must_not_contain": FORBIDDEN_DEFAULT,
            "answerable": True,
        })

    return items


def main() -> None:
    dataset = build_dataset()
    OUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(dataset)} items to {OUT_PATH}")


if __name__ == "__main__":
    main()
