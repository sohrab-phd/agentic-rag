"""
Benchmark Persian PDF extraction methods.
Usage: python pdf_extraction_benchmark.py [path/to/file.pdf]
If no PDF is given, uses docs/sample_persian.pdf (generated if missing).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
OUT = ROOT / "extraction_benchmark.json"

# Representative paragraph from the project's indexed document topic.
SAMPLE_PARAGRAPH = (
    "استخراج روابط از منابع متنی یکی از کاربردهای برنامه‌های خودکار پردازش زبان طبیعی است. "
    "هدف استخراج رابطه، توسعه استخراج‌کننده‌هایی است که موجودیت‌ها و روابط را از متن شناسایی کنند. "
    "قبل از استخراج روابط از متن پردازش‌نشده، پردازش زبان طبیعی الزم است. "
    "روش‌های یادگیری عمیق و یادگیری انتقالی در خط لوله استخراج اطلاعات به کار می‌روند."
)


def _quality_metrics(text: str) -> dict:
    persian = len(re.findall(r"[\u0600-\u06FF]", text))
    merged_glitch = len(re.findall(r"[اآی]{2,}|[\u0600-\u06FF]{1,2} [\u0600-\u06FF]{1,2} [\u0600-\u06FF]{1,2}", text))
    bad_patterns = sum(
        1
        for p in [
            r"فرآ ین",
            r"دادهاستخراج",
            r"کاربردتوسط",
            r"کنندهکی",
            r"ی ها ت",
            r"استی ی",
        ]
        if re.search(p, text)
    )
    good_patterns = sum(
        1
        for p in [
            r"استخراج روابط",
            r"پردازش زبان طبیعی",
            r"منابع متنی",
            r"یادگیری عمیق",
        ]
        if re.search(p, text)
    )
    return {
        "length": len(text),
        "persian_char_count": persian,
        "known_bad_pattern_hits": bad_patterns,
        "known_good_pattern_hits": good_patterns,
        "merged_glitch_hits": merged_glitch,
    }


def ensure_sample_pdf(path: Path) -> None:
    if path.exists():
        return
    import fitz

    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    # RTL paragraph block
    rect = fitz.Rect(50, 80, 545, 300)
    page.insert_textbox(
        rect,
        SAMPLE_PARAGRAPH,
        fontsize=12,
        fontname="tiro",  # built-in Unicode font in recent PyMuPDF
        align=fitz.TEXT_ALIGN_RIGHT,
    )
    doc.save(path)
    doc.close()


def extract_pymupdf4llm_current(pdf: Path) -> str:
    import pymupdf.layout  # noqa: F401
    import pymupdf4llm

    doc = __import__("pymupdf").open(pdf)
    md = pymupdf4llm.to_markdown(
        doc,
        header=False,
        footer=False,
        page_separators=True,
        ignore_images=True,
        write_images=False,
        image_path=None,
    )
    return md.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")


def extract_pymupdf_raw(pdf: Path) -> str:
    import fitz

    doc = fitz.open(pdf)
    parts = []
    for page in doc:
        parts.append(page.get_text("text", sort=True))
    return "\n".join(parts)


def extract_pymupdf_blocks(pdf: Path) -> str:
    import fitz

    doc = fitz.open(pdf)
    parts = []
    for page in doc:
        blocks = page.get_text("blocks", sort=True)
        for b in blocks:
            if b[6] == 0:  # text block
                parts.append(b[4])
    return "\n".join(parts)


def extract_pdfplumber(pdf: Path) -> str:
    import pdfplumber

    parts = []
    with pdfplumber.open(pdf) as pdf_doc:
        for page in pdf_doc.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def extract_bidi_postprocess(raw: str) -> str:
    import arabic_reshaper
    from bidi.algorithm import get_display

    reshaped = arabic_reshaper.reshape(raw)
    return get_display(reshaped)


def extract_pdf2text_arabic(pdf: Path) -> str:
    from pdf2text_arabic import extract_pdf

    return extract_pdf(str(pdf), ocr_strategy="never")


def run_benchmark(pdf_path: Path) -> dict:
    methods: list[tuple[str, callable]] = [
        ("A_pymupdf4llm_current", extract_pymupdf4llm_current),
        ("B_pymupdf_raw_sort", extract_pymupdf_raw),
        ("C_pymupdf_blocks_sort", extract_pymupdf_blocks),
    ]

    try:
        import pdfplumber  # noqa: F401

        methods.append(("D_pdfplumber", extract_pdfplumber))
    except ImportError:
        pass

    raw = extract_pymupdf_raw(pdf_path)
    methods.append(("E_pymupdf_raw_plus_bidi", lambda p: extract_bidi_postprocess(raw)))

    try:
        import pdf2text_arabic  # noqa: F401

        methods.append(("F_pdf2text_arabic", extract_pdf2text_arabic))
    except ImportError:
        pass

    results = {"pdf": str(pdf_path), "reference_paragraph": SAMPLE_PARAGRAPH, "methods": {}}
    for name, fn in methods:
        try:
            text = fn(pdf_path)
            results["methods"][name] = {
                "text": text[:2000],
                "metrics": _quality_metrics(text),
                "error": None,
            }
        except Exception as e:
            results["methods"][name] = {"text": "", "metrics": {}, "error": str(e)}
    return results


def main() -> None:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else DOCS / "sample_persian.pdf"
    ensure_sample_pdf(pdf)
    report = run_benchmark(pdf)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    for name, data in report["methods"].items():
        m = data.get("metrics", {})
        print(
            name,
            "bad=",
            m.get("known_bad_pattern_hits"),
            "good=",
            m.get("known_good_pattern_hits"),
            "err=",
            data.get("error"),
        )


if __name__ == "__main__":
    main()
