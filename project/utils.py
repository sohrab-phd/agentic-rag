import os
import re
import shutil
import unicodedata
import config
from pathlib import Path
import glob
import tiktoken


def clear_directory_contents(directory: Path) -> None:
    """Delete everything under directory but not the directory itself (safe for Docker volume / bind mount roots)."""
    directory = Path(directory)
    if not directory.is_dir():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _normalize_persian_text(text: str) -> str:
    """Light normalization after PDF extraction."""
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u200c", "\u200c")  # keep ZWNJ
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_with_pdf2text_arabic(pdf_path: Path) -> str:
    from pdf2text_arabic import extract_pdf

    return extract_pdf(
        str(pdf_path),
        ocr_strategy=config.PDF_OCR_STRATEGY,
        detect_footer=True,
    )


def _extract_with_pymupdf4llm(pdf_path: Path) -> str:
    import pymupdf.layout  # noqa: F401
    import pymupdf
    import pymupdf4llm

    doc = pymupdf.open(pdf_path)
    try:
        md = pymupdf4llm.to_markdown(
            doc,
            header=False,
            footer=False,
            page_separators=True,
            ignore_images=True,
            write_images=False,
            image_path=None,
        )
    finally:
        doc.close()
    return md.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF using the configured Persian-aware pipeline."""
    extractor = config.PDF_EXTRACTOR.lower()
    if extractor == "pdf2text_arabic":
        try:
            return _normalize_persian_text(_extract_with_pdf2text_arabic(pdf_path))
        except Exception as e:
            print(f"Warning: pdf2text_arabic failed for {pdf_path.name}: {e}. Falling back to pymupdf4llm.")
    return _normalize_persian_text(_extract_with_pymupdf4llm(pdf_path))


def pdf_to_markdown(pdf_path, output_dir):
    """Convert PDF to markdown/plain-text file for the RAG indexer."""
    pdf_path = Path(pdf_path)
    text = extract_pdf_text(pdf_path)
    output_path = Path(output_dir) / pdf_path.stem
    Path(output_path).with_suffix(".md").write_text(text, encoding="utf-8")


def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in map(Path, glob.glob(path_pattern)):
        md_path = (output_dir / pdf_path.stem).with_suffix(".md")
        if overwrite or not md_path.exists():
            pdf_to_markdown(pdf_path, output_dir)


def estimate_context_tokens(messages: list) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoding.encode(str(msg.content))) for msg in messages if hasattr(msg, "content") and msg.content)
