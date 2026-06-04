# src/pdf_reader.py
# ---------------------------------------------------------------------------
# Turns a PDF into plain text.
# The provided records are SCANNED (every page is an image, including handwritten
# nursing notes), so ordinary text extractors return nothing. Gemini reads the PDF
# directly as an image-capable model, which handles scans and handwriting.
# The result is cached to disk so we only pay for the upload once.
# ---------------------------------------------------------------------------

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

load_dotenv()
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Multimodal model used only for reading the PDF pages.
PDF_MODEL = "gemini-3-flash-preview"


def read_pdf(pdf_path: str) -> str:
    """Extract all text from every page. Caches to outputs/<patient>/raw_text.txt."""
    patient = Path(pdf_path).parent.name              # patients/patient_2/x.pdf -> patient_2
    cache = Path("outputs") / patient / "raw_text.txt"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists():                                # already read once -> reuse
        print(f"  (using cached text: {cache})")
        return cache.read_text()

    print(f"  uploading {pdf_path} to Gemini ...")
    uploaded = _client.files.upload(file=pdf_path)
    resp = _client.models.generate_content(
        model=PDF_MODEL,
        contents=[
            uploaded,
            "Transcribe ALL text from every page of this hospital record. "
            "Mark each page as === PAGE 1 ===, === PAGE 2 === and so on. "
            "Include handwritten notes, printed text, lab values, drug names and dates. "
            "Write [ILLEGIBLE] where you cannot read something. Plain text only.",
        ],
    )
    cache.write_text(resp.text)
    print(f"  saved raw text: {cache}")
    return resp.text
