"""PrivAI - Day 8: OCR (read text from scanned PDFs and images), fully offline.

Uses RapidOCR (models are bundled inside the pip package, no internet needed)
and pypdfium2 (turns PDF pages into images, no extra software to install).
"""
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

_engine = None          # loaded only when first needed (it takes a second)


def _get_engine():
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    return _engine


def ocr_image(image) -> str:
    """image can be a file path or a PIL image. Returns the text found."""
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    result, _ = _get_engine()(np.array(image))
    if not result:
        return ""
    return "\n".join(line[1] for line in result)   # each item: [box, text, score]


def pdf_page_count(path) -> int:
    pdf = pdfium.PdfDocument(str(path))
    n = len(pdf)
    pdf.close()
    return n


def ocr_pdf_page(path, page_index, scale=2.0) -> str:
    """Render one PDF page as an image and read it with OCR."""
    pdf = pdfium.PdfDocument(str(path))
    try:
        image = pdf[page_index].render(scale=scale).to_pil()
        return ocr_image(image)
    finally:
        pdf.close()