from dataclasses import dataclass
from io import BytesIO
from typing import Sequence

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import getAscentDescent, stringWidth
from reportlab.pdfgen import canvas

from ocr import NormalizedBox, OCRWord


PAGE_WIDTH, PAGE_HEIGHT = A4


@dataclass(frozen=True)
class ImagePlacement:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class PDFBox:
    x: float
    y: float
    width: float
    height: float


def is_usable_ocr_word(word: OCRWord) -> bool:
    return bool(word.text.strip()) and word.box.width > 0 and word.box.height > 0


def calculate_image_placement(image_width: int, image_height: int) -> ImagePlacement:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")

    scale = min(PAGE_WIDTH / image_width, PAGE_HEIGHT / image_height)
    width = image_width * scale
    height = image_height * scale
    return ImagePlacement(
        x=(PAGE_WIDTH - width) / 2,
        y=(PAGE_HEIGHT - height) / 2,
        width=width,
        height=height,
    )


def map_box_to_pdf(box: NormalizedBox, placement: ImagePlacement) -> PDFBox:
    return PDFBox(
        x=placement.x + box.x * placement.width,
        y=placement.y + box.y * placement.height,
        width=box.width * placement.width,
        height=box.height * placement.height,
    )


def create_text_layer(
    words: Sequence[OCRWord], placement: ImagePlacement
) -> BytesIO:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    for word in words:
        if not is_usable_ocr_word(word):
            continue

        box = map_box_to_pdf(word.box, placement)
        unit_ascent, unit_descent = getAscentDescent("Helvetica", 1.0)
        font_size = box.height / (unit_ascent - unit_descent)
        _ascent, descent = getAscentDescent("Helvetica", font_size)
        natural_width = max(stringWidth(word.text, "Helvetica", font_size), 0.01)
        horizontal_scale = min(100.0, 100.0 * box.width / natural_width)

        text = pdf.beginText()
        text.setTextRenderMode(3)
        text.setFont("Helvetica", font_size)
        text.setHorizScale(horizontal_scale)
        text.setTextOrigin(box.x, box.y - descent)
        text.textOut(word.text)
        pdf.drawText(text)

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer
