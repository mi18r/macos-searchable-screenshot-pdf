import unittest

from pypdf import PdfReader
from reportlab.pdfbase.pdfmetrics import getAscentDescent

from ocr import NormalizedBox, OCRWord
from pdf_layer import (
    PAGE_HEIGHT,
    PAGE_WIDTH,
    calculate_image_placement,
    create_text_layer,
    is_usable_ocr_word,
    map_box_to_pdf,
)


class PDFLayerTests(unittest.TestCase):
    def test_landscape_image_is_centered_without_distortion(self):
        placement = calculate_image_placement(2000, 1000)

        self.assertAlmostEqual(placement.width / placement.height, 2.0)
        self.assertAlmostEqual(placement.x, 0.0)
        self.assertAlmostEqual(
            placement.y, (PAGE_HEIGHT - placement.height) / 2
        )

    def test_portrait_image_is_centered_without_distortion(self):
        placement = calculate_image_placement(1000, 2000)

        self.assertAlmostEqual(placement.width / placement.height, 0.5)
        self.assertAlmostEqual(placement.y, 0.0)
        self.assertAlmostEqual(
            placement.x, (PAGE_WIDTH - placement.width) / 2
        )

    def test_normalized_box_maps_into_rendered_image_area(self):
        placement = calculate_image_placement(1000, 1000)
        mapped = map_box_to_pdf(NormalizedBox(0.25, 0.50, 0.20, 0.10), placement)

        self.assertAlmostEqual(mapped.x, placement.x + placement.width * 0.25)
        self.assertAlmostEqual(mapped.y, placement.y + placement.height * 0.50)
        self.assertAlmostEqual(mapped.width, placement.width * 0.20)
        self.assertAlmostEqual(mapped.height, placement.height * 0.10)

    def test_hidden_words_fill_their_mapped_pdf_rectangles(self):
        placement = calculate_image_placement(1000, 1000)
        words = [
            OCRWord("Alpha", NormalizedBox(0.10, 0.20, 0.15, 0.08), 0.9),
            OCRWord("Beta", NormalizedBox(0.60, 0.70, 0.15, 0.08), 0.9),
        ]
        page = PdfReader(create_text_layer(words, placement)).pages[0]
        emitted_text = {}

        def visitor(text, _cm, tm, _font, font_size):
            cleaned = text.strip()
            if cleaned:
                emitted_text[cleaned] = (tm[4], tm[5], font_size)

        page.extract_text(visitor_text=visitor)

        for word in words:
            with self.subTest(word=word.text):
                expected_x = placement.x + word.box.x * placement.width
                expected_y = placement.y + word.box.y * placement.height
                expected_height = word.box.height * placement.height
                baseline_x, baseline_y, font_size = emitted_text[word.text]
                ascent, descent = getAscentDescent("Helvetica", font_size)

                self.assertAlmostEqual(baseline_x, expected_x, places=4)
                self.assertAlmostEqual(baseline_y + descent, expected_y, places=4)
                self.assertAlmostEqual(
                    baseline_y + ascent, expected_y + expected_height, places=4
                )
                self.assertAlmostEqual(ascent - descent, expected_height, places=4)

    def test_zero_or_negative_image_dimensions_are_rejected(self):
        for dimensions in ((0, 100), (-1, 100), (100, 0), (100, -1)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    calculate_image_placement(*dimensions)

    def test_empty_or_non_positive_words_are_skipped(self):
        placement = calculate_image_placement(1000, 1000)
        words = [
            OCRWord("", NormalizedBox(0.10, 0.20, 0.15, 0.08), 0.9),
            OCRWord("NoWidth", NormalizedBox(0.10, 0.20, 0.0, 0.08), 0.9),
            OCRWord("NoHeight", NormalizedBox(0.10, 0.20, 0.15, -0.08), 0.9),
            OCRWord("Visible", NormalizedBox(0.10, 0.20, 0.15, 0.08), 0.9),
        ]

        extracted = PdfReader(create_text_layer(words, placement)).pages[0].extract_text()

        self.assertEqual(extracted.strip(), "Visible")

    def test_usable_word_requires_text_and_positive_box_dimensions(self):
        words = [
            OCRWord("Visible", NormalizedBox(0.10, 0.20, 0.15, 0.08), 0.9),
            OCRWord(" ", NormalizedBox(0.10, 0.20, 0.15, 0.08), 0.9),
            OCRWord("NoWidth", NormalizedBox(0.10, 0.20, 0.0, 0.08), 0.9),
            OCRWord("NoHeight", NormalizedBox(0.10, 0.20, 0.15, -0.08), 0.9),
        ]

        self.assertEqual(
            [is_usable_ocr_word(word) for word in words],
            [True, False, False, False],
        )
