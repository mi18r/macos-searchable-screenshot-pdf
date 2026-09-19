from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import ocr


class FakeRectangle:
    def __init__(self, x, y, width, height):
        self._bounding_box = (
            SimpleNamespace(x=x, y=y),
            SimpleNamespace(width=width, height=height),
        )

    def boundingBox(self):
        return self._bounding_box


class FakeCandidate:
    def __init__(self, text):
        self._text = text
        self.ranges = []

    def string(self):
        return self._text

    def confidence(self):
        return 0.92

    def boundingBoxForRange_error_(self, ns_range, _error):
        start, length = ns_range
        self.ranges.append((start, length))
        return FakeRectangle(start / 20, 0.25, length / 20, 0.10), None


class FakeRecognitionRequest:
    latest = None

    @classmethod
    def alloc(cls):
        return cls()

    def initWithCompletionHandler_(self, completion_handler):
        self.completion_handler = completion_handler
        type(self).latest = self
        return self

    def setRecognitionLevel_(self, level):
        self.level = level

    def setRecognitionLanguages_(self, languages):
        self.languages = languages

    def setUsesLanguageCorrection_(self, enabled):
        self.language_correction = enabled


class FakeImageRequestHandler:
    perform_result = (True, None)
    completion_error = None

    @classmethod
    def alloc(cls):
        return cls()

    def initWithData_options_(self, data, options):
        self.data = data
        self.options = options
        return self

    def performRequests_error_(self, _requests, _error):
        if self.completion_error is not None:
            FakeRecognitionRequest.latest.completion_handler(
                SimpleNamespace(results=lambda: []), self.completion_error
            )
        return self.perform_result


class FakeNSData:
    @staticmethod
    def dataWithContentsOfFile_(_path):
        return b"image data"


class FakeVisionError:
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class OCRTests(unittest.TestCase):
    def test_word_ranges_keep_searchable_punctuation_with_word(self):
        self.assertEqual(
            ocr.word_ranges("You are here."),
            [("You", 0, 3), ("are", 4, 3), ("here.", 8, 5)],
        )

    def test_candidate_to_words_returns_individual_boxes(self):
        words = ocr.candidate_to_words(FakeCandidate("You are here."))
        self.assertEqual([word.text for word in words], ["You", "are", "here."])
        self.assertEqual(words[1].box.x, 0.20)
        self.assertEqual(words[1].box.y, 0.25)
        self.assertAlmostEqual(words[1].box.width, 0.15)
        self.assertAlmostEqual(words[1].box.height, 0.10)

    def test_candidate_to_words_uses_utf16_ranges_after_non_bmp_character(self):
        candidate = FakeCandidate("🐍 hello")

        ocr.candidate_to_words(candidate)

        self.assertEqual(candidate.ranges, [(0, 2), (3, 5)])

    def test_perform_request_preserves_perform_requests_error(self):
        perform_error = FakeVisionError("NSOSStatusErrorDomain Code=-6662")
        FakeImageRequestHandler.perform_result = (False, perform_error)
        FakeImageRequestHandler.completion_error = None

        with patch.dict(
            sys.modules,
            {
                "Cocoa": SimpleNamespace(NSData=FakeNSData),
                "Vision": SimpleNamespace(
                    VNImageRequestHandler=FakeImageRequestHandler,
                    VNRecognizeTextRequest=FakeRecognitionRequest,
                ),
            },
        ):
            success, error, observations = ocr._perform_request(Path("image.png"))

        self.assertFalse(success)
        self.assertIs(error, perform_error)
        self.assertIn("NSOSStatusErrorDomain Code=-6662", str(error))
        self.assertEqual(observations, [])

    def test_perform_request_preserves_completion_handler_error(self):
        completion_error = FakeVisionError("completion failed")
        FakeImageRequestHandler.perform_result = (True, None)
        FakeImageRequestHandler.completion_error = completion_error

        with patch.dict(
            sys.modules,
            {
                "Cocoa": SimpleNamespace(NSData=FakeNSData),
                "Vision": SimpleNamespace(
                    VNImageRequestHandler=FakeImageRequestHandler,
                    VNRecognizeTextRequest=FakeRecognitionRequest,
                ),
            },
        ):
            success, error, observations = ocr._perform_request(Path("image.png"))

        self.assertTrue(success)
        self.assertIs(error, completion_error)
        self.assertEqual(observations, [])

    def test_extract_words_reports_vision_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "sample.png"
            Image.new("RGB", (10, 10), "white").save(image)
            with patch.object(
                ocr, "_perform_request", return_value=(False, "pixel buffer failed", [])
            ):
                with self.assertRaisesRegex(
                    ocr.OCRProcessingError, "pixel buffer failed"
                ):
                    ocr.extract_words(image)
