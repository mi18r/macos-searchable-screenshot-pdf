from dataclasses import dataclass
from pathlib import Path
import re

from PIL import Image


class OCRProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class NormalizedBox:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class OCRWord:
    text: str
    box: NormalizedBox
    confidence: float


def word_ranges(text: str) -> list[tuple[str, int, int]]:
    return [
        (match.group(0), match.start(), match.end() - match.start())
        for match in re.finditer(r"\S+", text)
    ]


def _rect_values(rectangle) -> tuple[float, float, float, float]:
    origin, size = rectangle.boundingBox()
    return float(origin.x), float(origin.y), float(size.width), float(size.height)


def candidate_to_words(candidate) -> list[OCRWord]:
    from Foundation import NSMakeRange

    words = []
    confidence = float(candidate.confidence())
    candidate_text = candidate.string()
    for text, start, length in word_ranges(candidate_text):
        utf16_start = len(candidate_text[:start].encode("utf-16-le")) // 2
        utf16_length = len(text.encode("utf-16-le")) // 2
        rectangle, error = candidate.boundingBoxForRange_error_(
            NSMakeRange(utf16_start, utf16_length), None
        )
        if error is not None or rectangle is None:
            continue
        x, y, width, height = _rect_values(rectangle)
        words.append(OCRWord(text, NormalizedBox(x, y, width, height), confidence))
    return words


def _perform_request(image_path: Path):
    from Cocoa import NSData
    from Vision import VNImageRequestHandler, VNRecognizeTextRequest

    observations = []
    completion_errors = []

    def completion_handler(request, error):
        if error is not None:
            completion_errors.append(error)
            return
        observations.extend(request.results() or [])

    request = VNRecognizeTextRequest.alloc().initWithCompletionHandler_(
        completion_handler
    )
    request.setRecognitionLevel_(1)
    request.setRecognitionLanguages_(["en-US"])
    request.setUsesLanguageCorrection_(True)

    data = NSData.dataWithContentsOfFile_(str(image_path))
    if data is None:
        return False, f"Could not read image data: {image_path}", []
    handler = VNImageRequestHandler.alloc().initWithData_options_(data, {})
    success, perform_error = handler.performRequests_error_([request], None)
    error = completion_errors[0] if completion_errors else perform_error
    return bool(success), error, observations


def extract_words(image_path: Path) -> list[OCRWord]:
    image_path = Path(image_path)
    with Image.open(image_path) as image:
        image.verify()
    success, error, observations = _perform_request(image_path)
    if not success or error is not None:
        raise OCRProcessingError(f"Vision OCR failed for {image_path}: {error}")
    words = []
    for observation in observations:
        candidate = observation.topCandidates_(1).firstObject()
        if candidate is not None:
            words.extend(candidate_to_words(candidate))
    return words
