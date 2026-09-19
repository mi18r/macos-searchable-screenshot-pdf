from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from discovery import ClassBatch
from ocr import OCRWord, extract_words
from ocr_cache import (
    CacheRecord,
    FileFingerprint,
    OCRCache,
    OCR_CONFIGURATION_VERSION,
    OCR_RECORD_SCHEMA_VERSION,
    fingerprint_file,
    source_key,
)


@dataclass(frozen=True)
class IncrementalStats:
    reused: int
    new: int
    changed: int
    stale: int
    removed: int


@dataclass(frozen=True)
class PreparedOCR:
    source_root: Path
    words_by_path: dict[Path, tuple[OCRWord, ...]]
    fingerprints_by_path: dict[Path, FileFingerprint]
    pending_records: tuple[CacheRecord, ...]
    keep_relative_paths: frozenset[str]
    stats: IncrementalStats

    def words_for(self, image_path: Path) -> list[OCRWord]:
        return list(self.words_by_path[image_path])


ProgressCallback = Callable[[int, int, Path, str], None]
OCRFunction = Callable[[Path], list[OCRWord]]


class InputChangedError(RuntimeError):
    pass


def validate_prepared_ocr(prepared: PreparedOCR) -> None:
    for image_path, expected in prepared.fingerprints_by_path.items():
        try:
            if image_path.is_symlink():
                raise OSError("path became a symbolic link")
            actual = fingerprint_file(prepared.source_root, image_path)
        except (OSError, ValueError) as error:
            raise InputChangedError(
                f"Screenshot changed during build: {image_path}: {error}"
            ) from error
        if actual != expected:
            raise InputChangedError(f"Screenshot changed during build: {image_path}")


def apply_prepared_ocr(cache: OCRCache, prepared: PreparedOCR) -> int:
    for record in prepared.pending_records:
        cache.upsert(record)
    return cache.prune(prepared.source_root, set(prepared.keep_relative_paths))


def prepare_ocr(
    source_root: Path,
    batches: list[ClassBatch],
    cache: OCRCache,
    ocr_function: OCRFunction = extract_words,
    progress: ProgressCallback | None = None,
) -> PreparedOCR:
    source_root = Path(source_root)
    image_paths = tuple(image for batch in batches for image in batch.images)
    total = len(image_paths)
    words_by_path: dict[Path, tuple[OCRWord, ...]] = {}
    fingerprints_by_path: dict[Path, FileFingerprint] = {}
    pending_records: list[CacheRecord] = []
    counts = {"reused": 0, "new": 0, "changed": 0, "stale": 0}
    current_relative_paths: set[str] = set()

    for current, image_path in enumerate(image_paths, start=1):
        image_path = Path(image_path)
        fingerprint = fingerprint_file(source_root, image_path)
        fingerprints_by_path[image_path] = fingerprint
        current_relative_paths.add(fingerprint.relative_path)
        record = cache.get(source_root, fingerprint.relative_path)

        if record is None:
            classification = "new"
        elif (
            record.record_schema_version != OCR_RECORD_SCHEMA_VERSION
            or record.ocr_configuration_version != OCR_CONFIGURATION_VERSION
        ):
            classification = "stale"
        elif (
            record.fingerprint.size != fingerprint.size
            or record.fingerprint.mtime_ns != fingerprint.mtime_ns
        ):
            classification = "changed"
        else:
            classification = "reused"

        if classification == "reused":
            words = record.words
        else:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            words = tuple(ocr_function(image_path))
            pending_records.append(
                CacheRecord(
                    source_root=source_key(source_root),
                    fingerprint=fingerprint,
                    image_width=image_width,
                    image_height=image_height,
                    record_schema_version=OCR_RECORD_SCHEMA_VERSION,
                    ocr_configuration_version=OCR_CONFIGURATION_VERSION,
                    words=words,
                )
            )

        words_by_path[image_path] = tuple(words)
        counts[classification] += 1
        if progress is not None:
            progress(current, total, image_path, classification)

    removed = cache.count_prunable(source_root, current_relative_paths)
    return PreparedOCR(
        source_root=source_root,
        words_by_path=words_by_path,
        fingerprints_by_path=fingerprints_by_path,
        pending_records=tuple(pending_records),
        keep_relative_paths=frozenset(current_relative_paths),
        stats=IncrementalStats(
            reused=counts["reused"],
            new=counts["new"],
            changed=counts["changed"],
            stale=counts["stale"],
            removed=removed,
        ),
    )
