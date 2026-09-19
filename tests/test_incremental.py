from pathlib import Path
import os
import shutil
import unittest

from PIL import Image

from discovery import ClassBatch
from incremental import IncrementalStats, apply_prepared_ocr, prepare_ocr
from ocr import NormalizedBox, OCRProcessingError, OCRWord
from ocr_cache import (
    CacheRecord,
    OCRCache,
    OCR_CONFIGURATION_VERSION,
    OCR_RECORD_SCHEMA_VERSION,
    fingerprint_file,
    source_key,
)


class IncrementalOCRTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(__file__).parent / ".tmp" / self._testMethodName
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir.mkdir(parents=True)
        self.cache = OCRCache(self.temp_dir / "cache.sqlite3")

    def tearDown(self):
        self.cache.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _image(
        self,
        source_root: Path,
        relative_path: str,
        *,
        size: tuple[int, int] = (12, 8),
        color: str = "white",
    ) -> Path:
        image_path = source_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(image_path)
        return image_path

    def _batches(self, *groups: tuple[str, tuple[Path, ...]]) -> list[ClassBatch]:
        return [ClassBatch(name, images) for name, images in groups]

    def _apply(self, prepared):
        with self.cache.transaction():
            apply_prepared_ocr(self.cache, prepared)
        return prepared

    @staticmethod
    def _counting_ocr(calls: list[Path]):
        def counting_ocr(path: Path) -> list[OCRWord]:
            calls.append(path)
            return [
                OCRWord(
                    path.stem,
                    NormalizedBox(0.1, 0.2, 0.5, 0.1),
                    1.0,
                )
            ]

        return counting_ocr

    def test_first_preparation_ocr_all_images_in_manifest_order_as_new(self):
        source_root = self.temp_dir / "source"
        first = self._image(source_root, "Class 1/first.png", size=(12, 8))
        second = self._image(source_root, "Class 1/second.png", size=(16, 9))
        third = self._image(source_root, "Class 2/third.png", size=(20, 10))
        batches = self._batches(
            ("Class 1", (first, second)),
            ("Class 2", (third,)),
        )
        original_batches = list(batches)
        calls = []
        progress = []

        prepared = prepare_ocr(
            source_root,
            batches,
            self.cache,
            ocr_function=self._counting_ocr(calls),
            progress=lambda *event: progress.append(event),
        )

        self.assertEqual(calls, [first, second, third])
        self.assertEqual(list(prepared.words_by_path), [first, second, third])
        self.assertTrue(
            all(isinstance(words, tuple) for words in prepared.words_by_path.values())
        )
        self.assertEqual(
            [prepared.words_for(path)[0].text for path in (first, second, third)],
            ["first", "second", "third"],
        )
        self.assertEqual(
            prepared.stats,
            IncrementalStats(reused=0, new=3, changed=0, stale=0, removed=0),
        )
        self.assertEqual(
            progress,
            [
                (1, 3, first, "new"),
                (2, 3, second, "new"),
                (3, 3, third, "new"),
            ],
        )
        self.assertEqual(batches, original_batches)
        self.assertEqual(
            prepared.fingerprints_by_path,
            {
                first: fingerprint_file(source_root, first),
                second: fingerprint_file(source_root, second),
                third: fingerprint_file(source_root, third),
            },
        )
        self.assertEqual(len(prepared.pending_records), 3)
        self.assertEqual(self.cache.count_for_source(source_root), 0)
        self._apply(prepared)
        first_record = self.cache.get(source_root, "Class 1/first.png")
        second_record = self.cache.get(source_root, "Class 1/second.png")
        self.assertEqual((first_record.image_width, first_record.image_height), (12, 8))
        self.assertEqual(
            (second_record.image_width, second_record.image_height), (16, 9)
        )

    def test_unchanged_preparation_reuses_all_cached_words_without_ocr(self):
        source_root = self.temp_dir / "source"
        first = self._image(source_root, "Class 1/first.png")
        second = self._image(source_root, "Class 1/second.png")
        batches = self._batches(("Class 1", (first, second)))
        initial_calls = []
        initial = prepare_ocr(
            source_root,
            batches,
            self.cache,
            ocr_function=self._counting_ocr(initial_calls),
        )
        self._apply(initial)
        reuse_calls = []
        progress = []

        prepared = prepare_ocr(
            source_root,
            batches,
            self.cache,
            ocr_function=self._counting_ocr(reuse_calls),
            progress=lambda *event: progress.append(event),
        )

        self.assertEqual(reuse_calls, [])
        self.assertEqual(
            prepared.stats,
            IncrementalStats(reused=2, new=0, changed=0, stale=0, removed=0),
        )
        self.assertEqual(
            progress,
            [(1, 2, first, "reused"), (2, 2, second, "reused")],
        )

    def test_size_or_mtime_change_reprocesses_exactly_changed_image(self):
        source_root = self.temp_dir / "source"
        first = self._image(source_root, "Class 1/first.png")
        second = self._image(source_root, "Class 1/second.png")
        batches = self._batches(("Class 1", (first, second)))
        initial = prepare_ocr(
            source_root,
            batches,
            self.cache,
            ocr_function=self._counting_ocr([]),
        )
        self._apply(initial)

        previous_mtime = first.stat().st_mtime_ns
        os.utime(first, ns=(previous_mtime + 1_000_000, previous_mtime + 1_000_000))
        calls = []
        progress = []
        prepared = prepare_ocr(
            source_root,
            batches,
            self.cache,
            ocr_function=self._counting_ocr(calls),
            progress=lambda *event: progress.append(event),
        )

        self.assertEqual(calls, [first])
        self.assertEqual(
            prepared.stats,
            IncrementalStats(reused=1, new=0, changed=1, stale=0, removed=0),
        )
        self.assertEqual(
            progress,
            [(1, 2, first, "changed"), (2, 2, second, "reused")],
        )

    def test_size_change_with_same_mtime_reprocesses_exactly_changed_image(self):
        source_root = self.temp_dir / "source"
        first = self._image(source_root, "Class 1/first.png")
        second = self._image(source_root, "Class 1/second.png")
        batches = self._batches(("Class 1", (first, second)))
        initial = prepare_ocr(
            source_root,
            batches,
            self.cache,
            ocr_function=self._counting_ocr([]),
        )
        self._apply(initial)

        original_mtime = first.stat().st_mtime_ns
        original_size = first.stat().st_size
        Image.new("RGB", (120, 80), "black").save(first)
        os.utime(first, ns=(original_mtime, original_mtime))
        self.assertNotEqual(first.stat().st_size, original_size)
        calls = []

        prepared = prepare_ocr(
            source_root,
            batches,
            self.cache,
            ocr_function=self._counting_ocr(calls),
        )

        self.assertEqual(calls, [first])
        self.assertEqual(
            prepared.stats,
            IncrementalStats(reused=1, new=0, changed=1, stale=0, removed=0),
        )

    def test_stale_schema_takes_precedence_over_changed_fingerprint(self):
        source_root = self.temp_dir / "source"
        image_path = self._image(source_root, "Class 1/one.png")
        fingerprint = fingerprint_file(source_root, image_path)
        stale_record = CacheRecord(
            source_root=source_key(source_root),
            fingerprint=fingerprint,
            image_width=12,
            image_height=8,
            record_schema_version=OCR_RECORD_SCHEMA_VERSION - 1,
            ocr_configuration_version=OCR_CONFIGURATION_VERSION,
            words=(),
        )
        self.cache.upsert(stale_record)
        prior_mtime = image_path.stat().st_mtime_ns
        os.utime(
            image_path,
            ns=(prior_mtime + 1_000_000, prior_mtime + 1_000_000),
        )
        calls = []
        progress = []

        prepared = prepare_ocr(
            source_root,
            self._batches(("Class 1", (image_path,))),
            self.cache,
            ocr_function=self._counting_ocr(calls),
            progress=lambda *event: progress.append(event),
        )

        self.assertEqual(calls, [image_path])
        self.assertEqual(
            prepared.stats,
            IncrementalStats(reused=0, new=0, changed=0, stale=1, removed=0),
        )
        self.assertEqual(progress, [(1, 1, image_path, "stale")])

    def test_stale_ocr_configuration_reprocesses_exactly_one_image(self):
        source_root = self.temp_dir / "source"
        image_path = self._image(source_root, "Class 1/one.png")
        stale_record = CacheRecord(
            source_root=source_key(source_root),
            fingerprint=fingerprint_file(source_root, image_path),
            image_width=12,
            image_height=8,
            record_schema_version=OCR_RECORD_SCHEMA_VERSION,
            ocr_configuration_version="old-ocr-configuration",
            words=(),
        )
        self.cache.upsert(stale_record)
        calls = []

        prepared = prepare_ocr(
            source_root,
            self._batches(("Class 1", (image_path,))),
            self.cache,
            ocr_function=self._counting_ocr(calls),
        )

        self.assertEqual(calls, [image_path])
        self.assertEqual(prepared.stats.stale, 1)
        self._apply(prepared)
        refreshed = self.cache.get(source_root, "Class 1/one.png")
        self.assertEqual(
            refreshed.ocr_configuration_version,
            OCR_CONFIGURATION_VERSION,
        )

    def test_deleted_image_is_excluded_counted_and_pruned(self):
        source_root = self.temp_dir / "source"
        retained = self._image(source_root, "Class 1/retained.png")
        deleted = self._image(source_root, "Class 1/deleted.png")
        initial = prepare_ocr(
            source_root,
            self._batches(("Class 1", (retained, deleted))),
            self.cache,
            ocr_function=self._counting_ocr([]),
        )
        self._apply(initial)
        deleted.unlink()

        prepared = prepare_ocr(
            source_root,
            self._batches(("Class 1", (retained,))),
            self.cache,
            ocr_function=self._counting_ocr([]),
        )

        self._apply(prepared)
        self.assertEqual(list(prepared.words_by_path), [retained])
        self.assertNotIn(deleted, prepared.words_by_path)
        self.assertEqual(prepared.stats.removed, 1)
        self.assertIsNone(self.cache.get(source_root, "Class 1/deleted.png"))

    def test_pruning_selected_source_preserves_another_source(self):
        selected_source = self.temp_dir / "selected"
        other_source = self.temp_dir / "other"
        selected = self._image(selected_source, "Class 1/selected.png")
        other = self._image(other_source, "Class 1/other.png")
        selected_prepared = prepare_ocr(
            selected_source,
            self._batches(("Class 1", (selected,))),
            self.cache,
            ocr_function=self._counting_ocr([]),
        )
        self._apply(selected_prepared)
        other_prepared = prepare_ocr(
            other_source,
            self._batches(("Class 1", (other,))),
            self.cache,
            ocr_function=self._counting_ocr([]),
        )
        self._apply(other_prepared)

        prepared = prepare_ocr(selected_source, [], self.cache, ocr_function=lambda _: [])
        self._apply(prepared)

        self.assertEqual(prepared.stats.removed, 1)
        self.assertEqual(self.cache.count_for_source(selected_source), 0)
        self.assertEqual(self.cache.count_for_source(other_source), 1)
        self.assertIsNotNone(self.cache.get(other_source, "Class 1/other.png"))

    def test_ocr_error_propagates_without_storing_failed_record(self):
        source_root = self.temp_dir / "source"
        image_path = self._image(source_root, "Class 1/failure.png")

        def failing_ocr(path: Path) -> list[OCRWord]:
            raise OCRProcessingError(f"failed: {path}")

        with self.assertRaisesRegex(OCRProcessingError, "failed"):
            prepare_ocr(
                source_root,
                self._batches(("Class 1", (image_path,))),
                self.cache,
                ocr_function=failing_ocr,
            )

        self.assertIsNone(self.cache.get(source_root, "Class 1/failure.png"))

    def test_empty_successful_ocr_result_is_cached_and_reused(self):
        source_root = self.temp_dir / "source"
        image_path = self._image(source_root, "Class 1/empty.png")
        first_calls = []

        first = prepare_ocr(
            source_root,
            self._batches(("Class 1", (image_path,))),
            self.cache,
            ocr_function=lambda path: first_calls.append(path) or [],
        )
        self._apply(first)
        second_calls = []
        second = prepare_ocr(
            source_root,
            self._batches(("Class 1", (image_path,))),
            self.cache,
            ocr_function=lambda path: second_calls.append(path) or [],
        )

        self.assertEqual(first_calls, [image_path])
        self.assertEqual(second_calls, [])
        self.assertEqual(first.words_by_path[image_path], ())
        self.assertEqual(second.words_by_path[image_path], ())
        self.assertEqual(second.stats.reused, 1)


if __name__ == "__main__":
    unittest.main()
