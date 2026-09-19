from pathlib import Path
import shutil
import sqlite3
import unittest

from ocr import NormalizedBox, OCRWord
from ocr_cache import (
    OCR_CONFIGURATION_VERSION,
    OCR_RECORD_SCHEMA_VERSION,
    CacheRecord,
    OCRCache,
    OCRCacheError,
    fingerprint_file,
    source_key,
)


class OCRCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(__file__).parent / ".tmp" / self._testMethodName
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir.mkdir(parents=True)
        self.cache_path = self.temp_dir / "cache.sqlite3"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _image(self, source_root: Path, relative_path: str, contents: bytes = b"image"):
        image_path = source_root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(contents)
        return image_path

    def _record(
        self,
        source_root: Path,
        relative_path: str,
        *,
        words: tuple[OCRWord, ...] = (),
        configuration_version: str = OCR_CONFIGURATION_VERSION,
    ):
        image_path = self._image(source_root, relative_path)
        return CacheRecord(
            source_root=source_key(source_root),
            fingerprint=fingerprint_file(source_root, image_path),
            image_width=1200,
            image_height=900,
            record_schema_version=OCR_RECORD_SCHEMA_VERSION,
            ocr_configuration_version=configuration_version,
            words=words,
        )

    def test_fingerprint_uses_source_relative_posix_path_and_file_metadata(self):
        source_root = self.temp_dir / "source"
        image_path = self._image(source_root, "Class 1/one.png", b"image bytes")

        fingerprint = fingerprint_file(source_root, image_path)

        self.assertEqual(fingerprint.relative_path, "Class 1/one.png")
        self.assertEqual(fingerprint.size, image_path.stat().st_size)
        self.assertEqual(fingerprint.mtime_ns, image_path.stat().st_mtime_ns)

    def test_upsert_and_get_round_trip_every_record_field(self):
        source_root = self.temp_dir / "source"
        words = (
            OCRWord("Hello", NormalizedBox(0.1, 0.2, 0.3, 0.4), 0.93),
            OCRWord("world", NormalizedBox(0.5, 0.6, 0.2, 0.1), 0.88),
        )
        record = self._record(source_root, "Class 1/one.png", words=words)

        cache = OCRCache(self.cache_path)
        try:
            cache.upsert(record)
            actual = cache.get(source_root, record.fingerprint.relative_path)
        finally:
            cache.close()

        self.assertIsNotNone(actual)
        self.assertEqual(actual.source_root, source_key(source_root))
        self.assertEqual(actual.fingerprint, record.fingerprint)
        self.assertEqual(actual.image_width, 1200)
        self.assertEqual(actual.image_height, 900)
        self.assertEqual(actual.record_schema_version, OCR_RECORD_SCHEMA_VERSION)
        self.assertEqual(actual.ocr_configuration_version, OCR_CONFIGURATION_VERSION)
        self.assertEqual(actual.words, words)
        self.assertIsInstance(actual.words, tuple)

    def test_transaction_rolls_back_an_upsert_when_body_raises(self):
        source_root = self.temp_dir / "source"
        record = self._record(source_root, "Class 1/one.png")
        cache = OCRCache(self.cache_path)
        try:
            with self.assertRaisesRegex(RuntimeError, "abort transaction"):
                with cache.transaction():
                    cache.upsert(record)
                    raise RuntimeError("abort transaction")
            actual = cache.get(source_root, record.fingerprint.relative_path)
        finally:
            cache.close()

        self.assertIsNone(actual)

    def test_prune_removes_missing_paths_only_for_selected_source(self):
        first_source = self.temp_dir / "first-source"
        second_source = self.temp_dir / "second-source"
        retained = self._record(first_source, "Class 1/retain.png")
        removed = self._record(first_source, "Class 1/remove.png")
        other_source = self._record(second_source, "Class 1/other.png")
        cache = OCRCache(self.cache_path)
        try:
            cache.upsert(retained)
            cache.upsert(removed)
            cache.upsert(other_source)

            cache.prune(first_source, {retained.fingerprint.relative_path})

            self.assertIsNotNone(
                cache.get(first_source, retained.fingerprint.relative_path)
            )
            self.assertIsNone(cache.get(first_source, removed.fingerprint.relative_path))
            self.assertIsNotNone(
                cache.get(second_source, other_source.fingerprint.relative_path)
            )
            self.assertEqual(cache.count_for_source(first_source), 1)
            self.assertEqual(cache.count_for_source(second_source), 1)
        finally:
            cache.close()

    def test_corrupt_database_raises_cache_error_with_cache_path(self):
        self.cache_path.write_text("not a SQLite database")

        with self.assertRaisesRegex(OCRCacheError, str(self.cache_path)):
            OCRCache(self.cache_path)

    def test_new_database_records_current_schema_version(self):
        cache = OCRCache(self.cache_path)
        cache.close()

        connection = sqlite3.connect(self.cache_path)
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
        finally:
            connection.close()

        self.assertEqual(version, 1)

    def test_unsupported_nonzero_database_version_is_rejected(self):
        connection = sqlite3.connect(self.cache_path)
        try:
            connection.execute("PRAGMA user_version = 2")
        finally:
            connection.close()

        with self.assertRaisesRegex(OCRCacheError, "unsupported database schema version 2"):
            OCRCache(self.cache_path)

    def test_old_ocr_configuration_record_remains_readable(self):
        source_root = self.temp_dir / "source"
        old_configuration = "apple-vision-en-us-word-boxes-v0"
        record = self._record(
            source_root,
            "Class 1/one.png",
            configuration_version=old_configuration,
        )
        cache = OCRCache(self.cache_path)
        try:
            cache.upsert(record)
            actual = cache.get(source_root, record.fingerprint.relative_path)
        finally:
            cache.close()

        self.assertIsNotNone(actual)
        self.assertEqual(actual.ocr_configuration_version, old_configuration)

    def test_aliases_for_same_directory_share_persistent_source_key(self):
        source_root = self.temp_dir / "source"
        source_root.mkdir()
        alias = self.temp_dir / "source-alias"
        alias.symlink_to(source_root, target_is_directory=True)
        stat = source_root.stat()

        self.assertEqual(source_key(source_root), source_key(alias))
        self.assertEqual(source_key(source_root), f"{stat.st_dev}:{stat.st_ino}")

    def test_old_record_schema_does_not_decode_incompatible_words_payload(self):
        source_root = self.temp_dir / "source"
        record = self._record(source_root, "Class 1/one.png")
        with OCRCache(self.cache_path) as cache:
            cache.upsert(record)
            cache.connection.execute(
                """
                UPDATE ocr_results
                SET record_schema_version = ?, words_json = ?
                WHERE source_root = ? AND relative_path = ?
                """,
                (
                    OCR_RECORD_SCHEMA_VERSION - 1,
                    "not valid JSON for this old schema",
                    source_key(source_root),
                    record.fingerprint.relative_path,
                ),
            )

            actual = cache.get(source_root, record.fingerprint.relative_path)

        self.assertIsNotNone(actual)
        self.assertEqual(actual.record_schema_version, OCR_RECORD_SCHEMA_VERSION - 1)
        self.assertEqual(actual.words, ())

    def test_current_record_schema_malformed_payload_names_cache_and_image(self):
        source_root = self.temp_dir / "source"
        record = self._record(source_root, "Class 1/broken.png")
        with OCRCache(self.cache_path) as cache:
            cache.upsert(record)
            cache.connection.execute(
                """
                UPDATE ocr_results
                SET words_json = ?
                WHERE source_root = ? AND relative_path = ?
                """,
                (
                    "not valid JSON",
                    source_key(source_root),
                    record.fingerprint.relative_path,
                ),
            )

            with self.assertRaises(OCRCacheError) as raised:
                cache.get(source_root, record.fingerprint.relative_path)

        self.assertIn(str(self.cache_path), str(raised.exception))
        self.assertIn(record.fingerprint.relative_path, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
