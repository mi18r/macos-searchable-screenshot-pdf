from contextlib import contextmanager, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
from pypdf import PdfReader

import main as builder
from incremental import IncrementalStats
from main import BuildSummary, RunSummary
from ocr import NormalizedBox, OCRProcessingError, OCRWord
from ocr_cache import (
    CacheRecord,
    OCRCache,
    OCRCacheError,
    OCR_CONFIGURATION_VERSION,
    OCR_RECORD_SCHEMA_VERSION,
    fingerprint_file,
    source_key,
)
from run_lock import BuildAlreadyRunningError, build_lock, source_identity


class IncrementalBuildTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / ".tmp" / self._testMethodName
        shutil.rmtree(self.root, ignore_errors=True)
        self.source = self.root / "Selected Screenshots"
        self.source.mkdir(parents=True)
        self.cache_directory = self.root / "builder-cache"
        self.cache_file = self.cache_directory / "ocr_cache.sqlite3"
        self.output = self.source / "Malayalam.pdf"
        self.staged = self.source / ".Malayalam.pdf.ready"

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def image(self, relative_path: str, label: str, color: str = "white") -> Path:
        path = self.source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (400, 200), color)
        ImageDraw.Draw(image).text((40, 80), label, fill="black")
        image.save(path)
        return path

    @staticmethod
    def counting_ocr(calls):
        def stub(path: Path):
            calls.append(path)
            return [
                OCRWord(
                    path.stem,
                    NormalizedBox(0.1, 0.4, 0.5, 0.1),
                    1.0,
                )
            ]

        return stub

    def run_build(self, calls):
        with patch.object(builder, "CACHE_DIRECTORY", self.cache_directory):
            return builder.run_selected_build(
                self.source,
                cache_file=self.cache_file,
                ocr_function=self.counting_ocr(calls),
            )

    def seed_collection(self):
        trial = self.image("Trial/Trial Class.png", "trial")
        class_one = self.image(
            "Class 1/Screenshot 2024-12-22 at 9.00.05 PM.png", "later"
        )
        class_two = self.image(
            "Class 2/Screenshot 2024-12-28 at 7.00.45 PM.png", "class two"
        )
        return trial, class_one, class_two

    def assert_no_transient_artifacts(self):
        self.assertFalse(self.staged.exists())
        self.assertFalse(self.source.joinpath(".Malayalam.pdf.ready.part").exists())
        self.assertEqual(list(self.source.glob(".Malayalam.pdf.*.ready")), [])
        work_directory = self.cache_directory / "work"
        if work_directory.exists():
            self.assertEqual(list(work_directory.iterdir()), [])

    def test_sequential_builds_reuse_add_change_delete_and_prune(self):
        trial, class_one, class_two = self.seed_collection()

        first_calls = []
        first = self.run_build(first_calls)
        first_reader = PdfReader(self.output)
        self.assertEqual(first_calls, [trial, class_one, class_two])
        self.assertEqual(first.build.page_count, 3)
        self.assertEqual(first.build.searchable_pages, 3)
        self.assertEqual(
            [item.title for item in first_reader.outline],
            ["Trial", "Class 1", "Class 2"],
        )
        self.assertEqual(
            [first_reader.get_destination_page_number(item) for item in first_reader.outline],
            [0, 1, 2],
        )

        unchanged_calls = []
        unchanged = self.run_build(unchanged_calls)
        self.assertEqual(unchanged_calls, [])
        self.assertEqual(unchanged.build.searchable_pages, 3)
        self.assertEqual(unchanged.incremental.reused, 3)

        earlier = self.image(
            "Class 1/Screenshot 2024-12-22 at 8.22.21 PM.png", "earlier"
        )
        added_calls = []
        added = self.run_build(added_calls)
        added_reader = PdfReader(self.output)
        self.assertEqual(added_calls, [earlier])
        self.assertEqual(added.incremental.new, 1)
        self.assertEqual(
            [page.extract_text().strip() for page in added_reader.pages],
            [trial.stem, earlier.stem, class_one.stem, class_two.stem],
        )

        old_mtime = class_one.stat().st_mtime_ns
        self.image(
            "Class 1/Screenshot 2024-12-22 at 9.00.05 PM.png",
            "changed",
            color="yellow",
        )
        os.utime(class_one, ns=(old_mtime + 1_000_000, old_mtime + 1_000_000))
        changed_calls = []
        changed = self.run_build(changed_calls)
        self.assertEqual(changed_calls, [class_one])
        self.assertEqual(changed.incremental.changed, 1)

        class_two.unlink()
        deleted_calls = []
        deleted = self.run_build(deleted_calls)
        deleted_reader = PdfReader(self.output)
        self.assertEqual(deleted_calls, [])
        self.assertEqual(deleted.incremental.removed, 1)
        self.assertEqual(len(deleted_reader.pages), 3)
        self.assertEqual(
            [item.title for item in deleted_reader.outline], ["Trial", "Class 1"]
        )
        with OCRCache(self.cache_file) as cache:
            self.assertIsNone(cache.get(self.source, "Class 2/" + class_two.name))
            self.assertEqual(cache.count_for_source(self.source), 3)
        self.assert_no_transient_artifacts()

    def test_one_discovered_manifest_is_stable_during_a_run(self):
        first = self.image("Class 1/first.png", "first")
        added_during_ocr = self.source / "Class 1" / "second.png"
        calls = []

        def ocr_that_changes_source(path):
            calls.append(path)
            self.image("Class 1/second.png", "second")
            return [OCRWord(path.stem, NormalizedBox(0.1, 0.4, 0.5, 0.1), 1.0)]

        with patch.object(builder, "CACHE_DIRECTORY", self.cache_directory):
            summary = builder.run_selected_build(
                self.source,
                cache_file=self.cache_file,
                ocr_function=ocr_that_changes_source,
            )

        self.assertEqual(calls, [first])
        self.assertTrue(added_during_ocr.exists())
        self.assertEqual(summary.screenshot_count, 1)
        self.assertEqual(len(PdfReader(self.output).pages), 1)

    def test_ocr_error_preserves_master_and_rolls_back_cache(self):
        image_path = self.image("Class 1/one.png", "visible")
        previous = b"previous successful output"
        self.output.write_bytes(previous)

        def failed_ocr(path):
            raise OCRProcessingError(f"Vision failed for {path}")

        with patch.object(builder, "CACHE_DIRECTORY", self.cache_directory):
            with self.assertRaisesRegex(OCRProcessingError, str(image_path)):
                builder.run_selected_build(
                    self.source,
                    cache_file=self.cache_file,
                    ocr_function=failed_ocr,
                )

        self.assertEqual(self.output.read_bytes(), previous)
        with OCRCache(self.cache_file) as cache:
            self.assertEqual(cache.count_for_source(self.source), 0)
        self.assert_no_transient_artifacts()

    def test_cache_commit_error_preserves_master_and_removes_staged_files(self):
        self.image("Class 1/one.png", "visible")
        previous = b"previous successful output"
        self.output.write_bytes(previous)
        cache_file = self.cache_file

        class CommitFailingCache(OCRCache):
            @contextmanager
            def transaction(inner_self):
                inner_self.connection.execute("BEGIN IMMEDIATE")
                try:
                    yield inner_self
                except BaseException:
                    inner_self.connection.rollback()
                    raise
                else:
                    inner_self.connection.rollback()
                    raise OCRCacheError(
                        f"OCR cache error at {cache_file}: simulated commit failure"
                    )

        with (
            patch.object(builder, "CACHE_DIRECTORY", self.cache_directory),
            patch.object(builder, "OCRCache", CommitFailingCache),
        ):
            with self.assertRaisesRegex(OCRCacheError, str(self.cache_file)):
                builder.run_selected_build(
                    self.source,
                    cache_file=self.cache_file,
                    ocr_function=self.counting_ocr([]),
                )

        self.assertEqual(self.output.read_bytes(), previous)
        with OCRCache(self.cache_file) as cache:
            self.assertEqual(cache.count_for_source(self.source), 0)
        self.assert_no_transient_artifacts()

    def test_final_replace_error_preserves_master_and_removes_staged_files(self):
        self.image("Class 1/one.png", "visible")
        previous = b"previous successful output"
        self.output.write_bytes(previous)
        original_replace = Path.replace

        def fail_only_final_replace(path, target):
            if Path(target) == self.output:
                raise OSError("simulated final replace failure")
            return original_replace(path, target)

        with (
            patch.object(builder, "CACHE_DIRECTORY", self.cache_directory),
            patch.object(Path, "replace", autospec=True, side_effect=fail_only_final_replace),
        ):
            with self.assertRaisesRegex(RuntimeError, str(self.output)):
                builder.run_selected_build(
                    self.source,
                    cache_file=self.cache_file,
                    ocr_function=self.counting_ocr([]),
                )

        self.assertEqual(self.output.read_bytes(), previous)
        self.assert_no_transient_artifacts()

    def test_rejected_contender_does_not_delete_active_build_staging_files(self):
        self.image("Class 1/one.png", "visible")
        staged_contents = b"active owner staged PDF"
        partial_contents = b"active owner partial PDF"
        staged_partial = self.source / ".Malayalam.pdf.ready.part"
        self.staged.write_bytes(staged_contents)
        staged_partial.write_bytes(partial_contents)

        with patch.object(builder, "CACHE_DIRECTORY", self.cache_directory):
            with build_lock(self.cache_directory, self.source):
                with self.assertRaises(BuildAlreadyRunningError):
                    builder.run_selected_build(
                        self.source,
                        cache_file=self.cache_file,
                        ocr_function=self.counting_ocr([]),
                    )

                self.assertEqual(self.staged.read_bytes(), staged_contents)
                self.assertEqual(staged_partial.read_bytes(), partial_contents)

    def test_acquired_build_does_not_delete_preexisting_legacy_transients_on_failure(self):
        image_path = self.image("Class 1/one.png", "visible")
        self.output.write_bytes(b"previous master")
        staged_partial = self.source / ".Malayalam.pdf.ready.part"
        self.staged.write_bytes(b"foreign staged")
        staged_partial.write_bytes(b"foreign partial")
        legacy_working = builder.working_path_for(self.source, self.cache_directory)
        legacy_working.write_bytes(b"foreign working")

        def failed_ocr(path):
            raise OCRProcessingError(f"Vision failed for {path}")

        with patch.object(builder, "CACHE_DIRECTORY", self.cache_directory):
            with self.assertRaisesRegex(OCRProcessingError, str(image_path)):
                builder.run_selected_build(
                    self.source,
                    cache_file=self.cache_file,
                    ocr_function=failed_ocr,
                )

        self.assertEqual(self.output.read_bytes(), b"previous master")
        self.assertEqual(self.staged.read_bytes(), b"foreign staged")
        self.assertEqual(staged_partial.read_bytes(), b"foreign partial")
        self.assertEqual(legacy_working.read_bytes(), b"foreign working")

    def test_success_ignores_legacy_symlink_hardlink_and_regular_transients(self):
        screenshot = self.image("Class 1/one.png", "visible")
        original_screenshot = screenshot.read_bytes()
        sentinel = self.root / "sentinel.bin"
        sentinel.write_bytes(b"foreign sentinel")
        self.staged.symlink_to(sentinel)
        staged_partial = self.source / ".Malayalam.pdf.ready.part"
        os.link(screenshot, staged_partial)
        legacy_working = builder.working_path_for(self.source, self.cache_directory)
        legacy_working.write_bytes(b"foreign working")

        with patch.object(builder, "CACHE_DIRECTORY", self.cache_directory):
            summary = builder.run_selected_build(
                self.source,
                cache_file=self.cache_file,
                ocr_function=self.counting_ocr([]),
            )

        self.assertEqual(summary.output_file, self.output)
        self.assertEqual(len(PdfReader(self.output).pages), 1)
        self.assertTrue(self.staged.is_symlink())
        self.assertEqual(self.staged.readlink(), sentinel)
        self.assertEqual(sentinel.read_bytes(), b"foreign sentinel")
        self.assertEqual(staged_partial.stat().st_ino, screenshot.stat().st_ino)
        self.assertEqual(screenshot.read_bytes(), original_screenshot)
        self.assertEqual(legacy_working.read_bytes(), b"foreign working")

    def test_output_symlink_is_rejected_before_discovery_or_cache_work(self):
        self.image("Class 1/one.png", "visible")
        for target_exists in (True, False):
            with self.subTest(target_exists=target_exists):
                target = self.root / f"target-{target_exists}.pdf"
                if target_exists:
                    target.write_bytes(b"foreign target")
                self.output.symlink_to(target)

                with patch.object(builder, "CACHE_DIRECTORY", self.cache_directory):
                    with self.assertRaisesRegex(RuntimeError, "Malayalam.pdf") as raised:
                        builder.run_selected_build(
                            self.source,
                            cache_file=self.cache_file,
                            ocr_function=lambda _path: self.fail("OCR was called"),
                        )

                self.assertIn("symbolic link", str(raised.exception).lower())
                self.assertTrue(self.output.is_symlink())
                self.assertEqual(self.output.readlink(), target)
                if target_exists:
                    self.assertEqual(target.read_bytes(), b"foreign target")
                self.assertFalse(self.cache_file.exists())
                self.assertFalse((self.cache_directory / "work").exists())
                self.assertEqual(list(self.source.glob(".Malayalam.pdf.*.ready")), [])
                self.output.unlink()

    def test_symlinked_class_folder_is_rejected_before_discovery_and_cache(self):
        external_class = self.root / "External Class"
        external_class.mkdir()
        external_image = external_class / "one.png"
        Image.new("RGB", (40, 20), "white").save(external_image)
        external_contents = external_image.read_bytes()
        linked_class = self.source / "Class 1"
        linked_class.symlink_to(external_class, target_is_directory=True)
        self.output.write_bytes(b"previous master")

        with patch.object(builder, "CACHE_DIRECTORY", self.cache_directory):
            with self.assertRaisesRegex(RuntimeError, str(linked_class)) as raised:
                builder.run_selected_build(
                    self.source,
                    cache_file=self.cache_file,
                    ocr_function=lambda _path: self.fail("OCR was called"),
                )

        self.assertIn("symbolic link", str(raised.exception).lower())
        self.assertTrue(linked_class.is_symlink())
        self.assertEqual(external_image.read_bytes(), external_contents)
        self.assertEqual(self.output.read_bytes(), b"previous master")
        self.assertFalse(self.cache_file.exists())
        self.assertFalse((self.cache_directory / "work").exists())
        self.assertEqual(list(self.source.glob(".Malayalam.pdf.*.ready")), [])

    def test_symlinked_screenshot_is_rejected_before_discovery_and_cache(self):
        class_folder = self.source / "Class 1"
        class_folder.mkdir()
        external_image = self.root / "external.png"
        Image.new("RGB", (40, 20), "white").save(external_image)
        external_contents = external_image.read_bytes()
        linked_image = class_folder / "linked.png"
        linked_image.symlink_to(external_image)
        self.output.write_bytes(b"previous master")

        with patch.object(builder, "CACHE_DIRECTORY", self.cache_directory):
            with self.assertRaisesRegex(RuntimeError, str(linked_image)) as raised:
                builder.run_selected_build(
                    self.source,
                    cache_file=self.cache_file,
                    ocr_function=lambda _path: self.fail("OCR was called"),
                )

        self.assertIn("symbolic link", str(raised.exception).lower())
        self.assertTrue(linked_image.is_symlink())
        self.assertEqual(external_image.read_bytes(), external_contents)
        self.assertEqual(self.output.read_bytes(), b"previous master")
        self.assertFalse(self.cache_file.exists())
        self.assertFalse((self.cache_directory / "work").exists())
        self.assertEqual(list(self.source.glob(".Malayalam.pdf.*.ready")), [])

    def test_image_mutation_during_render_aborts_and_preserves_master_and_cache(self):
        image_path = self.image("Class 1/one.png", "visible")
        self.output.write_bytes(b"previous master")
        original_create_image_pdf = builder.create_image_pdf

        def render_then_mutate(*args, **kwargs):
            result = original_create_image_pdf(*args, **kwargs)
            previous_mtime = image_path.stat().st_mtime_ns
            Image.new("RGB", (800, 300), "yellow").save(image_path)
            os.utime(
                image_path,
                ns=(previous_mtime + 1_000_000, previous_mtime + 1_000_000),
            )
            return result

        with (
            patch.object(builder, "CACHE_DIRECTORY", self.cache_directory),
            patch.object(builder, "create_image_pdf", side_effect=render_then_mutate),
        ):
            with self.assertRaisesRegex(RuntimeError, str(image_path)) as raised:
                builder.run_selected_build(
                    self.source,
                    cache_file=self.cache_file,
                    ocr_function=self.counting_ocr([]),
                )

        self.assertIn("changed", str(raised.exception).lower())
        self.assertEqual(self.output.read_bytes(), b"previous master")
        with OCRCache(self.cache_file) as cache:
            self.assertEqual(cache.count_for_source(self.source), 0)
        self.assert_no_transient_artifacts()

    def test_manifest_is_validated_at_all_three_publish_boundaries(self):
        self.image("Class 1/one.png", "visible")
        original_validate = builder.validate_prepared_ocr

        with (
            patch.object(builder, "CACHE_DIRECTORY", self.cache_directory),
            patch.object(
                builder,
                "validate_prepared_ocr",
                wraps=original_validate,
            ) as validate,
        ):
            builder.run_selected_build(
                self.source,
                cache_file=self.cache_file,
                ocr_function=self.counting_ocr([]),
            )

        self.assertEqual(validate.call_count, 3)

    def test_writer_failure_preserves_master_and_stores_no_pending_cache_rows(self):
        self.image("Class 1/one.png", "visible")
        self.output.write_bytes(b"previous master")

        def failed_write(_writer, stream):
            stream.write(b"partial staged PDF")
            raise OSError("simulated writer failure")

        with (
            patch.object(builder, "CACHE_DIRECTORY", self.cache_directory),
            patch.object(
                builder.PdfWriter,
                "write",
                autospec=True,
                side_effect=failed_write,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, str(self.output)):
                builder.run_selected_build(
                    self.source,
                    cache_file=self.cache_file,
                    ocr_function=self.counting_ocr([]),
                )

        self.assertEqual(self.output.read_bytes(), b"previous master")
        with OCRCache(self.cache_file) as cache:
            self.assertEqual(cache.count_for_source(self.source), 0)
        self.assert_no_transient_artifacts()

    def test_long_ocr_preparation_does_not_hold_shared_cache_write_transaction(self):
        self.image("Class 1/a.png", "source a")
        source_b = self.root / "Source B"
        image_b_ocr = source_b / "Class 1" / "during-ocr.png"
        image_b_render = source_b / "Class 1" / "during-render.png"
        image_b_ocr.parent.mkdir(parents=True)
        Image.new("RGB", (50, 25), "white").save(image_b_ocr)
        Image.new("RGB", (50, 25), "white").save(image_b_render)

        def record_for(image_path):
            return CacheRecord(
                source_root=source_key(source_b),
                fingerprint=fingerprint_file(source_b, image_path),
                image_width=50,
                image_height=25,
                record_schema_version=OCR_RECORD_SCHEMA_VERSION,
                ocr_configuration_version=OCR_CONFIGURATION_VERSION,
                words=(),
            )

        def write_source_b(record):
            with OCRCache(self.cache_file) as cache_b:
                cache_b.connection.execute("PRAGMA busy_timeout = 1")
                with cache_b.transaction():
                    cache_b.upsert(record)

        def ocr_a(path):
            write_source_b(record_for(image_b_ocr))
            return self.counting_ocr([])(path)

        original_create_image_pdf = builder.create_image_pdf

        def render_a(*args, **kwargs):
            write_source_b(record_for(image_b_render))
            return original_create_image_pdf(*args, **kwargs)

        with (
            patch.object(builder, "CACHE_DIRECTORY", self.cache_directory),
            patch.object(builder, "create_image_pdf", side_effect=render_a),
        ):
            builder.run_selected_build(
                self.source,
                cache_file=self.cache_file,
                ocr_function=ocr_a,
            )

        with OCRCache(self.cache_file) as cache:
            self.assertEqual(cache.count_for_source(source_b), 2)
            self.assertEqual(cache.count_for_source(self.source), 1)

    def test_output_and_working_paths_stay_in_their_required_locations(self):
        self.image("Class 1/one.png", "visible")
        observed_temporaries = []
        real_new_build_temporaries = builder._new_build_temporaries

        def recording_new_build_temporaries(*args, **kwargs):
            temporaries = real_new_build_temporaries(*args, **kwargs)
            observed_temporaries.append(tuple(item.path for item in temporaries))
            return temporaries

        with (
            patch.object(builder, "CACHE_DIRECTORY", self.cache_directory),
            patch.object(
                builder,
                "_new_build_temporaries",
                side_effect=recording_new_build_temporaries,
            ),
        ):
            summary = builder.run_selected_build(
                self.source,
                cache_file=self.cache_file,
                ocr_function=self.counting_ocr([]),
            )

        self.assertEqual(summary.output_file, self.output)
        self.assertEqual(summary.output_file.parent, self.source.resolve())
        self.assertEqual(self.cache_file.parent, self.cache_directory)
        self.assertEqual(len(observed_temporaries), 1)
        staged, working = observed_temporaries[0]
        self.assertEqual(staged.parent, self.source)
        self.assertTrue(working.is_relative_to(self.cache_directory / "work"))
        self.assertNotEqual(staged, self.staged)
        self.assertNotEqual(working, builder.working_path_for(self.source, self.cache_directory))
        self.assert_no_transient_artifacts()

    def test_work_name_uses_physical_source_identity_for_path_aliases(self):
        alias = self.root / "Selected Screenshots Alias"
        alias.symlink_to(self.source, target_is_directory=True)
        identity_text = "-".join(str(value) for value in source_identity(self.source))

        real_path = builder.working_path_for(self.source, self.cache_directory)
        alias_path = builder.working_path_for(alias, self.cache_directory)

        self.assertEqual(real_path, alias_path)
        self.assertIn(identity_text, real_path.name)

    def test_progress_and_summary_use_plain_language_labels_and_counts(self):
        trial, class_one, class_two = self.seed_collection()
        output = StringIO()

        with redirect_stdout(output):
            summary = self.run_build([])
            builder.print_run_summary(summary)

        text = output.getvalue()
        for current, path in enumerate((trial, class_one, class_two), start=1):
            self.assertIn(f"Preparing OCR {current}/3 [new]: {path.name}", text)
            self.assertIn(f"Building image page {current}/3: {path.name}", text)
            self.assertIn(f"Adding search layer {current}/3: {path.name}", text)
        self.assertIn(f"Selected folder: {self.source.resolve()}", text)
        self.assertIn("Class folders: 3", text)
        self.assertIn("Screenshots found: 3", text)
        self.assertIn("Reused OCR: 0", text)
        self.assertIn("New screenshots: 3", text)
        self.assertIn("Changed screenshots: 0", text)
        self.assertIn("Stale screenshots: 0", text)
        self.assertIn("Removed screenshots: 0", text)
        self.assertIn("Searchable pages: 3/3", text)
        self.assertIn(f"Created: {self.output}", text)

    def test_summary_lists_each_non_searchable_page(self):
        missing_text = self.source / "Class 1" / "empty.png"
        summary = RunSummary(
            source_folder=self.source,
            output_file=self.output,
            class_count=1,
            screenshot_count=1,
            build=BuildSummary(1, 0, (missing_text,)),
            incremental=IncrementalStats(0, 1, 0, 0, 0),
        )
        output = StringIO()

        with redirect_stdout(output):
            builder.print_run_summary(summary)

        self.assertIn(f"Non-searchable page: {missing_text}", output.getvalue())

    def test_main_cancellation_and_error_return_expected_exit_codes(self):
        cancelled_output = StringIO()
        with (
            patch.object(builder, "select_source_folder", return_value=None),
            redirect_stdout(cancelled_output),
        ):
            cancelled = builder.main([])

        self.assertEqual(cancelled, 0)
        self.assertEqual(
            cancelled_output.getvalue(),
            "Folder selection cancelled. No files were changed.\n",
        )

        missing = self.root / "missing source"
        error_output = StringIO()
        with redirect_stdout(error_output):
            failed = builder.main(["--source", str(missing)])

        self.assertEqual(failed, 1)
        self.assertIn("ERROR:", error_output.getvalue())
        self.assertIn(str(missing.resolve()), error_output.getvalue())


if __name__ == "__main__":
    unittest.main()
