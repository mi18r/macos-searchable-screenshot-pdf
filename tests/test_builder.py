from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw, UnidentifiedImageError
from pypdf import PdfReader

import main as builder
from discovery import ClassBatch
from main import build_pdf
from ocr import NormalizedBox, OCRProcessingError, OCRWord


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / ".tmp" / self._testMethodName
        self.source = self.root / "Screenshots"
        self.source.mkdir(parents=True, exist_ok=False)
        self.output = self.root / "Result.pdf"
        self.working = self.root / "working.pdf"

    def tearDown(self):
        shutil.rmtree(self.root)

    def image(self, relative_path: str, label: str) -> Path:
        path = self.source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (400, 200), "white")
        ImageDraw.Draw(image).text((40, 80), label, fill="black")
        image.save(path)
        return path

    @staticmethod
    def stub_ocr(path: Path):
        return [OCRWord(path.stem, NormalizedBox(0.1, 0.4, 0.5, 0.1), 1.0)]

    def test_builds_ordered_pages_with_class_bookmarks_and_text(self):
        self.image("Class 2/Screenshot 2024-12-28 at 7.00.45 PM.png", "class two")
        self.image("Trial/Trial Class.png", "trial")
        self.image("Class 1/Screenshot 2024-12-22 at 9.00.05 PM.png", "later")
        self.image("Class 1/Screenshot 2024-12-22 at 8.22.21 PM.png", "earlier")

        summary = build_pdf(self.source, self.output, self.working, self.stub_ocr)
        reader = PdfReader(self.output)

        self.assertEqual(summary.page_count, 4)
        self.assertEqual(summary.searchable_pages, 4)
        self.assertEqual(len(reader.pages), 4)
        self.assertEqual(
            [item.title for item in reader.outline], ["Trial", "Class 1", "Class 2"]
        )
        self.assertEqual(
            [reader.get_destination_page_number(item) for item in reader.outline],
            [0, 1, 3],
        )
        page_text = [page.extract_text() for page in reader.pages]
        self.assertIn("Trial Class", page_text[0])
        self.assertIn("8.22.21", page_text[1])
        self.assertIn("9.00.05", page_text[2])
        self.assertIn("7.00.45", page_text[3])
        self.assertFalse(self.working.exists())
        self.assertFalse(self.output.with_suffix(".pdf.part").exists())

    def test_ocr_processing_error_aborts_and_preserves_existing_output(self):
        failed_image = self.image("Class 1/one.png", "visible")
        previous_output = b"previous successful output"
        self.output.write_bytes(previous_output)

        def failed_ocr(path):
            raise OCRProcessingError(f"Vision unavailable for {path}")

        with self.assertRaisesRegex(OCRProcessingError, str(failed_image)):
            build_pdf(self.source, self.output, self.working, failed_ocr)

        self.assertEqual(self.output.read_bytes(), previous_output)
        self.assertFalse(self.working.exists())
        self.assertFalse(self.output.with_suffix(".pdf.part").exists())

    def test_supplied_manifest_bypasses_rediscovery_and_preserves_its_order(self):
        second = self.image("Class 2/second.png", "second")
        first = self.image("Class 1/first.png", "first")
        supplied = [
            ClassBatch("Class 2", (second,)),
            ClassBatch("Class 1", (first,)),
        ]
        calls = []

        def ordered_ocr(path):
            calls.append(path)
            return self.stub_ocr(path)

        with patch(
            "main.discover_classes",
            side_effect=AssertionError("supplied manifest was rediscovered"),
        ):
            build_pdf(
                self.source,
                self.output,
                self.working,
                ordered_ocr,
                batches=supplied,
            )

        reader = PdfReader(self.output)
        self.assertEqual(calls, [second, first])
        self.assertEqual(
            [item.title for item in reader.outline], ["Class 2", "Class 1"]
        )
        self.assertEqual(
            [page.extract_text().strip() for page in reader.pages],
            ["second", "first"],
        )

    def test_image_progress_interleaves_with_rendering_in_manifest_order(self):
        second = self.image("Class 2/second.png", "second")
        first = self.image("Class 1/first.png", "first")
        supplied = [
            ClassBatch("Class 2", (second,)),
            ClassBatch("Class 1", (first,)),
        ]
        events = []
        real_image_reader = builder.ImageReader

        def recording_image_reader(image):
            events.append(("render", Path(image.filename)))
            return real_image_reader(image)

        def recording_progress(_current, _total, image_path, phase):
            if phase == "image":
                events.append(("progress", image_path))

        with patch.object(
            builder, "ImageReader", side_effect=recording_image_reader
        ):
            build_pdf(
                self.source,
                self.output,
                self.working,
                self.stub_ocr,
                batches=supplied,
                progress=recording_progress,
            )

        self.assertEqual(
            events,
            [
                ("progress", second),
                ("render", second),
                ("progress", first),
                ("render", first),
            ],
        )

    def test_empty_class_does_not_receive_bookmark(self):
        self.image("Class 1/one.png", "visible")
        (self.source / "Class 2").mkdir()

        build_pdf(self.source, self.output, self.working, self.stub_ocr)
        reader = PdfReader(self.output)

        self.assertEqual([item.title for item in reader.outline], ["Class 1"])

    def test_empty_ocr_keeps_visible_page_and_reports_it(self):
        image_path = self.image("Class 1/one.png", "visible")
        output = StringIO()

        with redirect_stdout(output):
            summary = build_pdf(
                self.source, self.output, self.working, lambda _path: []
            )
        reader = PdfReader(self.output)

        self.assertEqual(summary.page_count, 1)
        self.assertEqual(summary.searchable_pages, 0)
        self.assertEqual(summary.ocr_failures, (image_path,))
        self.assertEqual(len(reader.pages), 1)
        self.assertEqual(len(reader.pages[0].images), 1)
        self.assertEqual(
            output.getvalue(),
            f"WARNING: OCR found no English text in {image_path}\n",
        )

    def test_invalid_ocr_words_keep_visible_page_and_report_it(self):
        image_path = self.image("Class 1/one.png", "visible")
        invalid_words = [
            OCRWord(" ", NormalizedBox(0.1, 0.4, 0.5, 0.1), 1.0),
            OCRWord("NoWidth", NormalizedBox(0.1, 0.4, 0.0, 0.1), 1.0),
            OCRWord("NoHeight", NormalizedBox(0.1, 0.4, 0.5, -0.1), 1.0),
        ]
        output = StringIO()

        with redirect_stdout(output):
            summary = build_pdf(
                self.source,
                self.output,
                self.working,
                lambda _path: invalid_words,
            )
        reader = PdfReader(self.output)

        self.assertEqual(summary.page_count, 1)
        self.assertEqual(summary.searchable_pages, 0)
        self.assertEqual(summary.ocr_failures, (image_path,))
        self.assertEqual(len(reader.pages), 1)
        self.assertEqual(len(reader.pages[0].images), 1)
        self.assertEqual(reader.pages[0].extract_text(), "")
        self.assertEqual(
            output.getvalue(),
            f"WARNING: OCR found no English text in {image_path}\n",
        )

    def test_preexisting_transient_entries_survive_success_without_target_changes(self):
        self.image("Class 1/one.png", "visible")
        previous_output = b"previous successful output"
        self.output.write_bytes(previous_output)
        sentinel = self.root / "sentinel.bin"
        sentinel.write_bytes(b"foreign sentinel")
        partial = self.output.with_suffix(".pdf.part")
        os.link(sentinel, partial)
        self.working.symlink_to(self.output)

        build_pdf(self.source, self.output, self.working, self.stub_ocr)

        self.assertTrue(self.working.is_symlink())
        self.assertEqual(self.working.readlink(), self.output)
        self.assertTrue(partial.exists())
        self.assertEqual(partial.stat().st_ino, sentinel.stat().st_ino)
        self.assertEqual(sentinel.read_bytes(), b"foreign sentinel")
        self.assertEqual(len(PdfReader(self.output).pages), 1)

    def test_preexisting_transient_entries_survive_failure(self):
        corrupt_image = self.source / "Class 1" / "corrupt.png"
        corrupt_image.parent.mkdir(parents=True)
        corrupt_image.write_bytes(b"not a real PNG")
        previous_output = b"previous successful output"
        self.output.write_bytes(previous_output)
        sentinel = self.root / "sentinel.bin"
        sentinel.write_bytes(b"foreign sentinel")
        partial = self.output.with_suffix(".pdf.part")
        os.link(sentinel, partial)
        self.working.symlink_to(self.output)

        with self.assertRaisesRegex(RuntimeError, str(corrupt_image)):
            build_pdf(self.source, self.output, self.working, self.stub_ocr)

        self.assertEqual(self.output.read_bytes(), previous_output)
        self.assertTrue(self.working.is_symlink())
        self.assertEqual(self.working.readlink(), self.output)
        self.assertTrue(partial.exists())
        self.assertEqual(partial.stat().st_ino, sentinel.stat().st_ino)
        self.assertEqual(sentinel.read_bytes(), b"foreign sentinel")

    def test_failed_build_preserves_existing_output(self):
        self.output.write_bytes(b"previous successful output")
        with self.assertRaisesRegex(ValueError, "No supported screenshots"):
            build_pdf(self.source, self.output, self.working, self.stub_ocr)
        self.assertEqual(self.output.read_bytes(), b"previous successful output")

    def test_render_failure_preserves_output_and_removes_working_artifacts(self):
        self.image("Class 1/one.png", "visible")
        previous_output = b"previous successful output"
        self.output.write_bytes(previous_output)

        def failed_render(_page_entries, working_pdf, _progress=None):
            working_pdf.write(b"partial working state")
            raise RuntimeError("render failed")

        with patch("main.create_image_pdf", side_effect=failed_render):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                build_pdf(self.source, self.output, self.working, self.stub_ocr)

        self.assertEqual(self.output.read_bytes(), previous_output)
        self.assertFalse(self.working.exists())
        self.assertFalse(self.output.with_suffix(".pdf.part").exists())

    def test_corrupt_image_failure_names_image_and_preserves_output(self):
        corrupt_image = self.source / "Class 1" / "corrupt.png"
        corrupt_image.parent.mkdir(parents=True)
        corrupt_image.write_bytes(b"not a real PNG")
        previous_output = b"previous successful output"
        self.output.write_bytes(previous_output)

        with self.assertRaises(RuntimeError) as raised:
            build_pdf(self.source, self.output, self.working, self.stub_ocr)

        self.assertIn(str(corrupt_image), str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, UnidentifiedImageError)
        self.assertEqual(self.output.read_bytes(), previous_output)
        self.assertFalse(self.working.exists())
        self.assertFalse(self.output.with_suffix(".pdf.part").exists())

    def test_writer_failure_preserves_output_and_removes_partial_artifacts(self):
        self.image("Class 1/one.png", "visible")
        previous_output = b"previous successful output"
        self.output.write_bytes(previous_output)

        def failed_write(_writer, stream):
            stream.write(b"partial PDF output")
            raise OSError("write failed")

        with patch("main.PdfWriter.write", autospec=True, side_effect=failed_write):
            with self.assertRaisesRegex(RuntimeError, str(self.output)) as raised:
                build_pdf(self.source, self.output, self.working, self.stub_ocr)

        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertEqual(self.output.read_bytes(), previous_output)
        self.assertFalse(self.working.exists())
        self.assertFalse(self.output.with_suffix(".pdf.part").exists())


if __name__ == "__main__":
    unittest.main()
