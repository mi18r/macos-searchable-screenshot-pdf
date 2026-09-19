import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from pypdf import PdfReader

import main as builder


@unittest.skipUnless(
    os.environ.get("RUN_VISION_INTEGRATION") == "1",
    "set RUN_VISION_INTEGRATION=1 to run Apple Vision integration",
)
class SampleIntegrationTests(unittest.TestCase):
    def setUp(self):
        project = Path(__file__).parent.parent
        self.root = project / "tests" / ".tmp" / self._testMethodName
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

        self.source = self.root / "Sample Screenshots"
        shutil.copytree(project.parent / "Screenshots", self.source)
        self.cache_directory = self.root / "builder-cache"
        self.cache_file = self.cache_directory / "ocr_cache.sqlite3"
        self.output = self.source / "Malayalam.pdf"

    def test_sample_pdf_has_pages_bookmarks_positioned_english_and_cache_reuse(self):
        with patch.object(builder, "CACHE_DIRECTORY", self.cache_directory):
            first = builder.run_selected_build(
                self.source,
                cache_file=self.cache_file,
                ocr_function=builder.extract_words,
            )

        reader = PdfReader(self.output)

        self.assertEqual(first.build.page_count, 10)
        self.assertEqual(first.build.searchable_pages, 10)
        self.assertEqual(len(reader.pages), 10)
        self.assertEqual(
            [item.title for item in reader.outline],
            ["Trial", "Class 1", "Class 2"],
        )
        self.assertEqual(
            [reader.get_destination_page_number(item) for item in reader.outline],
            [0, 1, 7],
        )

        positions = []
        for page_number, page in enumerate(reader.pages):
            def visitor(text, _cm, tm, _font, _size):
                if text.strip() == "That":
                    positions.append((page_number, tm[4], tm[5]))
            page.extract_text(visitor_text=visitor)

        self.assertEqual({page for page, _x, _y in positions}, {4, 5, 7, 9})
        self.assertTrue(all((x, y) != (5, 5) for _page, x, y in positions))

        def unexpected_ocr(path):
            raise AssertionError(f"unchanged screenshot unexpectedly OCRed: {path}")

        with patch.object(builder, "CACHE_DIRECTORY", self.cache_directory):
            second = builder.run_selected_build(
                self.source,
                cache_file=self.cache_file,
                ocr_function=unexpected_ocr,
            )

        self.assertEqual(second.build.page_count, 10)
        self.assertEqual(second.build.searchable_pages, 10)
        self.assertEqual(second.incremental.reused, 10)
        self.assertEqual(second.incremental.new, 0)
        self.assertEqual(second.incremental.changed, 0)
        self.assertEqual(second.incremental.stale, 0)
