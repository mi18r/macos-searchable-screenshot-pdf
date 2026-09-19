from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from discovery import discover_classes, extract_timestamp


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent / ".tmp" / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=False)

    def tearDown(self):
        shutil.rmtree(self.root)

    def touch(self, relative_path: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def test_timestamp_accepts_macos_narrow_no_break_space(self):
        parsed = extract_timestamp(
            "Screenshot 2024-12-22 at 8.22.21\u202fPM.png"
        )
        self.assertEqual(parsed.isoformat(), "2024-12-22T20:22:21")

    def test_timestamp_returns_none_for_invalid_calendar_values(self):
        parsed = extract_timestamp(
            "Screenshot 2024-02-30 at 8.22.21 PM.png"
        )
        self.assertIsNone(parsed)

    def test_discovers_class_and_image_order_deterministically(self):
        for name in ("Notes", "Class 10", "Class 2", "Trial", "Class 1"):
            (self.root / name).mkdir()
        self.touch("Class 1/Screenshot 2024-12-22 at 9.00.05 PM.png")
        self.touch("Class 1/Screenshot 2024-12-22 at 8.22.21\u202fPM.png")
        self.touch("Class 1/No timestamp.png")
        self.touch("Class 1/.hidden.png")
        self.touch("Class 1/notes.txt")

        batches = discover_classes(self.root)

        self.assertEqual(
            [batch.name for batch in batches],
            ["Trial", "Class 1", "Class 2", "Class 10", "Notes"],
        )
        self.assertEqual(
            [path.name for path in batches[1].images],
            [
                "Screenshot 2024-12-22 at 8.22.21\u202fPM.png",
                "Screenshot 2024-12-22 at 9.00.05 PM.png",
                "No timestamp.png",
            ],
        )

    def test_discovers_supported_uppercase_image_extensions(self):
        (self.root / "Class 1").mkdir()
        self.touch("Class 1/Alpha.PNG")
        self.touch("Class 1/Bravo.JPG")
        self.touch("Class 1/Charlie.JPEG")
        self.touch("Class 1/ignored.gif")
        self.touch("Class 1/ignored.webp")

        batches = discover_classes(self.root)

        self.assertEqual(
            [path.name for path in batches[0].images],
            ["Alpha.PNG", "Bravo.JPG", "Charlie.JPEG"],
        )

    def test_multiple_fallback_folders_and_files_sort_by_name(self):
        for name in ("Zebra", "Archive"):
            (self.root / name).mkdir()
        self.touch("Archive/Zulu.png")
        self.touch("Archive/Alpha.png")

        batches = discover_classes(self.root)

        self.assertEqual([batch.name for batch in batches], ["Archive", "Zebra"])
        self.assertEqual(
            [path.name for path in batches[0].images],
            ["Alpha.png", "Zulu.png"],
        )

    def test_equal_numeric_classes_use_folder_name_as_tie_breaker(self):
        for name in ("Class 1", "Class 01"):
            (self.root / name).mkdir()

        original_iterdir = Path.iterdir

        def controlled_iterdir(path):
            if path == self.root:
                return iter((self.root / "Class 1", self.root / "Class 01"))
            return original_iterdir(path)

        with patch.object(Path, "iterdir", controlled_iterdir):
            batches = discover_classes(self.root)

        self.assertEqual([batch.name for batch in batches], ["Class 01", "Class 1"])

    def test_missing_source_raises_clear_error(self):
        missing = self.root / "missing"
        with self.assertRaisesRegex(FileNotFoundError, "Screenshot source folder"):
            discover_classes(missing)
