from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from subprocess import CompletedProcess
import unittest

from source_selection import (
    FINDER_SCRIPT,
    SourceSelectionError,
    parse_source_argument,
    select_source_folder,
)


class SourceSelectionTests(unittest.TestCase):
    def test_explicit_source_is_resolved_and_bypasses_picker(self):
        selected = parse_source_argument(["--source", "../Screenshots"])
        self.assertEqual(selected, Path("../Screenshots").resolve())

    def test_missing_source_argument_requests_finder(self):
        self.assertIsNone(parse_source_argument([]))

    def test_blank_explicit_source_is_rejected_by_argparse(self):
        for value in ("", "   ", "\t\n"):
            with self.subTest(value=value):
                stderr = StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    parse_source_argument(["--source", value])

                self.assertEqual(raised.exception.code, 2)
                self.assertIn("invalid source", stderr.getvalue().lower())
                self.assertIn("blank", stderr.getvalue().lower())

    def test_finder_returns_resolved_posix_path(self):
        def fake_run(command, **kwargs):
            self.assertEqual(command, ["osascript", "-e", FINDER_SCRIPT])
            self.assertEqual(
                kwargs,
                {"capture_output": True, "text": True, "check": False},
            )
            return CompletedProcess(command, 0, stdout="/tmp/My Screenshots/\n", stderr="")

        self.assertEqual(
            select_source_folder(fake_run),
            Path("/tmp/My Screenshots").resolve(),
        )

    def test_finder_cancellation_returns_none(self):
        def fake_run(command, **_kwargs):
            return CompletedProcess(command, 1, stdout="", stderr="User canceled. (-128)\n")

        self.assertIsNone(select_source_folder(fake_run))

    def test_unexpected_finder_failure_is_reported(self):
        def fake_run(command, **_kwargs):
            return CompletedProcess(command, 1, stdout="", stderr="Apple event failed")

        with self.assertRaisesRegex(SourceSelectionError, "Apple event failed"):
            select_source_folder(fake_run)
