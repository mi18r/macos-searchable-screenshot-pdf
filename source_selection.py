import argparse
from pathlib import Path
import subprocess


FINDER_SCRIPT = (
    'POSIX path of (choose folder with prompt '
    '"Select the folder containing Trial and Class folders")'
)


class SourceSelectionError(RuntimeError):
    pass


def _nonblank_source(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError("invalid source: path cannot be blank")
    return Path(value)


def parse_source_argument(argv: list[str] | None = None) -> Path | None:
    parser = argparse.ArgumentParser(
        description="Build one searchable PDF from class screenshot folders."
    )
    parser.add_argument(
        "--source",
        type=_nonblank_source,
        help="Folder containing Trial, Class 1, Class 2, and later classes.",
    )
    args = parser.parse_args(argv)
    return args.source.expanduser().resolve() if args.source is not None else None


def select_source_folder(run_command=subprocess.run) -> Path | None:
    result = run_command(
        ["osascript", "-e", FINDER_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        value = result.stdout.strip()
        if not value:
            raise SourceSelectionError("Finder returned an empty folder path")
        return Path(value).resolve()
    if "-128" in result.stderr or "User canceled" in result.stderr:
        return None
    detail = result.stderr.strip() or f"osascript exited {result.returncode}"
    raise SourceSelectionError(f"Could not select screenshot folder: {detail}")
