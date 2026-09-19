from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
SCREENSHOT_TIMESTAMP = re.compile(
    r"(\d{4}-\d{2}-\d{2}) at (\d{1,2}\.\d{2}\.\d{2})\s*([AP]M)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClassBatch:
    name: str
    images: tuple[Path, ...]


def extract_timestamp(filename: str) -> datetime | None:
    match = SCREENSHOT_TIMESTAMP.search(filename)
    if match is None:
        return None
    value = f"{match.group(1)} {match.group(2).replace('.', ':')} {match.group(3)}"
    try:
        return datetime.strptime(value, "%Y-%m-%d %I:%M:%S %p")
    except ValueError:
        return None


def folder_sort_key(folder: Path) -> tuple[int, int | str, str]:
    name = folder.name.casefold()
    if name == "trial":
        return (0, 0, name)
    match = re.fullmatch(r"class\s*(\d+)", name)
    if match:
        return (1, int(match.group(1)), name)
    return (2, name, name)


def image_sort_key(path: Path) -> tuple[int, datetime, str]:
    timestamp = extract_timestamp(path.name)
    if timestamp is not None:
        return (0, timestamp, path.name.casefold())
    return (1, datetime.max, path.name.casefold())


def discover_classes(source_folder: Path) -> list[ClassBatch]:
    if not source_folder.is_dir():
        raise FileNotFoundError(
            f"Screenshot source folder does not exist: {source_folder}"
        )

    batches = []
    for folder in sorted(
        (
            path
            for path in source_folder.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=folder_sort_key,
    ):
        images = tuple(
            sorted(
                (
                    path
                    for path in folder.iterdir()
                    if path.is_file()
                    and not path.name.startswith(".")
                    and path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
                ),
                key=image_sort_key,
            )
        )
        batches.append(ClassBatch(folder.name, images))
    return batches
