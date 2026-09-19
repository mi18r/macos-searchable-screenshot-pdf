from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tempfile
from typing import BinaryIO, Callable

from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from discovery import ClassBatch, SUPPORTED_IMAGE_SUFFIXES, discover_classes
from incremental import (
    IncrementalStats,
    apply_prepared_ocr,
    prepare_ocr,
    validate_prepared_ocr,
)
from ocr import extract_words
from ocr_cache import OCRCache
from pdf_layer import (
    calculate_image_placement,
    create_text_layer,
    is_usable_ocr_word,
)
from run_lock import build_lock, source_identity
from source_selection import parse_source_argument, select_source_folder


# ==============================
# Configuration
# ==============================

BUILDER_DIRECTORY = Path(__file__).resolve().parent

CACHE_DIRECTORY = BUILDER_DIRECTORY / ".cache"

CACHE_FILE = CACHE_DIRECTORY / "ocr_cache.sqlite3"

SOURCE_FOLDER = BUILDER_DIRECTORY.parent / "Screenshots"

OUTPUT_FILE = BUILDER_DIRECTORY.parent / "Malayalam.pdf"

TEMP_FILE = BUILDER_DIRECTORY / "temporary.pdf"


@dataclass(frozen=True)
class BuildSummary:
    page_count: int
    searchable_pages: int
    ocr_failures: tuple[Path, ...]


@dataclass(frozen=True)
class RunSummary:
    source_folder: Path
    output_file: Path
    class_count: int
    screenshot_count: int
    build: BuildSummary
    incremental: IncrementalStats


PDFProgress = Callable[[int, int, Path, str], None]


@dataclass
class OwnedTemporaryFile:
    path: Path
    stream: BinaryIO
    device: int
    inode: int

    def close(self) -> None:
        if not self.stream.closed:
            self.stream.close()

    def validate_for_replace(self) -> None:
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.close()
        try:
            status = self.path.lstat()
        except OSError as error:
            raise RuntimeError(
                f"Owned temporary file disappeared before replacement: {self.path}"
            ) from error
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or (status.st_dev, status.st_ino) != (self.device, self.inode)
        ):
            raise RuntimeError(
                f"Owned temporary file changed before replacement: {self.path}"
            )


def _ensure_real_directory(path: Path, purpose: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    status = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"Unsafe {purpose} directory: {path}")


def _create_owned_temporary(
    directory: Path, *, prefix: str, suffix: str, purpose: str
) -> OwnedTemporaryFile:
    directory = Path(directory)
    _ensure_real_directory(directory, purpose)
    try:
        stream = tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=directory,
            prefix=prefix,
            suffix=suffix,
            delete=False,
        )
        status = os.fstat(stream.fileno())
        temporary = OwnedTemporaryFile(
            path=Path(stream.name),
            stream=stream,
            device=status.st_dev,
            inode=status.st_ino,
        )
        path_status = temporary.path.lstat()
        if (
            not stat.S_ISREG(status.st_mode)
            or status.st_nlink != 1
            or (path_status.st_dev, path_status.st_ino)
            != (status.st_dev, status.st_ino)
        ):
            raise RuntimeError(f"Could not securely create {purpose}: {temporary.path}")
        return temporary
    except Exception:
        if "stream" in locals() and not stream.closed:
            stream.close()
        if "temporary" in locals():
            try:
                current = temporary.path.lstat()
            except OSError:
                pass
            else:
                if (current.st_dev, current.st_ino) == (
                    temporary.device,
                    temporary.inode,
                ):
                    try:
                        temporary.path.unlink(missing_ok=True)
                    except OSError:
                        pass
        raise


def _cleanup_owned_temporaries(
    temporaries: list[OwnedTemporaryFile], *, suppress_errors: bool
) -> None:
    failures = []
    for temporary in reversed(temporaries):
        try:
            temporary.close()
            try:
                status = temporary.path.lstat()
            except FileNotFoundError:
                continue
            if (status.st_dev, status.st_ino) != (
                temporary.device,
                temporary.inode,
            ):
                failures.append(
                    f"temporary path ownership changed; left untouched: {temporary.path}"
                )
                continue
            temporary.path.unlink()
        except OSError as error:
            failures.append(f"{temporary.path}: {error}")
    if failures and not suppress_errors:
        raise RuntimeError("Failed to clean owned temporary files: " + "; ".join(failures))


def _validate_output_destination(output_file: Path) -> None:
    output_file = Path(output_file)
    if output_file.is_symlink():
        raise RuntimeError(
            f"Refusing to replace symbolic link output: {output_file}"
        )
    try:
        status = output_file.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(status.st_mode):
        raise RuntimeError(f"Output path is not a regular file: {output_file}")


def _validate_source_symlinks(source_folder: Path) -> None:
    for class_folder in source_folder.iterdir():
        if class_folder.name.startswith("."):
            continue
        if class_folder.is_symlink():
            raise RuntimeError(
                f"Symbolic link class entry is not allowed: {class_folder}"
            )
        if not class_folder.is_dir():
            continue
        for image_path in class_folder.iterdir():
            if image_path.name.startswith("."):
                continue
            if (
                image_path.suffix.casefold() in SUPPORTED_IMAGE_SUFFIXES
                and image_path.is_symlink()
            ):
                raise RuntimeError(
                    f"Symbolic link screenshot is not allowed: {image_path}"
                )


def create_image_pdf(
    page_entries: list[tuple[str, Path]],
    working_pdf: Path | BinaryIO,
    progress: PDFProgress | None = None,
) -> None:
    destination = str(working_pdf) if isinstance(working_pdf, Path) else working_pdf
    pdf = canvas.Canvas(destination, pagesize=A4)
    total = len(page_entries)
    for current, (_class_name, image_path) in enumerate(page_entries, start=1):
        try:
            if progress is not None:
                progress(current, total, image_path, "image")
            with Image.open(image_path) as image:
                image.load()
                placement = calculate_image_placement(*image.size)
                pdf.drawImage(
                    ImageReader(image),
                    placement.x,
                    placement.y,
                    width=placement.width,
                    height=placement.height,
                )
            pdf.showPage()
        except Exception as error:
            raise RuntimeError(f"Failed to render image: {image_path}") from error
    pdf.save()


def _assemble_pdf(
    page_entries: list[tuple[str, Path]],
    output_temporary: OwnedTemporaryFile,
    working_temporary: OwnedTemporaryFile,
    output_display_path: Path,
    ocr_function,
    progress: PDFProgress | None,
) -> BuildSummary:
    ocr_failures = []
    searchable_pages = 0
    working_temporary.stream.seek(0)
    working_temporary.stream.truncate(0)
    create_image_pdf(page_entries, working_temporary.stream, progress)
    working_temporary.stream.flush()
    working_temporary.stream.seek(0)
    base_pdf = PdfReader(working_temporary.stream)
    writer = PdfWriter()
    bookmark_pages = {}
    for page_index, ((class_name, image_path), page) in enumerate(
        zip(page_entries, base_pdf.pages, strict=True)
    ):
        bookmark_pages.setdefault(class_name, page_index)
        writer.add_page(page)
        output_page = writer.pages[-1]
        if progress is not None:
            progress(page_index + 1, len(page_entries), image_path, "search")
        with Image.open(image_path) as image:
            placement = calculate_image_placement(*image.size)
        words = ocr_function(image_path)
        usable_words = [word for word in words if is_usable_ocr_word(word)]
        if usable_words:
            overlay = PdfReader(create_text_layer(usable_words, placement)).pages[0]
            output_page.merge_page(overlay)
            searchable_pages += 1
        else:
            print(f"WARNING: OCR found no English text in {image_path}")
            ocr_failures.append(image_path)
    for title, page_index in bookmark_pages.items():
        writer.add_outline_item(title, page_number=page_index)
    try:
        output_temporary.stream.seek(0)
        output_temporary.stream.truncate(0)
        writer.write(output_temporary.stream)
    except Exception as error:
        raise RuntimeError(
            f"Failed to write PDF {output_display_path}: {error}"
        ) from error
    return BuildSummary(len(page_entries), searchable_pages, tuple(ocr_failures))


def _page_entries(batches: list[ClassBatch]) -> list[tuple[str, Path]]:
    return [(batch.name, image) for batch in batches for image in batch.images]


def _new_build_temporaries(
    source_folder: Path,
    output_file: Path,
    working_pdf: Path,
) -> tuple[OwnedTemporaryFile, OwnedTemporaryFile]:
    identity_text = "-".join(str(value) for value in source_identity(source_folder))
    output_temporary = _create_owned_temporary(
        output_file.parent,
        prefix=f".{output_file.name}.",
        suffix=".ready",
        purpose="staged PDF",
    )
    try:
        working_temporary = _create_owned_temporary(
            working_pdf.parent,
            prefix=f"{identity_text}.",
            suffix=".images.pdf",
            purpose="working PDF",
        )
    except Exception:
        _cleanup_owned_temporaries([output_temporary], suppress_errors=True)
        raise
    return output_temporary, working_temporary


def build_pdf(
    source_folder: Path,
    output_file: Path,
    working_pdf: Path,
    ocr_function=extract_words,
    batches: list[ClassBatch] | None = None,
    progress: PDFProgress | None = None,
) -> BuildSummary:
    source_folder = Path(source_folder)
    output_file = Path(output_file)
    if batches is None:
        batches = discover_classes(source_folder)
    page_entries = _page_entries(batches)
    if not page_entries:
        raise ValueError(f"No supported screenshots found in: {source_folder}")

    _validate_output_destination(output_file)
    output_temporary, working_temporary = _new_build_temporaries(
        source_folder, output_file, Path(working_pdf)
    )
    primary_error = None
    try:
        summary = _assemble_pdf(
            page_entries,
            output_temporary,
            working_temporary,
            output_file,
            ocr_function,
            progress,
        )
        _cleanup_owned_temporaries([working_temporary], suppress_errors=False)
        output_temporary.validate_for_replace()
        _validate_output_destination(output_file)
        try:
            output_temporary.path.replace(output_file)
        except Exception as error:
            raise RuntimeError(f"Failed to replace PDF {output_file}: {error}") from error
        return summary
    except BaseException as error:
        primary_error = error
        raise
    finally:
        _cleanup_owned_temporaries(
            [working_temporary, output_temporary],
            suppress_errors=primary_error is not None,
        )


def working_path_for(source_folder: Path, cache_directory: Path) -> Path:
    work_directory = cache_directory / "work"
    _ensure_real_directory(work_directory, "working PDF")
    identity_text = "-".join(str(value) for value in source_identity(source_folder))
    return work_directory / f"{identity_text}.images.pdf"


def _ocr_progress(current: int, total: int, image_path: Path, state: str) -> None:
    print(f"Preparing OCR {current}/{total} [{state}]: {image_path.name}")


def _pdf_progress(current: int, total: int, image_path: Path, phase: str) -> None:
    if phase == "image":
        label = "Building image page"
    else:
        label = "Adding search layer"
    print(f"{label} {current}/{total}: {image_path.name}")


def run_selected_build(
    source_folder: Path,
    cache_file: Path = CACHE_FILE,
    ocr_function=extract_words,
) -> RunSummary:
    source = Path(source_folder).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(
            f"Screenshot source folder does not exist: {source}"
        )

    output = source / "Malayalam.pdf"
    with build_lock(CACHE_DIRECTORY, source):
        _validate_output_destination(output)
        _validate_source_symlinks(source)
        output_temporary = None
        working_temporary = None
        primary_error = None
        try:
            batches = discover_classes(source)
            screenshot_count = sum(len(batch.images) for batch in batches)
            if screenshot_count == 0:
                raise ValueError(f"No supported screenshots found in: {source}")

            with OCRCache(Path(cache_file)) as cache:
                prepared = prepare_ocr(
                    source,
                    batches,
                    cache,
                    ocr_function,
                    _ocr_progress,
                )
                validate_prepared_ocr(prepared)
                working = working_path_for(source, CACHE_DIRECTORY)
                output_temporary, working_temporary = _new_build_temporaries(
                    source, output, working
                )
                build = _assemble_pdf(
                    _page_entries(batches),
                    output_temporary,
                    working_temporary,
                    output,
                    prepared.words_for,
                    _pdf_progress,
                )
                validate_prepared_ocr(prepared)
                output_temporary.validate_for_replace()
                _cleanup_owned_temporaries(
                    [working_temporary], suppress_errors=False
                )
                with cache.transaction():
                    apply_prepared_ocr(cache, prepared)

            validate_prepared_ocr(prepared)
            output_temporary.validate_for_replace()
            _validate_output_destination(output)
            try:
                output_temporary.path.replace(output)
            except Exception as error:
                raise RuntimeError(
                    f"Failed to replace output {output}: {error}"
                ) from error

            return RunSummary(
                source_folder=source,
                output_file=output,
                class_count=len(batches),
                screenshot_count=screenshot_count,
                build=build,
                incremental=prepared.stats,
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            _cleanup_owned_temporaries(
                [
                    temporary
                    for temporary in (working_temporary, output_temporary)
                    if temporary is not None
                ],
                suppress_errors=primary_error is not None,
            )


def print_run_summary(summary: RunSummary) -> None:
    print(f"Selected folder: {summary.source_folder}")
    print(f"Class folders: {summary.class_count}")
    print(f"Screenshots found: {summary.screenshot_count}")
    print(f"Reused OCR: {summary.incremental.reused}")
    print(f"New screenshots: {summary.incremental.new}")
    print(f"Changed screenshots: {summary.incremental.changed}")
    print(f"Stale screenshots: {summary.incremental.stale}")
    print(f"Removed screenshots: {summary.incremental.removed}")
    print(
        f"Searchable pages: {summary.build.searchable_pages}/"
        f"{summary.build.page_count}"
    )
    for path in summary.build.ocr_failures:
        print(f"Non-searchable page: {path}")
    print(f"Created: {summary.output_file}")


def create_pdf() -> BuildSummary:
    print(f"Creating searchable PDF: {OUTPUT_FILE}")
    summary = build_pdf(SOURCE_FOLDER, OUTPUT_FILE, TEMP_FILE)
    print(f"Created {summary.page_count}-page PDF at {OUTPUT_FILE}")
    print(f"Searchable pages: {summary.searchable_pages}/{summary.page_count}")
    for path in summary.ocr_failures:
        print(f"Non-searchable page: {path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    try:
        source = parse_source_argument(argv)
        if source is None:
            source = select_source_folder()
        if source is None:
            print("Folder selection cancelled. No files were changed.")
            return 0
        summary = run_selected_build(source)
    except Exception as error:
        print(f"ERROR: {error}")
        return 1
    print_run_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
