"""Source-specific advisory locks for PDF builds on macOS."""

from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import stat


class BuildAlreadyRunningError(RuntimeError):
    """Raised when another process is building a PDF for the same source."""


def source_identity(source_root: Path) -> tuple[int, int]:
    """Return the physical filesystem identity of a source directory."""
    path = Path(source_root)
    try:
        status = path.stat()
    except OSError as error:
        raise RuntimeError(f"Could not identify screenshot source: {path}: {error}") from error
    if not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"Screenshot source is not a directory: {path}")
    return status.st_dev, status.st_ino


def lock_name(source_root: Path) -> str:
    """Return the stable lock-file name for one physical source directory."""
    value = ":".join(str(part) for part in source_identity(source_root)).encode("ascii")
    return hashlib.sha256(value).hexdigest() + ".lock"


def _ensure_real_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    status = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(status.st_mode):
        raise RuntimeError(f"Build lock directory is not a real directory: {path}")


@contextmanager
def build_lock(cache_directory: Path, source_root: Path):
    """Acquire a nonblocking exclusive lock for one source directory."""
    cache_directory = Path(cache_directory)
    _ensure_real_directory(cache_directory)
    lock_directory = cache_directory / "locks"
    _ensure_real_directory(lock_directory)
    lock_path = lock_directory / lock_name(source_root)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
        descriptor_status = os.fstat(descriptor)
        path_status = lock_path.lstat()
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or descriptor_status.st_nlink != 1
            or (descriptor_status.st_dev, descriptor_status.st_ino)
            != (path_status.st_dev, path_status.st_ino)
        ):
            raise RuntimeError(f"Unsafe build lock file: {lock_path}")
    except Exception:
        if "descriptor" in locals():
            os.close(descriptor)
        raise
    stream = os.fdopen(descriptor, "a+")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BuildAlreadyRunningError(
                f"A PDF build is already running for: {source_root.resolve()}"
            ) from error
        yield lock_path
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()
