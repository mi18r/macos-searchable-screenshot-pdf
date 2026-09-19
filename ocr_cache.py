from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from ocr import NormalizedBox, OCRWord
from run_lock import source_identity


DATABASE_SCHEMA_VERSION = 1
OCR_RECORD_SCHEMA_VERSION = 1
OCR_CONFIGURATION_VERSION = "apple-vision-en-us-word-boxes-v1"


class OCRCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileFingerprint:
    relative_path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class CacheRecord:
    source_root: str
    fingerprint: FileFingerprint
    image_width: int
    image_height: int
    record_schema_version: int
    ocr_configuration_version: str
    words: tuple[OCRWord, ...]


def source_key(source_root: Path) -> str:
    return ":".join(str(part) for part in source_identity(source_root))


def fingerprint_file(source_root: Path, image_path: Path) -> FileFingerprint:
    stat = image_path.stat()
    return FileFingerprint(
        relative_path=image_path.resolve().relative_to(source_root.resolve()).as_posix(),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _serialize_words(words: tuple[OCRWord, ...]) -> str:
    return json.dumps(
        [
            {
                "text": word.text,
                "confidence": word.confidence,
                "x": word.box.x,
                "y": word.box.y,
                "width": word.box.width,
                "height": word.box.height,
            }
            for word in words
        ],
        separators=(",", ":"),
    )


def _deserialize_words(words_json: str) -> tuple[OCRWord, ...]:
    return tuple(
        OCRWord(
            text=word["text"],
            box=NormalizedBox(
                x=float(word["x"]),
                y=float(word["y"]),
                width=float(word["width"]),
                height=float(word["height"]),
            ),
            confidence=float(word["confidence"]),
        )
        for word in json.loads(words_json)
    )


class OCRCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.connection = None
        try:
            self.connection = sqlite3.connect(self.path, isolation_level=None)
            database_version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if database_version == 0:
                self.connection.execute(
                    f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}"
                )
            elif database_version != DATABASE_SCHEMA_VERSION:
                raise OCRCacheError(
                    f"OCR cache error at {self.path}: unsupported database schema "
                    f"version {database_version}"
                )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_results (
                    source_root TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    image_width INTEGER NOT NULL,
                    image_height INTEGER NOT NULL,
                    record_schema_version INTEGER NOT NULL,
                    ocr_configuration_version TEXT NOT NULL,
                    words_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source_root, relative_path)
                )
                """
            )
        except sqlite3.DatabaseError as error:
            self.close()
            raise OCRCacheError(f"OCR cache error at {self.path}: {error}") from error
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _database_error(self, error: sqlite3.DatabaseError) -> OCRCacheError:
        return OCRCacheError(f"OCR cache error at {self.path}: {error}")

    def get(self, source_root: Path, relative_path: str) -> CacheRecord | None:
        try:
            row = self.connection.execute(
                """
                SELECT file_size, mtime_ns, image_width, image_height,
                       record_schema_version, ocr_configuration_version, words_json
                FROM ocr_results
                WHERE source_root = ? AND relative_path = ?
                """,
                (source_key(Path(source_root)), relative_path),
            ).fetchone()
            if row is None:
                return None
            record_schema_version = row[4]
            if record_schema_version == OCR_RECORD_SCHEMA_VERSION:
                try:
                    words = _deserialize_words(row[6])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise OCRCacheError(
                        f"OCR cache error at {self.path} for {relative_path}: "
                        f"invalid OCR words payload: {error}"
                    ) from error
            else:
                words = ()
            return CacheRecord(
                source_root=source_key(Path(source_root)),
                fingerprint=FileFingerprint(
                    relative_path=relative_path,
                    size=row[0],
                    mtime_ns=row[1],
                ),
                image_width=row[2],
                image_height=row[3],
                record_schema_version=record_schema_version,
                ocr_configuration_version=row[5],
                words=words,
            )
        except sqlite3.DatabaseError as error:
            raise self._database_error(error) from error

    def upsert(self, record: CacheRecord):
        try:
            self.connection.execute(
                """
                INSERT INTO ocr_results (
                    source_root, relative_path, file_size, mtime_ns,
                    image_width, image_height, record_schema_version,
                    ocr_configuration_version, words_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_root, relative_path) DO UPDATE SET
                    file_size = excluded.file_size,
                    mtime_ns = excluded.mtime_ns,
                    image_width = excluded.image_width,
                    image_height = excluded.image_height,
                    record_schema_version = excluded.record_schema_version,
                    ocr_configuration_version = excluded.ocr_configuration_version,
                    words_json = excluded.words_json,
                    updated_at = excluded.updated_at
                """,
                (
                    record.source_root,
                    record.fingerprint.relative_path,
                    record.fingerprint.size,
                    record.fingerprint.mtime_ns,
                    record.image_width,
                    record.image_height,
                    record.record_schema_version,
                    record.ocr_configuration_version,
                    _serialize_words(record.words),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        except sqlite3.DatabaseError as error:
            raise self._database_error(error) from error

    def prune(self, source_root: Path, keep_relative_paths: set[str]) -> int:
        key = source_key(Path(source_root))
        try:
            existing_paths = {
                row[0]
                for row in self.connection.execute(
                    "SELECT relative_path FROM ocr_results WHERE source_root = ?", (key,)
                )
            }
            paths_to_remove = existing_paths - keep_relative_paths
            self.connection.executemany(
                "DELETE FROM ocr_results WHERE source_root = ? AND relative_path = ?",
                ((key, relative_path) for relative_path in paths_to_remove),
            )
            return len(paths_to_remove)
        except sqlite3.DatabaseError as error:
            raise self._database_error(error) from error

    def count_prunable(
        self, source_root: Path, keep_relative_paths: set[str] | frozenset[str]
    ) -> int:
        key = source_key(Path(source_root))
        try:
            existing_paths = {
                row[0]
                for row in self.connection.execute(
                    "SELECT relative_path FROM ocr_results WHERE source_root = ?", (key,)
                )
            }
            return len(existing_paths - set(keep_relative_paths))
        except sqlite3.DatabaseError as error:
            raise self._database_error(error) from error

    @contextmanager
    def transaction(self):
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self
            except BaseException:
                self.connection.rollback()
                raise
            else:
                try:
                    self.connection.commit()
                except BaseException:
                    self.connection.rollback()
                    raise
        except sqlite3.DatabaseError as error:
            raise self._database_error(error) from error

    def count_for_source(self, source_root: Path) -> int:
        try:
            return self.connection.execute(
                "SELECT COUNT(*) FROM ocr_results WHERE source_root = ?",
                (source_key(Path(source_root)),),
            ).fetchone()[0]
        except sqlite3.DatabaseError as error:
            raise self._database_error(error) from error
