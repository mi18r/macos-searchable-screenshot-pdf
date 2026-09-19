# Implementation Plan

## Goal

Create a reliable macOS command-line application that turns ordered screenshot folders into one bookmarked PDF with an invisible, position-aware English search layer. All screenshots and OCR data remain on the local Mac.

## Constraints

- Apple Vision OCR makes this version macOS-only.
- The input folder contains direct child folders such as `Trial` and `Class N`.
- The output is `<selected source>/Malayalam.pdf`.
- The current release recognizes English text.
- A failed run must preserve the previous output.

## Architecture

1. `source_selection.py` parses `--source` or opens the macOS Finder folder picker.
2. `discovery.py` validates the selected directory, identifies class folders, accepts supported image formats, and produces deterministic class and screenshot order.
3. `run_lock.py` derives a physical filesystem identity for the source and prevents two builds from writing the same output concurrently.
4. `ocr.py` uses Apple Vision to return words and normalized bounding boxes.
5. `ocr_cache.py` stores source-relative fingerprints and serialized OCR results in SQLite.
6. `incremental.py` classifies screenshots as new, changed, stale, or reusable; it also validates fingerprints before publication and prunes deleted images after a successful build.
7. `pdf_layer.py` calculates A4 image placement and maps normalized OCR boxes into PDF coordinates.
8. `main.py` renders image pages, merges invisible text layers, adds class bookmarks, writes secure temporary files, commits cache updates, and atomically replaces the output.

## Data flow

```text
folder selection
    -> ordered class/image manifest
    -> source-specific build lock
    -> cached or fresh Apple Vision OCR
    -> A4 image pages + invisible text overlays
    -> class bookmarks
    -> staged complete PDF
    -> final fingerprint validation
    -> atomic replacement of Malayalam.pdf
```

## Safety properties

- Class-folder, screenshot, and output symbolic links are rejected.
- Temporary files are created with random exclusive names and tracked by filesystem identity.
- OCR cache writes occur only after the staged PDF is complete.
- Screenshot fingerprints are checked before and after assembly and before publication.
- Build failures leave the previous PDF intact and clean up only temporaries owned by that run.
- Cache transactions remain short; OCR and PDF rendering do not hold a SQLite write transaction.

## Verification milestones

1. Parse every Python module successfully.
2. Install only the direct dependencies in `requirements.txt` into a fresh virtual environment.
3. Run the ordinary unit and integration suite with zero failures.
4. Run the Apple Vision fixture test only with a private, known fixture set.
5. Open a generated PDF in Preview and inspect page count, bookmarks, search results, and highlight positions.
6. Verify that a second unchanged run reuses OCR.
7. Verify add, change, and delete workflows.
8. Audit tracked files for screenshots, generated PDFs, caches, personal absolute paths, credentials, and local instruction files.

## Test coverage

The test suite covers deterministic discovery, timestamp parsing, image layout, OCR word conversion, UTF-16 ranges, cache schemas, source aliases, incremental classifications, pruning, locking, failure preservation, temporary-file ownership, symlink rejection, fingerprint validation, PDF bookmarks, progress reporting, and command-line source selection.

## Release checklist

- Run the complete ordinary suite.
- Confirm the real Vision fixture test remains opt-in.
- Run `git diff --check`.
- Review `git status --short`.
- Confirm ignored assets are not tracked.
- Scan tracked text for personal paths and credential-like filenames.
- Commit to `main` and push the verified commit.

## Future improvements

- Add a configurable output path and filename.
- Replace the private fixture integration test with redistributable synthetic fixtures plus a generic live-OCR smoke test.
- Add a packaged macOS application or signed command-line release.
- Add selectable OCR languages.
- Add a cross-platform OCR backend behind the existing OCR interface.
- Add CI for platform-independent tests and a macOS runner for Apple Vision checks.
