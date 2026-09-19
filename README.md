# macOS Searchable Screenshot PDF

Build one chronological, bookmarked, searchable PDF from folders of screenshots on macOS. Each screenshot becomes an A4 page, while Apple Vision OCR adds an invisible English text layer at the matching word positions.

The application runs locally. It does not upload screenshots or OCR data.

## Features

- Orders `Folder 0` first, followed by numerically ordered `Folder N` folders.
- Sorts standard macOS screenshot filenames by their embedded timestamps.
- Uses Apple Vision for word-level English OCR.
- Adds Folder bookmarks to the finished PDF.
- Reuses cached OCR for screenshots that have not changed.
- Rebuilds the complete PDF safely when images are added, changed, or removed.
- Preserves an existing output file if OCR, validation, or PDF generation fails.
- Rejects symbolic links for folders, screenshots, and the output file.

## Requirements

- macOS with Apple Vision support
- Python 3.13
- Git

The current release was verified with Python 3.13.5. Other Python versions have not been tested.

## Installation

```bash
git clone https://github.com/mi18r/macos-searchable-screenshot-pdf.git
cd macos-searchable-screenshot-pdf
python3.13 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## Input folder

Choose a folder whose direct subfolders contain the screenshots. For example:

```text
My Screenshots Folder/
├── Folder 0/
│   └── screenshot1.png
├── Folder 1/
│   ├── Screenshot 2026-09-01 at 9.00.00 AM.png
│   └── Screenshot 2026-09-01 at 9.05.00 AM.png
└── Folder 2/
    └── Screenshot 2026-09-08 at 9.00.00 AM.png
```

Supported image extensions are `.png`, `.jpg`, and `.jpeg`, case-insensitively. Hidden entries are ignored. Empty folders do not receive bookmarks.

## Usage

Open the macOS folder picker:

```bash
./.venv/bin/python main.py
```

Or supply the source folder directly:

```bash
./.venv/bin/python main.py --source "/path/to/My Screenshots Folder"
```

The application creates `output.pdf` inside the selected source folder. The filename is currently fixed.

The first run performs OCR for every screenshot. Later runs classify each file as:

- `new`: no cache entry exists;
- `changed`: its size or modification time changed;
- `stale`: the cached OCR format or settings are outdated; or
- `reused`: the cached OCR is still current.

OCR data, locks, and temporary working files are stored under the repository's ignored `.cache/` directory. When no build is running, removing `.cache/` is safe; the next run performs OCR again. Do not remove it during an active build because it contains the coordination locks.

## Verify the PDF

Open `output.pdf` in Preview, show the table-of-contents sidebar, and confirm that each non-empty folders has a bookmark. Search for an English word visible in multiple screenshots and use **Next** to confirm the matches appear on the correct pages and locations.

## Tests

Run the ordinary automated suite from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -m unittest discover -s tests -v
```

The suite uses generated temporary images and does not need personal screenshots. The real Apple Vision sample integration test remains opt-in because it expects the original developer's private fixture set, which is intentionally excluded from this repository.

## Troubleshooting

- **No folder picker appears:** pass the source explicitly with `--source`.
- **No supported screenshots found:** confirm that images are inside `Folder 0` or `Folder N` subfolders and use a supported extension.
- **Apple Vision import or OCR fails:** confirm you are running on macOS and installed all packages from `requirements.txt` in the active virtual environment.
- **Permission denied:** choose a source folder that Terminal is allowed to read and write. macOS may ask you to grant access.
- **Another build is running:** wait for the active build for that source folder to finish.
- **Search finds no text:** Apple Vision may not have recognized usable English text on that screenshot. The visible page is still included.

## Privacy and repository contents

Common screenshot image formats, screenshot folders, generated PDFs, OCR caches, virtual environments, internal planning records, local instruction files such as `AGENTS.md`, and machine-specific files are excluded by `.gitignore`. Review `git status` before every commit if you intentionally add a non-sensitive fixture with `git add -f`.

## Current limitations

- OCR is macOS-only because it uses Apple Vision through PyObjC.
- English search is the acceptance target; other scripts remain visible but may not be searchable.
- The output filename is fixed as `output.pdf`.
- The opt-in sample integration test requires a private fixture set and is skipped by default.

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the architecture, verification strategy, and possible next steps.
