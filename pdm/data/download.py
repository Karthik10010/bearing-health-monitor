"""Download CWRU .mat files into data/raw/.

Files already present (and readable) are skipped. Each new file is opened with scipy.io.loadmat
(data only, nothing executed) and must contain a drive-end signal. If scripted download is blocked,
download the files manually from the CWRU site into data/raw/ using their numeric names (e.g. 105.mat).
"""
from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path

from pdm.config import path
from pdm.data.cwru_catalog import CATALOG, CwruFile
from pdm.data.loader import read_cwru_mat

log = logging.getLogger(__name__)
USER_AGENT = "Mozilla/5.0 (research-download)"


def _is_valid(dest: Path, entry: CwruFile) -> bool:
    try:
        read_cwru_mat(dest, entry.file_number)
        return True
    except Exception as exc:  # corrupted / partial / HTML error page
        log.warning("%s is not a valid CWRU file: %s", dest.name, exc)
        return False


def download_file(entry: CwruFile, raw_dir: Path, retries: int = 3, timeout: int = 60) -> Path:
    dest = raw_dir / entry.filename
    if dest.exists() and _is_valid(dest, entry):
        log.info("skip %s (already present)", dest.name)
        return dest
    tmp = dest.with_suffix(".part")
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(entry.url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
                while chunk := resp.read(1 << 16):
                    out.write(chunk)
            tmp.replace(dest)
            if not _is_valid(dest, entry):
                raise ValueError("downloaded file failed validation")
            log.info("downloaded %s (%s)", dest.name, entry.name)
            return dest
        except Exception as exc:
            last_exc = exc
            log.warning("attempt %d/%d for %s failed: %s", attempt, retries, entry.url, exc)
            time.sleep(2 * attempt)
    raise RuntimeError(
        f"Could not download {entry.url}: {last_exc}. Download it manually into {raw_dir} as {entry.filename}."
    )


def download_all(raw_dir: Path | None = None) -> list[Path]:
    raw_dir = raw_dir or path("raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    return [download_file(e, raw_dir) for e in CATALOG]


def download_holdout(unseen_dir: Path | None = None) -> list[Path]:
    """The 4 never-used outer-race recordings (@3 / @12 o'clock) into data/unseen/."""
    from pdm.data.cwru_catalog import HOLDOUT

    unseen_dir = unseen_dir or path("unseen_dir")
    unseen_dir.mkdir(parents=True, exist_ok=True)
    return [download_file(e, unseen_dir) for e in HOLDOUT]
