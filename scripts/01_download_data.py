"""Step 1: download the CWRU recordings.

  data/raw/     the 48 recordings used for training and evaluation (pdm/data/cwru_catalog.py: CATALOG)
  data/unseen/  4 extra outer-race recordings NEVER used for training (HOLDOUT) - genuine test files
  data/samples/ a few short demo recordings cut from data/raw/ for the dashboard
"""
import _bootstrap  # noqa: F401

from pdm.data.cwru_catalog import CATALOG, HOLDOUT
from pdm.data.download import download_all, download_holdout
from pdm.data.samples import make_samples

if __name__ == "__main__":
    files = download_all()
    print(f"OK: {len(files)}/{len(CATALOG)} CWRU training files present in data/raw/")
    unseen = download_holdout()
    print(f"OK: {len(unseen)}/{len(HOLDOUT)} never-used test files present in data/unseen/ "
          f"({', '.join(e.filename for e in HOLDOUT)}: 12000 Hz, 1797 RPM, 0 HP)")
    print(f"Demo samples written: {[s['file'] for s in make_samples()]}")
