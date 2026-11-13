"""Stage 11 verification: is the packed feature set complete and correct?

The packed set is the one notebook 11 wrote: features.npy, index.parquet and
tiles.parquet next to the per-slide files, and a packed section in the
manifest. This test runs src.features.verify on it with the round trip on, so
all nine checks run (handbook Stage 11):
- every expected slide is packed once or listed as excluded, and no packed
  slide is empty;
- the index ranges are contiguous, and the array and tile table match them;
- no NaN or infinity, masks_used is false, the fingerprint matches;
- every slide's packed rows equal its .pt file exactly.

Run it from the repo root (add -s to see the report):

    .venv/bin/pytest tests/test_feature_set.py

or as a plain script:

    PYTHONPATH=. .venv/bin/python tests/test_feature_set.py

It takes about 12 seconds on this PC with the files read from disk (6 when
they are already in memory), most of it the round trip over 10,615 slides and
the fingerprint of the 2 GB array, and fails if any check fails.

Steps
-----
1. The expected slides are every slide in data/folds.csv.
2. If the set is not packed, fail with a message saying so.
3. verify; its error, listing every failed check, is the test's failure.
"""

import os

import pandas as pd

from src.features import FEATURES_FILE, VerificationError, set_dir, verify
from src.paths import DATA_DIR, find_input

VERSION = "uni2h-L1-224-t144-hsv-v1"


def test_feature_set():
    expected_ids = sorted(pd.read_csv(find_input("folds.csv", local_dir=DATA_DIR))["slide_id"])
    if not os.path.exists(os.path.join(set_dir(VERSION), FEATURES_FILE)):
        raise AssertionError(f"{VERSION} is not packed yet: run notebook 11")
    try:
        report = verify(VERSION, expected_ids)
    except VerificationError as error:
        raise AssertionError(str(error)) from None
    print("\n".join(report))


if __name__ == "__main__":
    test_feature_set()
