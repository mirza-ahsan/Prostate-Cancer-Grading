"""Where things are, locally and on Kaggle.

1. ON_KAGGLE is True inside a Kaggle notebook, where /kaggle/input exists.
2. DATA_DIR holds the competition data: /kaggle/input/... on Kaggle, the
   repository's data/ folder locally. Local paths are built from this file's
   location, so they work from any working directory.
3. IMAGE_DIR holds the slide images; MASK_DIR holds the label masks, which only
   scoring code may read, never tiling.
4. DERIVED_DIR is where the project writes its own files. On Kaggle it is
   /kaggle/working/derived/, because /kaggle/input is read-only.
5. find_input(name) locates a file made by an earlier notebook: in local_dir
   first (DERIVED_DIR unless told otherwise), then on Kaggle in any attached
   dataset, at the top level or inside a derived/ or data/ folder. It prints
   where it found the file. If the file is nowhere, it stops with a clear
   error, or returns None when required=False.
6. slide_path(slide_id) is the full path of a slide's .tiff.
7. weights_path() is the UNI2-h weights file: data/weights/uni2h.pt locally,
   the attached private dataset on Kaggle. It is a function rather than a
   constant so that importing this module never fails on a Kaggle notebook
   that has no weights attached.
"""

import glob
import os

ON_KAGGLE = os.path.exists("/kaggle/input")

if ON_KAGGLE:
    DATA_DIR = "/kaggle/input/prostate-cancer-grade-assessment/"
    DERIVED_DIR = "/kaggle/working/derived/"
else:
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/")
    DERIVED_DIR = os.path.join(DATA_DIR, "derived/")

IMAGE_DIR = os.path.join(DATA_DIR, "train_images/")
MASK_DIR = os.path.join(DATA_DIR, "train_label_masks/")
WEIGHTS_NAME = "uni2h.pt"


def find_input(name, required=True, local_dir=None):
    """Return the path of a file made earlier, or None if optional and missing."""
    candidates = [os.path.join(local_dir or DERIVED_DIR, name)]
    if ON_KAGGLE:
        for subfolder in ("", "derived", "data"):
            candidates += sorted(glob.glob(os.path.join("/kaggle/input", "*", subfolder, name)))
    for path in candidates:
        if os.path.exists(path):
            print(f"{name}: {path}")
            return path
    if required:
        raise FileNotFoundError(f"{name} not found; run the notebook that makes it, "
                                f"or attach its output as a dataset on Kaggle")
    return None


def slide_path(slide_id):
    """Full path of a slide's .tiff."""
    return os.path.join(IMAGE_DIR, f"{slide_id}.tiff")


def weights_path():
    """Path of the UNI2-h weights file: local copy, or the attached dataset on Kaggle."""
    if ON_KAGGLE:
        return find_input(WEIGHTS_NAME)
    return os.path.join(DATA_DIR, "weights", WEIGHTS_NAME)
