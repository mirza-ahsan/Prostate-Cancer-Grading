"""Where things are, locally and on Kaggle.

1. ON_KAGGLE is True inside a Kaggle notebook, where /kaggle/input exists.
2. DATA_DIR holds the competition data: on Kaggle the first of
   KAGGLE_DATA_DIRS that exists (Kaggle has mounted competitions both directly
   under /kaggle/input and under /kaggle/input/competitions), the repository's
   data/ folder locally. Local paths are built from this file's location, so
   they work from any working directory.
3. IMAGE_DIR holds the slide images; MASK_DIR holds the label masks, which only
   scoring code may read, never tiling.
4. DERIVED_DIR is where the project writes its own files. On Kaggle it is
   /kaggle/working/derived/, because /kaggle/input is read-only.
5. FEATURES_DIR holds the feature sets, one folder per version string, next to
   derived/: data/features/ locally, /kaggle/working/features/ on Kaggle.
6. find_input(name) locates a file made by an earlier notebook: in local_dir
   first (DERIVED_DIR unless told otherwise), then on Kaggle anywhere under
   /kaggle/input, at any depth, because Kaggle nests attached datasets and
   notebook outputs differently over time. It prints where it found the file.
   If the file is nowhere, it stops with a clear error, or returns None when
   required=False.
7. slide_path(slide_id) is the full path of a slide's .tiff.
8. weights_path() is the UNI2-h weights file: data/weights/uni2h.pt locally,
   the attached private dataset on Kaggle. It is a function rather than a
   constant so that importing this module never fails on a Kaggle notebook
   that has no weights attached.
9. MODELS_DIR holds trained heads locally (data/models/, not tracked);
   head_path(name) is one head's file there, or in an attached dataset on
   Kaggle.
"""

import glob
import os

ON_KAGGLE = os.path.exists("/kaggle/input")

KAGGLE_DATA_DIRS = (
    "/kaggle/input/competitions/prostate-cancer-grade-assessment/",
    "/kaggle/input/prostate-cancer-grade-assessment/",
)

if ON_KAGGLE:
    DATA_DIR = next((d for d in KAGGLE_DATA_DIRS if os.path.isdir(d)), KAGGLE_DATA_DIRS[-1])
    DERIVED_DIR = "/kaggle/working/derived/"
    FEATURES_DIR = "/kaggle/working/features/"
    MODELS_DIR = "/kaggle/working/models/"
else:
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/")
    DERIVED_DIR = os.path.join(DATA_DIR, "derived/")
    FEATURES_DIR = os.path.join(DATA_DIR, "features/")
    MODELS_DIR = os.path.join(DATA_DIR, "models/")

IMAGE_DIR = os.path.join(DATA_DIR, "train_images/")
MASK_DIR = os.path.join(DATA_DIR, "train_label_masks/")
WEIGHTS_NAME = "uni2h.pt"


def find_input(name, required=True, local_dir=None):
    """Return the path of a file made earlier, or None if optional and missing."""
    candidates = [os.path.join(local_dir or DERIVED_DIR, name)]
    if ON_KAGGLE:
        candidates += sorted(glob.glob(os.path.join("/kaggle/input", "**", name), recursive=True))
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


def head_path(name):
    """Path of a trained head's file, <name>.pt: in MODELS_DIR locally, in an attached dataset on Kaggle."""
    if ON_KAGGLE:
        return find_input(f"{name}.pt")
    return os.path.join(MODELS_DIR, f"{name}.pt")
