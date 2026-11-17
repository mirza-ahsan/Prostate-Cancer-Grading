"""Extract a feature set: tile, optionally stain-normalise, encode, save one file per slide; then its manifest.

Notebook 9 did this in its own cells, one slide after another. Here CPU worker
processes tile and normalise the next slides while the GPU encodes the current
one, so the extra CPU work of stain normalisation hides behind encoding.

1. SlideTiles is a PyTorch Dataset over slide IDs. Each item runs get_tiles on
   the slide file and, if a stain method is given, src.stain.normalise_slide
   with that method's target. It returns the tiles as one uint8 array, the
   level-0 coords, the tissue fractions and whether the stain fit ran. An
   error is returned as a string rather than raised, so one bad slide cannot
   stop an overnight run.
2. extract_set feeds those items to the encoder in the main process and saves
   each slide in notebook 9's per-slide format (src.features reads it),
   atomically: a temporary name, then a rename. Slides already saved are
   skipped, so a re-run resumes. A slide with no tiles is saved with n = 0, so
   it is reported later instead of vanishing. Failed slides are printed as
   FAILED and returned; they have no file, so check_complete stops on them.
3. write_extraction_manifest writes the handbook Part 7 fields notebook 9 wrote, plus a
   stain section: method, target, settings and the slides whose stain fit did
   not run (too little tissue, so their tiles were encoded unchanged).

Every worker is limited to one CPU thread for OpenCV, PyTorch and the numeric
libraries, so NUM_WORKERS workers use about NUM_WORKERS cores.
"""

import datetime
import os
import platform

import cv2
import numpy as np
import timm
import torch
from PIL import Image
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from src import encode, stain, tiling
from src.encode import DEVICE, FEATURE_DIM, encode_tiles
from src.features import (
    PART_SUFFIX,
    SLIDES_SUBDIR,
    code_commit,
    file_sha256,
    read_slide,
    set_dir,
    slide_file,
    write_manifest,
)
from src.paths import FEATURES_DIR, slide_path, weights_path

NUM_WORKERS = 6
GPU_NAME = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"


def one_thread(_worker_id=None):
    """Keep a worker to one CPU thread in every library that would otherwise start its own pool."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    threadpool_limits(1)


class SlideTiles(torch.utils.data.Dataset):
    """Tiles of each slide, stain-normalised if method is given; errors come back as a string."""

    def __init__(self, slide_ids, method=None, target=None):
        self.slide_ids, self.method, self.target = list(slide_ids), method, target

    def __len__(self):
        return len(self.slide_ids)

    def __getitem__(self, i):
        slide_id = self.slide_ids[i]
        try:
            tiles, coords, tissue_frac = tiling.get_tiles(slide_path(slide_id))
            fitted = None
            if self.method is not None:
                tiles, fitted = stain.normalise_slide(self.method, tiles, self.target)
            arrays = stain.to_arrays(tiles)
            coords = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
            return slide_id, arrays, coords, np.asarray(tissue_frac, dtype=np.float32), fitted, None
        except Exception as error:  # noqa: BLE001 -- returned and reported as FAILED by extract_set
            return slide_id, None, None, None, None, repr(error)


def save_slide(version, slide_id, features, coords, tissue_frac, stain_fitted, features_dir=FEATURES_DIR):
    """Write one slide's file atomically, in notebook 9's format plus a stain_fitted flag."""
    record = {
        "version": version,
        "slide_id": slide_id,
        "features": features,
        "coords": torch.from_numpy(coords),
        "tissue_frac": torch.from_numpy(tissue_frac),
        "gpu": GPU_NAME,
        "stain_fitted": stain_fitted,
    }
    path = slide_file(version, slide_id, features_dir)
    torch.save(record, path + PART_SUFFIX)
    os.replace(path + PART_SUFFIX, path)


def extract_set(version, slide_ids, encoder, method=None, target=None, features_dir=FEATURES_DIR,
                num_workers=NUM_WORKERS):
    """Extract every slide not yet saved; returns {slide_id: error} for the slides that failed.

    encoder is a model from src.encode.prepare. method None is the plain
    tissue pipeline; otherwise one of src.stain.METHODS, with target
    (src.stain.TARGETS[method] when None).
    """
    os.makedirs(os.path.join(set_dir(version, features_dir), SLIDES_SUBDIR), exist_ok=True)
    todo = [s for s in slide_ids if not os.path.exists(slide_file(version, s, features_dir))]
    print(f"{version}: {len(slide_ids) - len(todo)} already done, {len(todo)} to do")
    loader = torch.utils.data.DataLoader(SlideTiles(todo, method, target), batch_size=None, shuffle=False,
                                         num_workers=num_workers, worker_init_fn=one_thread,
                                         prefetch_factor=2 if num_workers else None)
    failed = {}
    for slide_id, arrays, coords, tissue_frac, fitted, error in tqdm(loader, total=len(todo), desc=version):
        if error is not None:
            failed[slide_id] = error
            tqdm.write(f"FAILED {slide_id}: {error}")
            continue
        tiles = [Image.fromarray(array) for array in np.asarray(arrays)]
        save_slide(version, slide_id, encode_tiles(encoder, tiles), np.asarray(coords), np.asarray(tissue_frac),
                   fitted, features_dir)
    print(f"{version}: {len(failed)} slides failed")
    return failed


def write_extraction_manifest(version, slide_ids, method=None, target=None, features_dir=FEATURES_DIR,
                              progress=None):
    """Write the set's manifest from its per-slide files (each read and checked with read_slide).

    Same fields as notebook 9's manifests, plus "stain": the method, its target
    and settings, and the slides whose stain fit did not run.
    """
    counts, gpus, not_fitted = {}, set(), []
    for slide_id in progress(slide_ids) if progress else slide_ids:
        counts[slide_id] = len(read_slide(version, slide_id, features_dir)[0])
        record = torch.load(slide_file(version, slide_id, features_dir), map_location="cpu", weights_only=True)
        gpus.add(record["gpu"])
        if method is not None and counts[slide_id] > 0 and not record.get("stain_fitted"):
            not_fitted.append(slide_id)
    stain_section = None
    if method is not None:
        stain_section = {
            "method": method,
            "target": target if target is not None else stain.TARGETS[method],
            "settings": {"io": stain.IO, "od_beta": stain.OD_BETA, "macenko_alpha": stain.MACENKO_ALPHA,
                         "max_c_percentile": stain.MAX_C_PERCENTILE, "vahadane_pixels": stain.VAHADANE_PIXELS,
                         "vahadane_lambda": stain.VAHADANE_LAMBDA, "retinex": stain.RETINEX,
                         "min_tissue_pixels": stain.MIN_TISSUE_PIXELS, "fit_pixels": stain.FIT_PIXELS,
                         "seed": stain.SEED},
            "tissue_pixels": "src.tiling saturation cut and ink rule, per pixel",
            "not_fitted_slides": sorted(not_fitted),
        }
    manifest = {
        "version": version,
        "encoder": encode.HF_MODEL,
        "architecture": encode.ARCHITECTURE,
        "weights_sha256": file_sha256(weights_path()),
        "feature_dim": FEATURE_DIM,
        "dtype": "float16",
        "encoded_on": sorted(gpus),
        "tile_size": tiling.TILE_SIZE,
        "read_level": tiling.READ_LEVEL,
        "tile_cap": tiling.TILE_PER_SLIDE_CAP,
        "placement": {
            "method": "src.tiling.get_tiles: tissue mask on the smallest level, grid of tile-sized cells, "
                      "best cells by tissue fraction, then reading order",
            "saturation_cut": tiling.SATURATION_CUT,
            "min_cc_area": tiling.MIN_CC_AREA,
            "ink_hue": list(tiling.INK_HUE_THRESH),
            "ink_saturation": tiling.INK_SAT_THRESH,
            "min_tissue_frac": tiling.MIN_FRAC,
            "pad_edges": tiling.PAD_EDGES,
        },
        "stain": stain_section,
        "masks_used": False,
        "coordinate_space": "level-0 (x, y) of each tile's top-left corner",
        "slides": len(counts),
        "tiles": int(sum(counts.values())),
        "zero_tile_slides": sorted(s for s, n in counts.items() if n == 0),
        "slide_selection": {"source": "data/folds.csv", "method": "every slide"},
        "torch": torch.__version__,
        "timm": timm.__version__,
        "python": platform.python_version(),
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "code_commit": code_commit(),
    }
    write_manifest(version, manifest, features_dir)
    return manifest
