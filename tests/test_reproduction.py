"""Stage 9 reproduction test: does inference reproduce the stored features?

Five slides are tiled and encoded again, fresh from the .tiff, the way a
submission would do it live, and compared against what the extraction stored.
Two different bugs show up two different ways (handbook Stage 9):
- the coordinates differ: tiling is not deterministic;
- the coordinates match but the features differ: the preprocessing between the
  slide and the encoder is not the same as it was at extraction, for example a
  second copy of the pixel transform.

Run it from the repo root, after any change to tiling, tile count, level,
encoder or preprocessing:

    PYTHONPATH=. .venv/bin/python tests/test_reproduction.py

It takes about a minute, most of it loading the encoder. It exits non-zero on
failure. It is also a pytest test, should pytest be added.

Steps
-----
1. The stored set's manifest must describe the code as it is now: the same
   weights (sha256), tile size, read level, cap and tissue settings, and no
   masks. If not, the stored set came from other code: give the new code a new
   version string rather than reading the difference as a bug.
2. Load the encoder offline, as a submission does.
3. For each slide: get_tiles from the .tiff, encode_tiles, and compare with the
   stored file. Coordinates and tissue fractions must match exactly.
4. Features must match exactly when they were stored on the GPU in use now.
   On another GPU, fp16 arithmetic differs slightly (FINDINGS, Stage 7) and by
   how much has not been measured. The test then prints the largest difference
   and the lowest cosine similarity per slide, and fails until
   CROSS_GPU_MIN_COSINE is set from those measurements.

The slides
----------
Chosen once, by rule, from the stored tissue set (all 10,616 slides, encoded on
the RTX 4060):
- per hospital, the slide that hits the 144-tile cap with the most kept tiles
  tied at the cut-off tissue fraction (most slides id first), since the cut
  is where the tie-break on (row, col) decides;
- per hospital, the first slide by id at the median tile count;
- the one slide with zero tiles, which must stay empty.
"""

import hashlib
import json
import os

import numpy as np
import torch

from src import tiling
from src.encode import DEVICE, encode_tiles, load_encoder_offline, prepare
from src.paths import DERIVED_DIR, slide_path, weights_path
from src.tiling import get_tiles

VERSION = "uni2h-L1-224-t144-hsv-v1"
FEATURES_DIR = os.path.join(os.path.dirname(os.path.normpath(DERIVED_DIR)), "features")  # as in notebook 9
SLIDES = {
    "1836f6539ccc9e37d426603cc4526f8b": "Radboud, at the 144-tile cap",
    "4726ffb66a6516baff6de98df2b5a26b": "Karolinska, at the cap, two kept tiles tied at the cut-off",
    "0240270d611683829ea5bee2a8c52eb0": "Radboud, median tile count (59)",
    "03b113d63b06f6740d82b4cfa1ee4a75": "Karolinska, median tile count (70)",
    "3790f55cad63053e956fb73027179707": "the blank slide, zero tiles",
}
CROSS_GPU_MIN_COSINE = None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifest():
    """Step 1: the stored set must come from the code and weights in use now."""
    with open(os.path.join(FEATURES_DIR, VERSION, "manifest.json")) as f:
        manifest = json.load(f)
    expected = {
        "version": VERSION,
        "masks_used": False,
        "weights_sha256": file_sha256(weights_path()),
        "tile_size": tiling.TILE_SIZE,
        "read_level": tiling.READ_LEVEL,
        "tile_cap": tiling.TILE_PER_SLIDE_CAP,
    }
    placement = {
        "saturation_cut": tiling.SATURATION_CUT,
        "min_cc_area": tiling.MIN_CC_AREA,
        "ink_hue": list(tiling.INK_HUE_THRESH),
        "ink_saturation": tiling.INK_SAT_THRESH,
        "min_tissue_frac": tiling.MIN_FRAC,
        "pad_edges": tiling.PAD_EDGES,
    }
    wrong = [f"{key}: stored {manifest.get(key)!r}, now {value!r}" for key, value in expected.items()
             if manifest.get(key) != value]
    wrong += [f"placement.{key}: stored {manifest['placement'].get(key)!r}, now {value!r}"
              for key, value in placement.items() if manifest["placement"].get(key) != value]
    assert not wrong, "the stored set was made with other settings:\n  " + "\n  ".join(wrong)
    print(f"manifest matches the code now (commit {manifest['code_commit']})")


def compare_slide(encoder, gpu, slide_id):
    """Step 3 and 4 for one slide. Returns a list of what did not match."""
    stored = torch.load(os.path.join(FEATURES_DIR, VERSION, "slides", f"{slide_id}.pt"), weights_only=True)
    tiles, coords, tissue_frac = get_tiles(slide_path(slide_id))
    features = encode_tiles(encoder, tiles)
    coords = torch.from_numpy(np.asarray(coords, dtype=np.int32).reshape(-1, 2))
    tissue_frac = torch.from_numpy(np.asarray(tissue_frac, dtype=np.float32))

    problems = []
    if not torch.equal(coords, stored["coords"]):
        problems.append(f"coordinates differ ({len(coords)} tiles now, {len(stored['coords'])} stored)")
    if not torch.equal(tissue_frac, stored["tissue_frac"]):
        problems.append("tissue fractions differ")
    if problems:
        return problems

    if stored["gpu"] == gpu:
        if not torch.equal(features, stored["features"]):
            difference = (features.float() - stored["features"].float()).abs().max().item()
            problems.append(f"features differ on the same GPU (largest difference {difference:.3g})")
        return problems

    if len(features) == 0:
        return problems
    difference = (features.float() - stored["features"].float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(features.float(), stored["features"].float(), dim=1).min().item()
    print(f"    stored on {stored['gpu']}, now {gpu}: largest difference {difference:.3g}, lowest cosine {cosine:.6f}")
    if CROSS_GPU_MIN_COSINE is None:
        problems.append("encoded on another GPU, and the cross-GPU tolerance has not been measured yet")
    elif cosine < CROSS_GPU_MIN_COSINE:
        problems.append(f"lowest cosine {cosine:.6f} is below {CROSS_GPU_MIN_COSINE}")
    return problems


def test_reproduction():
    check_manifest()
    encoder = prepare(load_encoder_offline(weights_path()))
    gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    print(f"encoding on {gpu}")

    failures = {}
    for slide_id, why in SLIDES.items():
        problems = compare_slide(encoder, gpu, slide_id)
        print(f"{'FAIL' if problems else 'ok  '}  {slide_id}  {why}")
        for problem in problems:
            print(f"    {problem}")
        if problems:
            failures[slide_id] = problems
    assert not failures, f"{len(failures)} of {len(SLIDES)} slides did not reproduce"
    print(f"all {len(SLIDES)} slides reproduce")


if __name__ == "__main__":
    test_reproduction()
