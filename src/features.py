"""Feature sets on disk: check, pack, open and verify (handbook Stage 11).

A feature set is one folder per version string under FEATURES_DIR. Extraction
(notebook 9) writes one .pt file per slide into its slides/ folder, plus
manifest.json. Packing adds three files next to them:

    features.npy    every tile's features, float16 [tiles, FEATURE_DIM],
                    opened memory-mapped, so nothing is read until it is used
    index.parquet   one row per slide: slide_id, start, end (one past the
                    last row), n_tiles
    tiles.parquet   one row per tile: slide_id, tile (0 to n - 1 within the
                    slide), level-0 x and y (int32), tissue_frac (float32)

and a "packed" section in the manifest. Rows follow the slides sorted by ID,
and within a slide the order of its .pt file, so tile k of a slide is row
start + k of features.npy and of tiles.parquet.

Functions
---------
1. set_dir and slide_file: where a set and one slide's file are. Every function
   that touches files takes features_dir, FEATURES_DIR by default, so the same
   code runs on a scratch copy and on the real set.
2. file_sha256 and code_commit: a file's fingerprint and the commit that wrote
   it, for the manifest.
3. read_manifest and write_manifest. Writes are atomic: a temporary name, then
   a rename.
4. check_complete: the slides folder holds exactly the expected slides, and
   the manifest names the version and says masks_used: false.
5. read_slide: the one place a per-slide file is read, with its format checked.
6. count_tiles, then pack: pre-allocate, fill one slide at a time, rename the
   three files only when all are complete, then extend the manifest.
7. open_packed: the array memory-mapped read-only, plus the index, the tile
   table and the manifest. Cheap, keeps no global state, so a PyTorch Dataset
   can call it once per DataLoader worker.
8. verify: nine checks, every failure collected, then one VerificationError
   naming them all.

This module reads feature files only. It imports nothing that reaches labels
or masks.
"""

import datetime
import hashlib
import json
import os
import subprocess
from collections import namedtuple

import numpy as np
import pandas as pd
import torch

from src.encode import FEATURE_DIM
from src.paths import FEATURES_DIR

SLIDES_SUBDIR = "slides"
FEATURES_FILE = "features.npy"
INDEX_FILE = "index.parquet"
TILES_FILE = "tiles.parquet"
MANIFEST_FILE = "manifest.json"
PART_SUFFIX = ".part"
FEATURES_DTYPE = np.float16

Packed = namedtuple("Packed", ["features", "index", "starts", "ends", "tiles", "manifest"])


class VerificationError(ValueError):
    """A packed set failed verify(). failed lists the names of the checks that failed."""

    def __init__(self, message, failed):
        super().__init__(message)
        self.failed = failed


def require(condition, message):
    """Raise ValueError(message) unless condition holds. Used by the checks; unlike assert, never switched off."""
    if not condition:
        raise ValueError(message)


def set_dir(version, features_dir=FEATURES_DIR):
    """The folder of one feature set."""
    return os.path.join(features_dir, version)


def slide_file(version, slide_id, features_dir=FEATURES_DIR):
    """The path of one slide's .pt file in a feature set."""
    return os.path.join(set_dir(version, features_dir), SLIDES_SUBDIR, f"{slide_id}.pt")


def file_sha256(path):
    """The sha256 of a file, as a hex string, read in 16 MB blocks so a large file never sits in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def code_commit():
    """The current short commit, with -dirty if a tracked file other than a manifest has changed.

    Manifests are left out because they are what is being written: counting
    them would mark the commit dirty by the act of recording it. Returns
    "unknown" when git cannot run, as on Kaggle.
    """

    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "--short", "HEAD").strip()
        changed = [
            line
            for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
            if not line.endswith("/" + MANIFEST_FILE)
        ]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + ("-dirty" if changed else "")


def read_manifest(version, features_dir=FEATURES_DIR):
    """A feature set's manifest, as a dictionary."""
    with open(os.path.join(set_dir(version, features_dir), MANIFEST_FILE)) as f:
        return json.load(f)


def write_manifest(version, manifest, features_dir=FEATURES_DIR):
    """Write a feature set's manifest atomically.

    1. Write indented JSON to the manifest's name plus PART_SUFFIX.
    2. Rename it over the real name, so a crash never leaves half a manifest.
    """
    path = os.path.join(set_dir(version, features_dir), MANIFEST_FILE)
    with open(path + PART_SUFFIX, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(path + PART_SUFFIX, path)


def _short_list(items, limit=5):
    """At most limit items, then how many more, for error messages."""
    items = sorted(items)
    more = f" and {len(items) - limit} more" if len(items) > limit else ""
    return ", ".join(items[:limit]) + more


def check_complete(version, expected_ids, features_dir=FEATURES_DIR):
    """Stop unless a set's per-slide files are exactly the expected slides, and its manifest is ours.

    1. If the slides folder is missing, stop and say which folder was expected.
       On Kaggle the per-slide files exist only if attached as a dataset.
    2. List the folder: slide IDs from files ending .pt, and anything else
       (such as a .part left by a crashed write).
    3. Collect the expected slides that are missing, the slides not expected,
       and the leftovers.
    4. The manifest must name this version, and masks_used must be exactly
       false; a missing flag fails too.
    5. If anything was collected, stop once with the whole list.
    """
    folder = os.path.join(set_dir(version, features_dir), SLIDES_SUBDIR)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"no per-slide folder at {folder}; on Kaggle, the per-slide files exist "
                                f"only if the feature set is attached as a dataset")
    names = os.listdir(folder)
    found = {name[:-len(".pt")] for name in names if name.endswith(".pt")}
    leftovers = [name for name in names if not name.endswith(".pt")]
    expected = set(expected_ids)

    problems = []
    if expected - found:
        problems.append(f"{len(expected - found)} expected slides missing: {_short_list(expected - found)}")
    if found - expected:
        problems.append(f"{len(found - expected)} slides not expected: {_short_list(found - expected)}")
    if leftovers:
        problems.append(f"{len(leftovers)} other files in {SLIDES_SUBDIR}/: {_short_list(leftovers)}")
    manifest = read_manifest(version, features_dir)
    if manifest.get("version") != version:
        problems.append(f"manifest names version {manifest.get('version')!r}, not {version!r}")
    if manifest.get("masks_used") is not False:
        problems.append(f"manifest masks_used is {manifest.get('masks_used')!r}, not false")
    if problems:
        raise ValueError(f"{version} is not complete:\n  " + "\n  ".join(problems))


def read_slide(version, slide_id, features_dir=FEATURES_DIR):
    """One slide's features (float16 [n, FEATURE_DIM]), coords (int32 [n, 2]) and tissue fractions (float32 [n]).

    1. Load the .pt file with weights-only loading, which reads plain tensors
       and nothing executable.
    2. Check that its version and slide ID match, that the dtypes and widths
       are as above, and that all three have the same length n.
    3. Stop with the slide ID and what is wrong if any check fails.
    Returns numpy arrays.
    """
    stored = torch.load(slide_file(version, slide_id, features_dir), map_location="cpu", weights_only=True)
    features, coords, tissue_frac = stored["features"], stored["coords"], stored["tissue_frac"]
    problems = []
    if stored.get("version") != version or stored.get("slide_id") != slide_id:
        problems.append(f"file says version {stored.get('version')!r}, slide {stored.get('slide_id')!r}")
    if features.dtype != torch.float16 or features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        problems.append(f"features are {features.dtype} {tuple(features.shape)}")
    if coords.dtype != torch.int32 or coords.ndim != 2 or coords.shape[1] != 2:
        problems.append(f"coords are {coords.dtype} {tuple(coords.shape)}")
    if not torch.is_tensor(tissue_frac) or tissue_frac.dtype != torch.float32 or tissue_frac.ndim != 1:
        problems.append(f"tissue_frac is {getattr(tissue_frac, 'dtype', tissue_frac)!r}")
    elif not len(features) == len(coords) == len(tissue_frac):
        problems.append(f"lengths differ: {len(features)} features, {len(coords)} coords, "
                        f"{len(tissue_frac)} tissue fractions")
    if problems:
        raise ValueError(f"{slide_id}: " + "; ".join(problems))
    return features.numpy(), coords.numpy(), tissue_frac.numpy()


def count_tiles(version, slide_ids, features_dir=FEATURES_DIR, progress=None):
    """Tiles per slide, as a pandas Series indexed by slide ID, in the order given.

    Each slide is read with read_slide, so its format is checked, and only its
    tile count is kept: memory stays flat. progress, if given, wraps the loop
    (for example tqdm).
    """
    slide_ids = list(slide_ids)
    loop = progress(slide_ids) if progress else slide_ids
    counts = [len(read_slide(version, slide_id, features_dir)[0]) for slide_id in loop]
    return pd.Series(counts, index=pd.Index(slide_ids, name="slide_id"), name="n_tiles", dtype="int64")


def pack(version, expected_ids, excluded, features_dir=FEATURES_DIR, overwrite=False, progress=None):
    """Pack a complete per-slide set into features.npy, index.parquet and tiles.parquet, and extend the manifest.

    excluded maps slide ID to the reason it is left out. It is written by hand:
    a zero-tile slide that is not listed is an error to investigate.

    1. Stop if any packed file already exists and overwrite is false, or if an
       excluded slide is not an expected slide.
    2. check_complete.
    3. The slides to pack are the expected ones minus the excluded, sorted by ID.
    4. Count tiles. Stop if an excluded slide has tiles (its reason no longer
       holds) or a packed slide has none (an unlisted zero-tile slide).
    5. Create the array at full size, [total tiles, FEATURE_DIM] float16, as a
       memory-mapped .npy under its temporary name.
    6. Walk the slides in order with a running row counter: write each slide's
       features into rows counter to counter + n, record its index row and its
       n tile rows, move the counter on by n.
    7. The counter must end at the total. Flush the array and close it.
    8. Write the index and the tile table as Parquet under temporary names.
    9. Rename all three to their real names, only now that all are complete.
    10. Fingerprint features.npy.
    11. Add the packed section to the manifest, keeping the extraction's own
        code_commit; the packed section records the packing commit.
    Returns a summary dictionary.
    """
    started = datetime.datetime.now(datetime.UTC)
    folder = set_dir(version, features_dir)
    paths = {name: os.path.join(folder, name) for name in (FEATURES_FILE, INDEX_FILE, TILES_FILE)}
    existing = [name for name, path in paths.items() if os.path.exists(path)]
    if existing and not overwrite:
        raise FileExistsError(f"{version} is already packed ({', '.join(existing)}); pass overwrite=True to repack")
    expected = set(expected_ids)
    if set(excluded) - expected:
        raise ValueError(f"excluded slides that are not expected: {_short_list(set(excluded) - expected)}")

    check_complete(version, expected_ids, features_dir)

    slide_ids = sorted(expected - set(excluded))
    counts = count_tiles(version, slide_ids, features_dir, progress)
    excluded_counts = count_tiles(version, sorted(excluded), features_dir)
    if (excluded_counts > 0).any():
        raise ValueError(f"excluded slides that have tiles: {_short_list(excluded_counts[excluded_counts > 0].index)}")
    if (counts == 0).any():
        raise ValueError(f"zero-tile slides not in the exclusions: {_short_list(counts[counts == 0].index)}")
    total = int(counts.sum())

    features = np.lib.format.open_memmap(paths[FEATURES_FILE] + PART_SUFFIX, mode="w+",
                                         dtype=FEATURES_DTYPE, shape=(total, FEATURE_DIM))
    index_rows, tile_parts = [], []
    row = 0
    loop = progress(slide_ids) if progress else slide_ids
    for slide_id in loop:
        slide_features, coords, tissue_frac = read_slide(version, slide_id, features_dir)
        n = len(slide_features)
        features[row:row + n] = slide_features
        index_rows.append((slide_id, row, row + n, n))
        tile_parts.append(pd.DataFrame({
            "slide_id": slide_id,
            "tile": np.arange(n, dtype=np.int32),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "tissue_frac": tissue_frac,
        }))
        row += n
    if row != total:
        raise RuntimeError(f"wrote {row} rows, expected {total}")
    features.flush()
    del features

    index = pd.DataFrame(index_rows, columns=["slide_id", "start", "end", "n_tiles"])
    index = index.astype({"start": "int64", "end": "int64", "n_tiles": "int64"})
    tiles = pd.concat(tile_parts, ignore_index=True)
    index.to_parquet(paths[INDEX_FILE] + PART_SUFFIX, index=False)
    tiles.to_parquet(paths[TILES_FILE] + PART_SUFFIX, index=False)
    for path in paths.values():
        os.replace(path + PART_SUFFIX, path)

    manifest = read_manifest(version, features_dir)
    manifest["packed"] = {
        "slides": len(slide_ids),
        "tiles": total,
        "excluded_slides": dict(sorted(excluded.items())),
        "files": {"features": FEATURES_FILE, "index": INDEX_FILE, "tiles": TILES_FILE},
        "row_order": "slides sorted by slide_id; within a slide, the order of its .pt file",
        "features_sha256": file_sha256(paths[FEATURES_FILE]),
        "created": started.isoformat(timespec="seconds"),
        "code_commit": code_commit(),
    }
    write_manifest(version, manifest, features_dir)

    return {
        "slides": len(slide_ids),
        "tiles": total,
        "excluded": dict(excluded),
        "bytes": {name: os.path.getsize(path) for name, path in paths.items()},
        "seconds": (datetime.datetime.now(datetime.UTC) - started).total_seconds(),
        "code_commit": manifest["packed"]["code_commit"],
    }


def open_packed(version, features_dir=FEATURES_DIR):
    """A packed set, ready to read: Packed(features, index, starts, ends, tiles, manifest).

    features is features.npy memory-mapped read-only: opening it reads nothing,
    and slide i's features are features[starts[i]:ends[i]]. The index, the tile
    table and the manifest are small and read whole. Nothing is cached between
    calls, so each DataLoader worker can call this for itself.
    """
    folder = set_dir(version, features_dir)
    features = np.load(os.path.join(folder, FEATURES_FILE), mmap_mode="r")
    index = pd.read_parquet(os.path.join(folder, INDEX_FILE))
    tiles = pd.read_parquet(os.path.join(folder, TILES_FILE))
    return Packed(
        features=features,
        index=index,
        starts=index["start"].to_numpy(dtype=np.int64),
        ends=index["end"].to_numpy(dtype=np.int64),
        tiles=tiles,
        manifest=read_manifest(version, features_dir),
    )


def verify(version, expected_ids, features_dir=FEATURES_DIR, round_trip=True, chunk_rows=65536, progress=None):
    """Check a packed set. Returns the report (one line per check); raises VerificationError if any check fails.

    Every check runs, every failure is collected, then one error names them all:
    1. slides       index slide IDs unique and sorted; with the manifest's
                    excluded slides they are exactly the expected slides
    2. no empty     every n_tiles is at least 1; every excluded slide's .pt
                    file has zero tiles
    3. contiguity   first start 0, each start the previous end, last end the
                    array's row count, n_tiles = end - start
    4. array        shape [manifest tile count, FEATURE_DIM], dtype as the
                    manifest says (float16)
    5. tile table   one row per array row; within each slide's rows, its
                    slide ID and tile numbers 0 to n - 1; x and y int32
    6. finite       no NaN or infinity, scanned chunk_rows rows at a time
    7. manifest     masks_used exactly false, version right, packed counts
                    equal the index and the array
    8. fingerprint  sha256 of features.npy equals the manifest's
    9. round trip   every packed slide's .pt file equals its packed rows
                    exactly: features, coords, tissue fractions (if round_trip)
    """
    packed = open_packed(version, features_dir)
    features, index, starts, ends, tiles, manifest = packed
    section = manifest.get("packed", {})
    excluded = section.get("excluded_slides", {})
    ids = index["slide_id"].to_numpy()
    rows = len(features)

    def check_slides():
        problems = []
        if index["slide_id"].duplicated().any():
            problems.append(f"repeated: {_short_list(set(ids[index['slide_id'].duplicated()]))}")
        if not index["slide_id"].is_monotonic_increasing:
            problems.append("index not sorted by slide_id")
        listed, expected = set(ids) | set(excluded), set(expected_ids)
        if set(ids) & set(excluded):
            problems.append(f"both packed and excluded: {_short_list(set(ids) & set(excluded))}")
        if expected - listed:
            problems.append(f"{len(expected - listed)} missing: {_short_list(expected - listed)}")
        if listed - expected:
            problems.append(f"{len(listed - expected)} not expected: {_short_list(listed - expected)}")
        require(not problems, "; ".join(problems))
        return f"{len(ids)} packed + {len(excluded)} excluded = {len(expected)} expected"

    def check_no_empty():
        empty = ids[index["n_tiles"].to_numpy() < 1]
        require(len(empty) == 0, f"packed slides with no tiles: {_short_list(empty)}")
        with_tiles = [s for s in excluded if len(read_slide(version, s, features_dir)[0]) > 0]
        require(not with_tiles, f"excluded slides that have tiles: {_short_list(with_tiles)}")
        return f"smallest slide has {index['n_tiles'].min()} tiles; {len(excluded)} excluded with zero"

    def check_contiguity():
        problems = []
        if len(starts) and starts[0] != 0:
            problems.append(f"first start is {starts[0]}")
        breaks = np.flatnonzero(starts[1:] != ends[:-1]) + 1
        if len(breaks):
            problems.append(f"{len(breaks)} gaps or overlaps, first at {ids[breaks[0]]}")
        if len(ends) and ends[-1] != rows:
            problems.append(f"last end {ends[-1]}, array has {rows} rows")
        wrong = np.flatnonzero(index["n_tiles"].to_numpy() != ends - starts)
        if len(wrong):
            problems.append(f"{len(wrong)} slides with n_tiles != end - start, first {ids[wrong[0]]}")
        require(not problems, "; ".join(problems))
        return f"{len(ids)} ranges, rows 0 to {rows}, no gaps or overlaps"

    def check_array():
        want = (section.get("tiles"), FEATURE_DIM)
        require(features.shape == want, f"shape {features.shape}, manifest says {want}")
        require(str(features.dtype) == manifest.get("dtype"), f"dtype {features.dtype}, manifest {manifest.get('dtype')}")
        return f"shape {features.shape}, {features.dtype}"

    def check_tile_table():
        require(len(tiles) == rows, f"{len(tiles)} rows, array has {rows}")
        require(tiles["x"].dtype == np.int32 and tiles["y"].dtype == np.int32,
                f"x {tiles['x'].dtype}, y {tiles['y'].dtype}, not int32")
        require(tiles["tissue_frac"].dtype == np.float32, f"tissue_frac {tiles['tissue_frac'].dtype}, not float32")
        owner = np.full(rows, -1, dtype=np.int64)
        for i, (start, end) in enumerate(zip(starts, ends)):
            owner[max(start, 0):max(end, 0)] = i
        require((owner >= 0).all(), f"{(owner < 0).sum()} rows belong to no slide")
        tile_ids, tile_numbers = tiles["slide_id"].to_numpy(), tiles["tile"].to_numpy()
        wrong_slide = np.flatnonzero(tile_ids != ids[owner])
        wrong_number = np.flatnonzero(tile_numbers != np.arange(rows) - starts[owner])
        require(len(wrong_slide) == 0, f"{len(wrong_slide)} rows name the wrong slide, first row {wrong_slide[:1]}")
        require(len(wrong_number) == 0, f"{len(wrong_number)} rows have the wrong tile number, first {wrong_number[:1]}")
        return f"{len(tiles)} rows, each in its slide's range, tiles numbered from 0"

    def check_finite():
        bad = 0
        for start in range(0, rows, chunk_rows):
            bad += int((~np.isfinite(features[start:start + chunk_rows])).sum())
        require(bad == 0, f"{bad} non-finite values")
        return f"all {rows * FEATURE_DIM:,} values finite"

    def check_manifest():
        problems = []
        if manifest.get("masks_used") is not False:
            problems.append(f"masks_used is {manifest.get('masks_used')!r}, not false")
        if manifest.get("version") != version:
            problems.append(f"version {manifest.get('version')!r}")
        if section.get("slides") != len(index):
            problems.append(f"packed slides {section.get('slides')}, index has {len(index)}")
        if section.get("tiles") != rows:
            problems.append(f"packed tiles {section.get('tiles')}, array has {rows}")
        require(not problems, "; ".join(problems))
        return f"masks_used false; {section['slides']} slides, {section['tiles']} tiles"

    def check_fingerprint():
        found = file_sha256(os.path.join(set_dir(version, features_dir), FEATURES_FILE))
        require(found == section.get("features_sha256"), f"sha256 {found[:12]}..., manifest {section.get('features_sha256')}")
        return f"sha256 {found[:12]}... matches"

    def check_round_trip():
        wrong = []
        x, y, fraction = (tiles[c].to_numpy() for c in ("x", "y", "tissue_frac"))
        loop = progress(range(len(ids))) if progress else range(len(ids))
        for i in loop:
            slide_features, coords, tissue_frac = read_slide(version, ids[i], features_dir)
            rows_i = slice(starts[i], ends[i])
            same = (
                np.array_equal(features[rows_i], slide_features)
                and np.array_equal(x[rows_i], coords[:, 0])
                and np.array_equal(y[rows_i], coords[:, 1])
                and np.array_equal(fraction[rows_i], tissue_frac)
            )
            if not same:
                wrong.append(ids[i])
        require(not wrong, f"{len(wrong)} slides differ from their .pt file: {_short_list(wrong)}")
        return f"all {len(ids)} slides identical to their .pt files"

    checks = [
        ("slides", check_slides), ("no empty", check_no_empty), ("contiguity", check_contiguity),
        ("array", check_array), ("tile table", check_tile_table), ("finite", check_finite),
        ("manifest", check_manifest), ("fingerprint", check_fingerprint),
    ]
    if round_trip:
        checks.append(("round trip", check_round_trip))

    report, failed = [], []
    for name, check in checks:
        try:
            report.append(f"ok    {name}: {check()}")
        except Exception as error:  # noqa: BLE001 -- every failure is collected and raised together below
            failed.append(name)
            report.append(f"FAIL  {name}: {type(error).__name__}: {error}")
    if failed:
        raise VerificationError(f"{version}: {len(failed)} checks failed\n" + "\n".join(report), failed)
    return report
