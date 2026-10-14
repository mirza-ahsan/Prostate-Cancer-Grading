# PANDA — prostate cancer ISUP grading

## Read this first

`docs/01-preprocessing_and_feature_extraction_handbook.md` is the spec for this
phase. Read Parts 0–4 before doing anything technical, and read the current
stage in Part 6 before touching code for it. Do not re-derive things it already
decides.

`FINDINGS.md` holds facts we have established by measurement. Prefer it over
your priors. If a question in it is still unanswered, we do not know the
answer — say so rather than guessing.

`README.md` has the status table, the setup steps, what each notebook reads and
writes, and the fold table.

## What the project is

Predict ISUP grade (integer 0–5) from prostate biopsy whole-slide images.
Metric is quadratic weighted kappa. Data is the Kaggle PANDA dataset: 10,616
slides, ~400 GB, from two hospitals (Radboud and Karolinska) with different
scanners, stains and label pipelines.

Pipeline: slide → tiles → frozen UNI2-h encoder → feature vectors → attention
MIL model → grade. This repo currently covers slide → feature vectors.

A previous attempt chose tile locations by clustering `train_label_masks`,
which do not exist for test slides, so it could not run at inference and was
never submitted. We are rebuilding the preprocessing half so that it can be.
When writing about it (reports, slides), state that limitation factually and
without criticism.

Longer term this goes to a clinical collaboration with King Edward Medical
College, Lahore. So generalisation across sites matters more than a leaderboard
number. Published external validation of the winning PANDA algorithms dropped
from ~0.93 internal to ~0.86 on independent cohorts; expect worse on a
Pakistani cohort. Treat cross-site robustness as the real objective.

## Current state

- Stages 1 to 8 are done, with findings and artifacts recorded.
- Stage in progress: **Stage 9 — pilot, then prove it reproduces.** See
  "Stage 9 so far" below.
- Settled code lives in `src/`. Notebooks 5 onward import from it. Notebooks 1
  to 4 are finished and keep their own path cells; leave them alone.
- There is no notebook 08: Stage 8 was moving code into `src/`.

## Setting up a machine

1. Python 3.12 (`.python-version`) and [uv](https://docs.astral.sh/uv/).
   Run `uv sync`. Torch comes from the PyTorch CUDA 12.6 index (set in
   `pyproject.toml`). OpenSlide comes as a wheel (`openslide-bin`); no system
   library needed.
2. Competition data into `data/`: `train.csv`, `test.csv`,
   `sample_submission.csv`, `train_images/` (~347 GB), `train_label_masks/`
   (~37 GB). None of it is tracked.
3. Tracked derived files already in the repo: `data/derived/*.parquet`
   (inventory, colour stats, duplicate groups) and `data/folds.csv`.
   `data/derived/cache/` is rebuildable and not tracked.
4. UNI2-h is gated on HuggingFace. Request access, then log in once with
   `.venv/bin/python -m huggingface_hub.commands.huggingface_cli login`. The
   token lives in `~/.cache/huggingface/token`, outside the repo. Then run
   notebook 7 once: it saves the weights to `data/weights/uni2h.pt` (fp32
   state_dict, 2.73 GB). `*.pt` and `weights/` are git-ignored.
5. Run notebooks from the `notebooks/` folder. Notebooks 5 to 7 start with a
   bootstrap cell that puts the repo root (or `/kaggle/input/*/src` on Kaggle)
   on `sys.path`; notebook 9's only adds the repo root so far. Scripts run from
   the repo root need `PYTHONPATH=.`.

Current local machine: Linux, NVIDIA RTX A1000 Laptop GPU with 4 GB. UNI2-h in
fp16 takes about 1.4 GB, so a Jupyter kernel holding the model plus a second
process loading it will run out of GPU memory. Check `nvidia-smi` first.

## Kaggle

- `src/paths.py` detects Kaggle by `/kaggle/input`. Competition data is read
  from `/kaggle/input/prostate-cancer-grade-assessment/`; outputs go to
  `/kaggle/working/derived/` (`/kaggle/input` is read-only, so never write
  under `DATA_DIR` there).
- `find_input(name)` looks in `DERIVED_DIR`, then in any attached dataset at
  the top level or under `derived/` or `data/`. It prints where it found the
  file and stops with a clear error if it is missing.
- Attach the repo's `src/` folder as a dataset. Attach the UNI2-h weights as a
  separate **private** dataset. Submission notebooks run with internet off, so
  they use `load_encoder_offline(weights_path())`.
- The PANDA notebook runtime limit is not confirmed; notebook 7 assumes 9 h.

## The code in `src/`

- `paths.py`: `ON_KAGGLE`, `DATA_DIR`, `DERIVED_DIR`, `IMAGE_DIR`, `MASK_DIR`,
  `find_input`, `slide_path(slide_id)`, `weights_path()`.
- `tiling.py`: `get_tiles(slide_path)` returns `(tiles, coords, tissue_frac)`:
  a list of 224×224 PIL RGB tiles, int32 level-0 (x, y) coords, float32
  fractions. It imports only cv2, numpy, openslide and PIL; notebook 6 asserts
  this. Settings: `TILE_SIZE=224`, `READ_LEVEL=1`, `SATURATION_CUT=15`,
  `MIN_CC_AREA=40`, ink = saturation > 60 with hue 30–135, `MIN_FRAC=0.05`,
  `PAD_EDGES=True`, `TILE_PER_SLIDE_CAP=144`.
- `encode.py`: `load_encoder_online()` (HuggingFace, needs login),
  `load_encoder_offline(path)` (timm `vit_giant_patch14_224` + `UNI2H_KWARGS`,
  strict load, `weights_only=True`), `prepare(model)` (eval, fp16 on GPU),
  `encode_tiles(encoder, tiles)` returns a float16 CPU tensor `[n, 1536]`. The
  pixel transform (scale to 0–1, ImageNet mean/std) lives only inside
  `encode_tiles`. There is no resize; a wrong-size tile raises.
- `display.py`: `show_positions` and `show_tiles`, the two pictures for
  checking tiling by eye.

## Decisions already made (evidence in FINDINGS.md)

- Folds: 5 folds, `StratifiedGroupKFold`, seed 0, grouped by duplicate cluster,
  stratified on grade × hospital. `data/folds.csv` is final.
- Duplicates: perceptual hash (phash, `hash_size=16`) at Hamming distance 44.
- Tissue: fixed HSV saturation cut 15 (not Otsu), ink removed, open 1 then
  close 2 iterations (pass `iterations=` by keyword), components ≥ 40 px at the
  smallest level, `connectivity=8` by keyword.
- Tiles: grid of tile-sized cells (~56 mask px), 5% tissue floor, padded edges,
  cap 144, ordered by tissue fraction with ties broken on (row, col), then put
  in reading order. Median ~70 tiles per slide.
- Encoder: UNI2-h, frozen, fp16 on GPU (cosine ≥ 0.99998 against fp32). The
  offline path matches the online path exactly. Tiling plus encoding takes
  about 2.0 s per slide on the A1000: ~34 min per 1,000 slides, ~6 h for the
  training set.
- Microns per pixel are about the same on both hospitals, so tile size is set
  in pixels.

## Stage 9 so far (`notebooks/09_pilot_study.ipynb`)

- Pilot: 2,000 slides from `train_test_split`, stratified on (ISUP grade,
  provider), `random_state=42`.
- Tissue set: `get_tiles` → `encode_tiles` → one `.pt` per slide in
  `TISSUE_EMBED_PATH`.
- Floor set ("exhaustive"): `get_grid_tiles` in the notebook places up to 144
  tiles on an evenly spaced grid over the whole slide, ignoring tissue (~74%
  blank glass), and uses the same `read_tiles` and `encode_tiles`. Saved in
  `EXHAUSTIVE_EMBED_PATH`. It is a floor that the tissue set must clearly beat.
- Known gaps, not yet fixed:
  - the output folders sit under `DATA_DIR`, which is read-only on Kaggle;
    `DERIVED_DIR` works in both places;
  - the tissue loop does not create its folder;
  - only features are saved, not coords. The reproduction test needs coords,
    and the handbook requires a version string and a manifest per feature set.
- Still to do: train a simple model on both sets using `data/folds.csv`,
  record both scores (mixed folds and Radboud→Karolinska), then the saved
  reproduction test on five slides (re-tile and re-encode from the `.tiff`,
  compare coords and features against what is stored).

## How we work

Two people, sitting together, one stage at a time.

- **Stay in the current stage.** Do not scaffold directories, config systems
  or module layouts ahead of need.
- **Default is review.** When we explicitly ask you to write, fix or run
  something, do the whole of it and verify it (run it, compare against the old
  result), but keep to what was asked. Mention anything else and let us decide.
- **Small diffs.** Prefer pointing at the three lines that are wrong over
  rewriting the cell. "Minimal changes" means exactly that.
- **Say when you don't know.** Several facts here are dataset-specific and
  version-specific. If it's checkable in ten lines, give us the check instead
  of an assertion. Prefer research-backed answers.
- **Style.** Brief, plain English. No emojis in markdown files. Code gets
  multi-line docstring-style comments explaining the steps in order, like the
  ones in `src/`. Notebooks must run both locally and on Kaggle.

## Git

- Commit or push only when asked.
- Conventional, one-line messages: `feat:`, `docs:`, `fix:`, `chore:`. Usually
  one file per commit.
- Two people commit from this laptop: `mirza-ahsan` (repo owner) and
  `abdullahahmad0629-glitch` (collaborator). When told to commit under one of
  them, set both author and committer (the emails are in `git log`). Do not add
  a Claude co-author line.
- Pushes go over SSH with the owner's key to
  `git@github.com:mirza-ahsan/Prostate-Cancer-Grading`.

## Security and licences

- Never paste, log or commit a HuggingFace token or any other secret. If one
  appears in chat, do not use it; advise revoking it.
- UNI2-h is CC-BY-NC-ND 4.0: the weights are never committed or shared
  publicly, and reach Kaggle only as a private dataset.

## Tooling

- Lint with ruff (line length 120, config in `pyproject.toml`).
- VS Code uses the Pyrefly language server, not Pylance. `[tool.pyrefly]
  search-path = ["."]` makes `from src...` imports resolve.
- `uv run --with <tool>` has hung on this machine; use a tool already
  installed instead.

## Hard invariants

1. **Tiling code touches nothing but the image file.** No masks, no CSVs, no
   labels, no provider info. If a tiling function needs any of those, it is
   wrong and cannot be submitted. Flag this immediately if you see it creeping
   in — including via an innocent-looking import.
2. **The fold assignment file is never regenerated.** Changing it invalidates
   every previously recorded result.
3. **Every feature set gets a version string and a manifest**, including an
   explicit flag that masks were not used. Every recorded result names the
   version that produced it.
4. **Thresholds are chosen by looking at images**, not by picking plausible
   numbers. If we ask you for a threshold, give us the diagnostic that picks
   it, not the number.
5. **A notebook is for looking at things; a module is for anything run twice.**

Masks may inform *what we learn* (auxiliary training labels, validating a
detector) but never *what we look at* (tile placement). Keep that boundary
visible in the code.

## Traps specific to this codebase

Check for these in every review:

- `read_region` takes its location in **level-0 coordinates always**, whatever
  level you pass. Its size is in that level's pixels. Mixing these produces
  tiles from the wrong part of the slide with no error.
- `read_region` returns **RGBA**. `.convert("RGB")` turns out-of-bounds pixels
  black; composite onto white instead (`rgba_to_rgb_white`).
- OpenSlide handles must be closed in a `finally`, or long loops exhaust file
  descriptors around slide 1,000.
- OpenCV hue is 0–179, not 0–359. Arrays from PIL are RGB; from `cv2.imread`
  they are BGR.
- Morphology functions need uint8, not bool.
- OpenCV optional arguments go by keyword. `cv2.connectedComponentsWithStats(mask, 8)`
  puts 8 into `labels`, not `connectivity`; a positional `iterations` once
  made the closing run once instead of twice.
- Tile ordering must be **deterministic** — break sort ties on position, never
  rely on insertion order. Stage 9's reproduction test depends on it.
- Coordinates are stored in level-0 units. Never anything else.
- The encoder's output dtype may be fp32 despite fp16 (final LayerNorm). Cast
  explicitly; `encode_tiles` already does.
- Preprocessing transform lives in exactly one function. Two copies is the
  failure mode that silently breaks submission.
- `get_tiles` returns a tuple; pass `[0]` (the tiles) to `encode_tiles`.

## Things we do NOT want

- Config frameworks, experiment registries, sweep orchestrators, YAML layers.
- Hyperparameter tuning before the loss function and the folds are correct.
- Suggestions to fine-tune UNI2-h. We costed it: ~14 GPU-hours per epoch at
  144 tiles/slide, versus a ~30 h/week Kaggle quota. Fine-tuning happens with a
  small CNN in a separate branch, much later.
- Optimising the mixed-fold score without also reporting the Radboud→Karolinska
  cross-site score.

## When reviewing a notebook

Work through, in this order:

1. Does anything in the tiling path read labels, masks or CSVs?
2. Coordinate-system errors around `read_region`.
3. Resource leaks — unclosed slides, full-resolution reads of whole slides.
4. Determinism — unseeded randomness, unstable sorts.
5. Silent-failure paths — bare excepts, defaults that mask a missing file.
6. Then, and only then, style and structure.

Prioritise findings by whether they cause **silent** wrongness. A crash is
cheap. A tile read from the wrong coordinates is not.
