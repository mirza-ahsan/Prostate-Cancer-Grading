# Prostate Cancer Grading — PANDA

Predicting the ISUP grade (0–5) of prostate biopsies from whole-slide images,
using the [PANDA challenge](https://www.kaggle.com/c/prostate-cancer-grade-assessment)
dataset: 10,616 slides, about 400 GB, from two hospitals (Radboud and
Karolinska). The metric is quadratic weighted kappa.

The longer-term aim is a clinical collaboration with King Edward Medical
College, Lahore, so **robustness across hospitals matters more than a
leaderboard score**. Every result is reported for the mixed folds *and* for
Radboud-to-Karolinska transfer.

```
slide.tiff ──> tissue mask ──> tiles ──> UNI2-h (frozen) ──> features ──> attention MIL ──> grade
               └──────────── this repository, so far ────────────┘        └──── later ────┘
```

The tiling reads the slide image and nothing else, so exactly the same code runs
on test slides at submission time. A previous attempt chose tiles using the
label masks, which do not exist for test slides, and so could never be
submitted. Rebuilding that half correctly is what this repository does.

---

## Status

| Stage | Notebook | State |
|---|---|---|
| 1. Inventory | `01_inventory` | done |
| 2. Visual survey | `02_visual_survey` | done |
| 3. Duplicate slides | `03_duplicates` | done |
| 4. Folds | `04_folds` | done — fold file committed |
| 5. Tissue detection | `05_tissue_detection` | done |
| 6. Tile placement | `06_tile_placement` | in progress |
| 7. Encoder (UNI2-h, offline) | — | next |
| 8–12. Modules, pilot, extraction, packing, first submission | — | planned |

The full plan, and the reasoning behind every stage, is in the
[handbook](docs/01-preprocessing_and_feature_extraction_handbook.md).
Everything we have measured is in [`FINDINGS.md`](FINDINGS.md).

---

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). OpenSlide comes as a
Python wheel (`openslide-bin`), so no system library is needed.

```bash
uv sync
```

Download the competition data (accepting the competition rules on Kaggle first)
into `data/`:

```
data/
├── train.csv, test.csv, sample_submission.csv
├── train_images/          10,616 slides, ~347 GB        (not tracked)
├── train_label_masks/     10,516 masks,  ~37 GB         (not tracked)
├── derived/               tables made by the notebooks  (tracked)
│   └── cache/             rebuildable bulk data         (not tracked)
└── folds.csv              the fold assignment           (tracked, never regenerated)
```

From Stage 7 on you need access to [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h),
which is gated: request access on HuggingFace, then log in once with
`.venv/bin/python -m huggingface_hub.commands.huggingface_cli login`. Never
commit or paste the token.

---

## Running the notebooks

Run them in order, from the `notebooks/` folder. Each one reads only what
earlier notebooks wrote, and re-running any of them reproduces its outputs
exactly.

| Notebook | Question | Reads | Writes | Time |
|---|---|---|---|---|
| `01_inventory` | What do we have? | `train.csv`, every slide | `derived/slide_inventory.parquet` | ~8 min |
| `02_visual_survey` | What does it look like? | inventory | `derived/slide_colour_stats.parquet`, thumbnail cache | ~8 min first run, then ~1 min |
| `03_duplicates` | Which slides are near-copies? | inventory, every slide | `derived/duplicate_groups.parquet`, distance cache | ~8 min first run, then seconds |
| `04_folds` | How is every result measured? | the three tables above | `folds.csv`, once | seconds |
| `05_tissue_detection` | Where is the tissue? | inventory, colour stats, masks (scoring only) | nothing: its output is the detector and its settings | ~4 min |

**On Kaggle** the notebooks detect the environment themselves. Competition data
is read from `/kaggle/input/prostate-cancer-grade-assessment/`, and outputs go
to `/kaggle/working/derived/`. Because each Kaggle notebook runs on its own,
attach the output of the earlier notebooks as datasets. Each notebook's
`find_input()` looks for them there, prints where it found each file, and stops
with a clear message if one is missing.

---

## The fold assignment

`data/folds.csv` is the shared ruler for the whole project: five folds, grouped
so that duplicate slides never straddle two folds, and stratified on grade and
hospital together.

**This file is never regenerated.** Changing it makes every previously
recorded result incomparable. `04_folds` refuses to overwrite it, and on Kaggle
it looks for an attached copy before writing anything. The copy committed here
is authoritative.

Worst drift of any grade or hospital share across folds: 0.0005.

| Fold | Slides | ISUP 0 | 1 | 2 | 3 | 4 | 5 | Karolinska | Radboud |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2,123 | 0.2723 | 0.2515 | 0.1267 | 0.1173 | 0.1173 | 0.1149 | 0.5144 | 0.4856 |
| 1 | 2,123 | 0.2727 | 0.2511 | 0.1262 | 0.1168 | 0.1178 | 0.1154 | 0.5134 | 0.4866 |
| 2 | 2,123 | 0.2723 | 0.2511 | 0.1267 | 0.1168 | 0.1178 | 0.1154 | 0.5144 | 0.4856 |
| 3 | 2,124 | 0.2721 | 0.2514 | 0.1266 | 0.1168 | 0.1177 | 0.1153 | 0.5137 | 0.4863 |
| 4 | 2,123 | 0.2727 | 0.2506 | 0.1262 | 0.1173 | 0.1178 | 0.1154 | 0.5139 | 0.4861 |
| all | 10,616 | 0.2724 | 0.2511 | 0.1265 | 0.1170 | 0.1177 | 0.1153 | 0.5139 | 0.4861 |

The 317 pen-marked slides are spread 66 / 62 / 64 / 61 / 64 across the folds.

---

## What we have learned so far

The short version; the evidence and caveats for each point are in
[`FINDINGS.md`](FINDINGS.md).

- **Both hospitals scan at the same resolution** (about 0.45–0.50 µm per pixel),
  contrary to the widely repeated factor-of-two claim. Magnification is not
  what separates them.
- **Karolinska slides are about 4.5× larger.** Any per-slide ratio with slide
  size underneath (percent blank, share of frame) separates the hospitals and
  means nothing; compare in physical units or within a hospital.
- **Stain is what differs:** Radboud slides are markedly paler.
- **Pen marks are on 317 slides (3.0%), all Radboud,** and are more common on
  some grades than others: a possible shortcut, since test slides have no pen.
- **555 slides (5.2%) are near-duplicates,** 505 of them Radboud, grouped by
  perceptual hash at a Hamming distance of 44.
- **The tissue detector** (a fixed saturation cut of 15, with ink excluded)
  finds 95.6% of the annotated cancer on Karolinska and 99.4% on Radboud,
  using the label masks only to score it.

---

## Rules this codebase keeps

1. **Tiling reads the image file and nothing else.** No masks, no CSVs, no
   labels, no hospital. Masks may inform what we learn, never what we look
   at.
2. **The fold file is never regenerated.**
3. **Every feature set gets a version string and a manifest** that states masks
   were not used.
4. **Thresholds are chosen by looking at images**, then checked with numbers.
5. **Coordinates are stored in level-0 units**, and tile order is deterministic.

---

## Licences

UNI2-h is released under CC-BY-NC-ND 4.0: non-commercial use, no derivatives,
no redistribution. Its weights must only ever be shared as a *private* Kaggle
dataset. The PANDA data is used under the competition's terms.
