# Findings

Facts we have established by measurement. If a question here is unanswered, we
do not know the answer — say so rather than guessing.

Each finding records the date and the notebook it came from. Questions move up
into the answered section when they are actually measured, not when they are
assumed.

## Answered

### Stage 1 — inventory

Measured 2026-09-08, from `notebooks/01_inventory.ipynb`, saved as
`data/derived/slide_inventory.parquet` (10,616 rows).

**Every slide opens, and every file is built the same way inside.** All 10,616
images in `train.csv` exist on disk and open with OpenSlide. All 10,616 have
exactly 3 pyramid levels with downsample factors 1, 4 and 16. Nothing in the
dataset is a special case at the file level.

**The two hospitals scan at the same physical resolution.** This contradicts
the widely repeated claim that Radboud is around 0.24 microns per pixel and
Karolinska around 0.48 — a factor of two. In our copy of the data they are the
same scale:

| Provider | Slides | Microns per pixel |
|---|---|---|
| Radboud | 5,160 | 0.4862, one value for every slide |
| Karolinska | 5,456 | 0.4520 on 3,263 slides, 0.5032 on 2,193 |

Pixels are square everywhere — the x and y values agree on every slide. The
practical consequence is that a tile of a fixed pixel size covers the same
amount of real tissue on both hospitals' slides, so tile size can be chosen in
pixels for now. Whatever drives the Radboud-to-Karolinska collapse, it is not
magnification.

**Karolinska slides are physically much larger.** Median level-0 area is 714
megapixels for Karolinska against 161 for Radboud, roughly four and a half
times. Any per-slide quantity has to be compared within provider, not across.

**A typical slide is mostly blank glass, and Karolinska more so.** Measured as
the fraction of pixels in the brightest eighth of the grayscale range, read at
the smallest pyramid level:

| Provider | Median blank | 90th pct | 99th pct |
|---|---|---|---|
| Karolinska | 95.8% | 97.0% | 97.9% |
| Radboud | 85.0% | 91.5% | 94.9% |

Seven slides are more than 99% blank. One, `3790f55cad63053e956fb73027179707`
(Karolinska, ISUP 0), measures exactly 100.0% and may be an empty or failed
scan — look at it in Stage 2 before assuming it is tissue.

This number is a rough ranking only. It is not a tissue detector and is never
used to place a tile.

**All 100 slides without a label mask are Radboud.** None are Karolinska. That
is 1.94% of Radboud, and it is systematic rather than scattered, so it is not
simply a dataset assembly slip. By grade they skew low: 19 at ISUP 0, 50 at 1,
2 at 2, 16 at 3, 4 at 4, 9 at 5. Consequence for Stage 5: masks can validate
the tissue detector on every Karolinska slide and on 98% of Radboud, which is
ample. Consequence for any future auxiliary mask task: the missing rows are all
from one hospital, so the auxiliary loss must be masked per slide rather than
assumed present.

**One row has a grade that contradicts its Gleason score.**
`b0a92a74cb53899311acc30b7405e101` (Karolinska) is recorded as Gleason 4+3 with
ISUP grade 2. Gleason 4+3 maps to ISUP 3, and ISUP 2 corresponds to 3+4, so
either field could be the error and we cannot tell which without a pathologist.
Decision: leave `train.csv` untouched and keep the row in training. One row in
10,616 cannot move quadratic weighted kappa.

Read this as evidence that the CSV was entered carefully, not that the labels
are 99.99% correct. The check only finds rows that contradict themselves. A
slide graded 3 that other pathologists would call 4 is invisible to it and far
more common.

**The two hospitals write "no cancer" differently.** Radboud writes `negative`
on 967 rows; Karolinska writes `0+0` on 1,925. Normalise one to the other
before any comparison, or every clean slide looks like a disagreement. A small
sign that the two labelling pipelines never met.

## Open questions

**Stage 2 — looking at the data**

- Do pen marks exist in this dataset, and on roughly how many slides?
- Is `3790f55cad63053e956fb73027179707` genuinely blank, and what is wrong with
  the other six slides above 99% blank?
- Why do only Radboud slides lack label masks?

**Stage 3 — duplicates**

- How many duplicate groups are there, and how big is the largest?

**Stage 6 — tile placement**

- What fraction of tiles overhang the edge of the slide?

**Later**

- Does `test.csv` contain a `data_provider` column?
- Seconds per slide for tiling plus encoding, on our hardware?
- How many slides are in the hidden test set?
