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

### Stage 2 — visual survey

Measured 2026-09-08, from `notebooks/02_visual_survey.ipynb`, using the
thumbnail cache at `data/derived/cache/thumbnails_256.npy` (10,616 slides, one padded
256 x 256 square each, 2.09 GB, built in about eight minutes with no failures).

**The two hospitals look clearly different, and the difference is in
saturation.** Measured over the non-blank pixels of every thumbnail:

| Provider | Median saturation | Median brightness |
|---|---|---|
| Karolinska | 88.3 | 193.6 |
| Radboud | 64.6 | 211.8 |

Radboud slides are paler and brighter; Karolinska's stain is noticeably
stronger. A saturation threshold tuned on one hospital will not transfer to the
other unchanged, which is the thing to carry into Stage 5.

**Pen marks exist, and they are a Radboud phenomenon.** Ranking all 10,616
slides by the fraction of coloured pixels sitting in the green-to-blue part of
the hue wheel, where stained tissue never sits, puts pen at the top of the list
with no false positives visible in the first twenty. Walking down the ranking:

| Rank | Ink score | What the slides look like |
|---|---|---|
| 0-100 | above 0.144 | heavy blue and green strokes across the tissue |
| 300 | 0.063 | clear ink, smaller marks |
| 600 | 0.018 | small ink dots and dashes, mostly at the biopsy tips |
| 900 | 0.003 | no visible ink |

Of the top 600 by ink score, 599 are Radboud and 1 is Karolinska. 611 Radboud
slides, 11.8% of that hospital, score above 0.017.

There is no clean edge in the ranking — the ink shrinks continuously from
strokes across the tissue, to coloured caps on the ends, to specks at the tips.
We set the cutoff at an ink score of **0.06**, which is the point above which
every slide still carries a visible coloured mark. Below it the marks are a few
pixels at the biopsy tip and would touch at most one or two tiles.

At that cutoff: **317 slides, 3.0% of the dataset, every one of them Radboud**
(6.1% of that hospital). No Karolinska slide qualifies.

**Pen is not evenly spread across grades, which makes it a leakage risk.**
Within the pen slides, ISUP 1 is 26.2% against 16.5% of Radboud overall, and
ISUP 0 is 8.2% against 18.7%. A model can therefore learn "ink means not grade
0" as a shortcut. Since test slides carry no pen at all, that shortcut is worth
nothing at submission time and would show up as an unexplained gap between our
score and the leaderboard. Worth revisiting if a model ever scores far better
locally than it does on Kaggle.

Relevant framing: the competition states that test slides are free of pen. So
removing pen is about making training look like test, not about surviving test.

**The emptiest slides are all Karolinska.** Every one of the 20 slides with the
highest blank fraction comes from Karolinska. Nineteen of them show a thin
sliver of real tissue. The twentieth, `3790f55cad63053e956fb73027179707`, shows
nothing at all at thumbnail scale — it is either an empty scan or has tissue too
small to see at 256 pixels. Confirm it at higher zoom in Stage 5 before letting
a tiling routine meet it.

**Grade is invisible at this zoom.** Ten slides from each ISUP grade, 0 through
5, are indistinguishable by eye at thumbnail scale. This is the reason the
pipeline exists: the pattern that defines a Gleason grade lives at cell scale,
far below one thumbnail pixel, so the grade cannot be read off the whole slide
and the image has to be cut into tiles.


### Stage 3 — duplicate slides

Measured 2026-09-09, from `notebooks/03_duplicates.ipynb`. Perceptual hash
(`imagehash.phash`, `hash_size=16`, so 256 bits) computed on each slide's
smallest pyramid level, then every pair compared by Hamming distance.

**We are using a distance threshold of 44.** Two slides whose hashes differ in
44 or fewer of their 256 bits are treated as the same biopsy.

How that number was chosen. Perceptual hashing gives a distance, not a verdict,
so the threshold is the whole stage. Three pieces of evidence were used, and
none of them is the hash itself:

*Grade agreement.* If two slides really are the same biopsy, the pathologist
gave both the same grade. Two slides drawn at random share a grade 19.4% of the
time overall (26.3% within Karolinska, 16.9% within Radboud), so that is the
floor that coincidence alone produces. Agreement among matched pairs stays at
95% or better out to distance 42, is 92.3% at 44, and then falls away — 89.6%
at 46, 82.5% at 50, 72.4% at 54. The grades were never shown to the hashing, so
this is an independent check rather than a circular one.

*Cross-hospital matches.* Two slides from different hospitals cannot be the
same physical biopsy, so any such match is proof of an error. The first one
appears at distance 46. At 44 there are none.

*Group sizes.* Matches are chained: if A matches B and B matches C, all three
form one group. At 44 the largest group is 4 slides. By 52 it is 9 and by 56 it
is 13, which is the signature of a threshold loose enough to string unrelated
slides together.

44 is the largest value that satisfies all three: agreement still far above the
floor, no cross-hospital matches, and no runaway groups.

*Why we lean loose rather than tight.* The two possible errors are not equally
costly. Missing a duplicate puts a slide in training and its twin in validation,
which inflates every score we record and is invisible in the metrics — and
because the fold assignment is never regenerated, it cannot be corrected later
without discarding every result derived from it. Merging two unrelated slides
only means they cannot be split across folds, which costs almost nothing. So
where the evidence was ambiguous we took the looser option.

*A rule we considered and rejected.* "Take the largest threshold with no
cross-hospital groups" would give 45. It is rejected as the selection rule
because it is decided by a single pair — had that one pair not existed, the same
rule would have allowed 49 or beyond, where groups have begun to chain. It also
fires late: slides from one hospital share a scanner and a stain, so nearly all
false matches are within a hospital, and by the time one happens to cross,
roughly one pair in ten is already wrong. Cross-hospital matching is kept as a
veto, never as the selector.

**What the threshold finds:** 285 pairs, forming 273 groups covering 555 slides,
5.2% of the dataset. Group sizes are 265 pairs, 7 triples and 1 group of four.

**Duplicates are overwhelmingly a Radboud phenomenon.** 505 of the 555 grouped
slides are Radboud — 9.8% of that hospital against 0.9% of Karolinska. Every
matched pair is same-hospital.

**The visual check, and what it showed.** Ten pairs spanning distances 40 to 46
were put on screen side by side. Most are convincing: same outline, same grade,
same hospital. But there are visible mistakes at 44 — `1d6450` (ISUP 0) paired
with `db472d` (ISUP 5), and `ae273c` (ISUP 2) with `e0c427` (ISUP 0), both
Karolinska, similar in silhouette but not the same tissue. At 46, `dd5c14`
(ISUP 3) with `e3431a` (ISUP 5) is the same kind of miss.

That is what 92.3% grade agreement looks like from the inside: roughly one pair
in twelve is wrong at this threshold. We are keeping 44 anyway, for the reason
above — a false merge only stops two unrelated slides being split across folds,
which costs nearly nothing, while a missed duplicate inflates every score
permanently. But 44 is at the edge of the usable range, not comfortably inside
it, and that is worth knowing if a fold ever behaves oddly.

Worth recording about method: at thumbnail scale every slide is a thin pink
streak, so the eye is a weaker instrument here than the grade statistic. This is
the one place in the project so far where the numbers are the primary evidence
and the pictures are the sanity check, rather than the other way round.

**Artifact:** `data/derived/duplicate_groups.parquet` — one row per slide, all
10,616, with `slide_id`, `duplicate_group` and `is_duplicate`. Slides with no
duplicate get a group of their own, so Stage 4 can pass `duplicate_group`
straight to the fold splitter without a cleanup step. The file is tracked in
git, because the fold assignment derives from it and is never regenerated.

The distance matrix is cached at
`data/derived/cache/hamming_phash16_smallest_level.npy` (901 MB, not tracked).
The hash settings are in the filename deliberately: change `hash_size` or the
pyramid level and the name no longer matches, so it rebuilds instead of silently
reusing a stale file.


## Open questions

**Stage 2 — looking at the data**

- Is `3790f55cad63053e956fb73027179707` empty, or does it hold tissue too small
  to see at thumbnail scale?
- Why do only Radboud slides lack label masks?
- Does the grade skew among pen slides actually leak into a trained model, or
  is it too small to matter?

**Stage 3 — duplicates**

- Why are duplicates ten times more common in Radboud than Karolinska? Is it a
  scanning practice, or an artefact of how each hospital assembled its cases?

**Stage 6 — tile placement**

- What fraction of tiles overhang the edge of the slide?

**Later**

- Does `test.csv` contain a `data_provider` column?
- Seconds per slide for tiling plus encoding, on our hardware?
- How many slides are in the hidden test set?
