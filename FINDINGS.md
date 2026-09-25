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


### Stage 4 — folds

Measured 2026-09-09, from `notebooks/04_folds.ipynb`, saved as `data/folds.csv`
(10,616 rows, committed to git).

**Five folds, grouped by duplicate cluster and stratified on grade and hospital
together.** `StratifiedGroupKFold` with seed 0. Three seeds were tried; the
metric for choosing was the largest drift of any grade or hospital share in any
fold away from the dataset as a whole. Seed 0 gave 0.0005, seeds 1 and 2 gave
0.0006 — so the choice barely mattered, which is itself worth knowing.

**The balance is better than expected.** Worst drift of five hundredths of a
percentage point:

| fold | ISUP 0 | 1 | 2 | 3 | 4 | 5 | Karolinska | Radboud |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.2723 | 0.2515 | 0.1267 | 0.1173 | 0.1173 | 0.1149 | 0.5144 | 0.4856 |
| 1 | 0.2727 | 0.2511 | 0.1262 | 0.1168 | 0.1178 | 0.1154 | 0.5134 | 0.4866 |
| 2 | 0.2723 | 0.2511 | 0.1267 | 0.1168 | 0.1178 | 0.1154 | 0.5144 | 0.4856 |
| 3 | 0.2721 | 0.2514 | 0.1266 | 0.1168 | 0.1177 | 0.1153 | 0.5137 | 0.4863 |
| 4 | 0.2727 | 0.2506 | 0.1262 | 0.1173 | 0.1178 | 0.1154 | 0.5139 | 0.4861 |
| overall | 0.2724 | 0.2511 | 0.1265 | 0.1170 | 0.1177 | 0.1153 | 0.5139 | 0.4861 |

Fold sizes are 2,123, 2,123, 2,123, 2,124, 2,123.

The handbook warned that grouping would fight stratification hardest on
Radboud, since duplicates are ten times more common there. It did not bind:
duplicate groups top out at 4 slides, which leaves the splitter plenty of room.
Worth recording so nobody re-litigates it.

**No duplicate group straddles a fold.** Asserted, not assumed. This is the
check the whole stage exists for.

**Pen-marked slides are spread across folds:** 66, 62, 64, 61, 64 against 63
expected if perfectly even. Stage 12 can report a score on them separately.

**The file is never regenerated.** The notebook refuses to overwrite an
existing `data/folds.csv` — re-running reports whether the file on disk matches
what the run produced and leaves it alone. Replacing it takes a deliberate
manual delete. Every result recorded from here on is measured against this file.

### Stage 5 — tissue detection

Measured 2026-09-16, from `notebooks/05_tissue_detection.ipynb`.

**The detector.** Convert to HSV, keep pixels whose saturation exceeds a fixed
cut, subtract pen ink (saturation above 60 with hue in the green-to-blue window
30-135, where stained tissue never sits), clean up with morphological opening
then closing, and drop connected blobs under 40 pixels at the smallest pyramid
level. It reads the image and nothing else — no masks, no CSVs, no hospital.

**Settings chosen: a fixed saturation cut of 15**, minimum component area 40
pixels at the smallest level. Not the per-slide automatic alternative.

**Why fixed rather than per-slide Otsu.** Otsu picks a cut from each slide's own
histogram, clamped here to 20-60. It lost on every measure, on both hospitals:

| | mean cancer recall | std | worst slide |
|---|---|---|---|
| Karolinska, fixed cut 30 | 0.904 | 0.058 | 0.721 |
| Karolinska, Otsu clamped | 0.863 | 0.065 | 0.570 |
| Radboud, fixed cut 30 | 0.961 | 0.038 | 0.794 |
| Radboud, Otsu clamped | 0.934 | 0.057 | 0.753 |

Precision was effectively tied. The worst-slide column decided it: on its worst
Karolinska slide Otsu loses 43% of the annotated cancer against the fixed cut's
28%. Since missed tissue is the expensive error, the tail matters more than the
average. A fixed number is also one value to record in a manifest rather than a
procedure whose output we cannot predict on a slide we have not seen.

**Why 15 and not 30.** 30 was an arbitrary starting point. Sweeping the cut
against annotated cancer recall, on 80 slides per hospital that have masks:

| cut | Karolinska mean / worst | Radboud mean / worst | tissue kept, mm² K / R |
|---|---|---|---|
| 10 | 0.954 / 0.867 | 0.993 / 0.945 | 6.29 / 5.67 |
| 15 | 0.934 / 0.832 | 0.986 / 0.914 | 6.13 / 5.51 |
| 20 | 0.918 / 0.800 | 0.977 / 0.873 | 6.04 / 5.39 |
| 25 | 0.903 / 0.761 | 0.964 / 0.808 | 5.95 / 5.29 |
| 30 | 0.885 / 0.667 | 0.941 / 0.722 | 5.87 / 5.16 |
| 40 | 0.838 / 0.309 | 0.841 / 0.166 | 5.64 / 4.49 |

Lower is monotonically better for recall and the cost is small: going from 30 to
15 gains about 5 points of mean cancer recall on Karolinska and 4.5 on Radboud,
and lifts the worst slide from 0.667 to 0.832, while keeping only 4-7% more
tissue by area. If a low cut were sweeping in blank glass the area would
balloon; it creeps, which is the tell that the extra is real tissue.

Confirmed by looking, which is what actually settled it: at cuts of 5, 10, 15
and 20 on the faintest and a typical slide from each hospital, the extra kept at
low cuts is the pale ends of real fragments. The grey halo visible around
Radboud tissue is *not* picked up even at a cut of 5. 15 rather than 10 leaves a
margin against a paler slide than anything in the sample.

**Scored against the label masks at the chosen setting**, 100 slides per
hospital. Masks are used here to grade the detector and nowhere else; the
detector never sees them.

| Hospital | Recall, all labels | Recall, cancer | Precision | Median tissue |
|---|---|---|---|---|
| Karolinska | 0.918 | 0.944 | 0.917 | 5.99 mm² |
| Radboud | 0.787 | 0.991 | 0.998 | 5.42 mm² |

Cancer recall spread: Karolinska mean 0.931, 5th percentile 0.854, worst 0.832;
Radboud mean 0.986, 5th percentile 0.947, worst 0.914.

**Read the cancer column, not the overall one.** The masks label stroma — the
pale connective tissue between glands — as tissue, and stroma carries very
little colour, so a saturation cut misses much of it. That holds Radboud's
overall recall to 0.79 while its cancer recall is 0.99. Only the second bears on
the grade. Reporting the combined number alone would argue for dropping the
threshold further and keeping background for no benefit.

**Tissue coverage is comparable across hospitals; share-of-frame is not.**
Measured over 200 slides per hospital, the share of the frame kept looks like a
detector heavily biased against Karolinska — median 0.039 against 0.131. It is
not. Karolinska frames are 3.9 times larger, and in physical units the detector
finds slightly *more* tissue there: median 5.94 mm² against 5.26, a ratio of
1.13, with heavily overlapping distributions.

Any per-slide quantity that is a ratio with slide size in its denominator will
separate the two hospitals cleanly and mean nothing. The notebook now plots both
and the thin-slide check uses mm² rather than percent — which changes the answer:
under 1% of frame flagged 1 slide of 400, all Karolinska, while under 1 mm² of
tissue flags 4, evenly split between the hospitals. The percentage version was
hiding genuinely thin Radboud slides.

**`3790f55cad63053e956fb73027179707` has no detectable tissue.** The Karolinska
slide that measured exactly 100% blank in Stage 2 returns 0.0% of frame at every
saturation cut from 5 to 30. It is not a threshold problem. Either an empty scan
or tissue below the 40-pixel component floor; worth one look at full resolution
before Stage 6 meets it.

**A smaller asymmetry that does survive.** Cancer recall is 0.944 on Karolinska
against 0.991 on Radboud. Precision runs the other way, 0.917 against 0.998, so
on Karolinska the detector keeps *more* unannotated material rather than less —
which is not the signature of being too strict. This may be a real gap or it may
reflect the two hospitals' annotation protocols, since Karolinska masks carry 3
classes and Radboud's carry 6. Unresolved.

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

**Stage 5 — tissue detection**

- Is the Karolinska/Radboud cancer-recall gap (0.944 against 0.991) a real
  detector weakness, or an artefact of the two annotation protocols?
- The threshold sweep was still improving at a cut of 10 and was not tested
  below it, so the floor is unknown. Does `MIN_COMPONENT_AREA = 40` still suit a
  cut of 15, given the two were tuned against each other at 30?
- Is `3790f55cad63053e956fb73027179707` an empty scan, or tissue too small to
  clear the component floor?

**Stage 6 — tile placement**

- What fraction of tiles overhang the edge of the slide?

**Later**

- Does `test.csv` contain a `data_provider` column?
- Seconds per slide for tiling plus encoding, on our hardware?
- How many slides are in the hidden test set?
