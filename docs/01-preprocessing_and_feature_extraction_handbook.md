# PANDA — Preprocessing & Feature Extraction Handbook
### Slides to feature vectors: everything you need to know, and everything you have to build

---

## Part 0 — How to use this document

Read Parts 1–5 once, start to finish, before writing any code. They give you the
background you need to understand *why* the tasks in Part 6 are what they are.

Part 6 is the actual work. It is a checklist. Each task has four sections:

- **Why** — the reason this task exists. If you skip this, you will build the wrong thing.
- **What** — the precise deliverable.
- **How** — code and procedure.
- **Done when** — the test that tells you it's finished.

Do the tasks in order. Several of them are cheap and boring (counting files,
looking at pictures). Do them anyway. Almost every problem in this project so
far came from somebody skipping a boring step.

Parts 7–10 are reference material. You will come back to them.

**Nothing in this document assumes you know any of the vocabulary.** Every term
is explained the first time it appears, and again in the glossary at the end.

---

## Part 1 — The problem, explained from zero

### 1.1 What we are trying to do

We have digital images of prostate biopsies. A biopsy is a thin needle-shaped
sliver of tissue taken from a patient's prostate, mounted on a glass slide,
stained with dye, and scanned by a microscope-scanner into a digital image.

A pathologist looks at that image and assigns an **ISUP grade** — an integer
from 0 to 5.

- **0** = no cancer
- **1** = cancer present, least aggressive
- **5** = cancer present, most aggressive

Our job is to build a program that looks at the image and predicts the ISUP
grade, agreeing with the pathologist as often as possible.

This is the **PANDA challenge** (Prostate cANcer graDe Assessment), a Kaggle
competition. The competition officially ended in 2020, but Kaggle still accepts
"late submissions" and still scores them against the hidden test set. That is
how we will measure ourselves.

### 1.2 Why the images are hard to work with

A normal photo is maybe 4000 × 3000 pixels. These images are **whole slide
images (WSI)** and are typically 20,000 × 20,000 pixels or larger. A single
image is 100–500 MB. There are 10,616 of them in the training set, about 400 GB
total.

You cannot load one into memory as a normal image. You cannot feed one to a
neural network. Everything about this project is a consequence of that one fact.

The standard solution is:

1. Cut the giant image into small squares called **tiles** (or **patches**),
   e.g. 224 × 224 pixels each.
2. Run each tile through a neural network to turn it into a list of numbers (a
   **feature vector**).
3. Feed the whole collection of feature vectors for one slide into a second,
   small model that outputs the grade.

**This handbook covers steps 1 and 2.** Step 3 — the model — comes afterwards,
and is a separate body of work.

### 1.3 The scoring metric: QWK

The competition is scored with **Quadratic Weighted Kappa (QWK)**.

Ordinary accuracy would treat "the true grade was 0 but we said 5" as exactly
as wrong as "the true grade was 0 but we said 1." That's obviously silly here —
grades are ordered, and being close matters.

QWK penalises errors by the **square of the distance**. Predicting 1 when the
truth is 0 costs 1 unit. Predicting 5 when the truth is 0 costs 25 units. It
then compares your total penalty against the penalty a random guesser would
have gotten, and scales the result so that:

- **1.0** = perfect agreement
- **0.0** = no better than random guessing
- **negative** = worse than random

Two things follow from this, and they will come up repeatedly:

1. Big mistakes are catastrophic. One slide predicted 5 when it's really 0
   hurts as much as 25 off-by-one errors.
2. Since grades are ordered, the model should be built to respect that order.
   (That's a modelling decision rather than a preprocessing one, but it
   explains some of the choices you will meet later.)

### 1.4 What "0.94" means

Our stated target is a QWK above 0.94. You should know what that number is.

In the original competition, the top private-test scores ranged roughly from
0.92 to 0.94, and the winning team (PND) scored **0.9408**. Second place scored
0.9377.

So **0.94 is the winning score.** It is not "a good score." Achieving it means
reproducing or beating a first-place solution. This is doable in 2026 because
the image-encoding models available now are much better than what existed in
2020 — but it should shape your expectations. This is a multi-month project
ending in an ensemble of several models, not a weekend of hyperparameter
tuning.

### 1.5 The two data providers — this matters to you specifically

The 10,616 slides come from two different hospitals:

- **Radboud** (Netherlands)
- **Karolinska** (Sweden)

These are not interchangeable. They differ in:

- **Scanner hardware**, so the images have different colour casts and different
  physical resolution (how many micrometres of real tissue one pixel covers).
- **Staining protocol**, so the pinks and purples look different.
- **Annotation protocol**, so the label masks mean different things (see 1.6).
- **Label noise level**, so one provider's grades are less reliable than the
  other's.

Published work shows how severe this is: a model built on frozen UNI features
with a standard attention model scores about 0.888 kappa when trained and
tested on the full mixed dataset, but collapses to about **0.247** when trained
only on Radboud and tested only on Karolinska.

**Practical consequence for you:** every tool you build (tissue detection,
tiling) must be checked separately on Radboud slides and Karolinska slides. A
tissue threshold that works beautifully on one may fail on the other. You will
check this explicitly in Stage 5.

### 1.6 The label masks — and why you must not use them

The dataset includes a folder called `train_label_masks/`. For most training
slides there is a mask image the same shape as the slide, where each pixel has
a small integer code.

The codes mean different things per provider:

**Radboud:** 0 = background/unknown, 1 = stroma (connective tissue), 2 = healthy
(benign) epithelium, 3 = cancerous epithelium Gleason 3, 4 = cancerous
epithelium Gleason 4, 5 = cancerous epithelium Gleason 5.

**Karolinska:** 0 = background/unknown, 1 = benign tissue, 2 = cancerous tissue.

These masks are useful. They are also **the single biggest thing wrong with the
project right now**, because of one fact:

> **The masks exist only for training slides. They do not exist for test
> slides.**

The current feature-extraction code chooses where to cut tiles by clustering
the pixels of the label mask. That means:

- During training, tiles were placed using expert human annotations.
- At test time there are no annotations, so the code silently falls back to a
  completely different method (random sampling + clustering).
- The model would therefore see one kind of tile during training and a
  different kind during scoring.

This is called a **train/test mismatch**. It guarantees that the score we
measure locally will not match the score Kaggle gives us, and it is why the
project currently has no working submission.

**Rule for you: the tiling code you write must open the slide image and
nothing else.** No masks, no CSV files, no labels, no provider information. If
your function needs anything besides the `.tiff` file, it is wrong.

**Nuance — masks are not banned entirely.** Using masks as *extra training
labels* is fine and legitimate (e.g. teaching a model "this tile contains
Gleason 4" as an auxiliary task). That's a training-time-only signal and it
never needs to exist at test time. What's forbidden is using masks to **decide
which pixels to look at**, because that decision has to be repeatable at test
time. Keep this distinction clear: **masks may inform what we learn, never what
we look at.**

---

## Part 2 — Where the project stands right now

A previous attempt produced three notebooks. Here is exactly what they do and
what's wrong with each.

### 2.1 `uni2-feature-extraction-1-4.ipynb`

Opens each slide with OpenSlide, picks up to 64 tile locations by running
k-means clustering on the label-mask pixels, reads 224 × 224 tiles at OpenSlide
level 1, pushes them through the **UNI2-h** model, and saves the result as one
`.pt` file per slide.

Problems:

| # | Problem | Severity |
|---|---|---|
| 1 | Tile placement depends on `train_label_masks` — doesn't exist at test time | **Fatal** |
| 2 | Fallback path (when no mask) does up to 5,120 random `read_region` calls per slide — extremely slow, and non-deterministic because the random seed isn't fixed | High |
| 3 | Mask path centres tiles on a point; fallback path uses the point as a top-left corner. Two different conventions in one function | Medium |
| 4 | `read_region(...).convert("RGB")` turns out-of-bounds pixels **black** instead of white. Tiles at slide edges get black borders that the encoder sees as real content | Medium |
| 5 | Only 64 tiles per slide, and no record of *where* each tile came from | High |
| 6 | No metadata saved at all — no coordinates, no provider, no encoder name, no version | High |
| 7 | No verification step; 47 slides silently have no features | Medium |

### 2.2 `uni2-features-pandas.ipynb`

Copies three people's output folders into one directory. Twelve lines of code,
does what it says.

Problem: no verification. `train.csv` has 10,616 slides. The merged folder has
**10,569**. Forty-seven slides are missing and nobody knows which or why.

### 2.3 `uni-attentionmil-hyperparameter-tuning.ipynb`

The modelling notebook. Trains an attention model on the saved features. Best
result: **QWK 0.9051** on a single random 15% holdout.

That number is not trustworthy for three separate reasons (biased split,
duplicate slides straddling the split, mask-based features). Assume the honest
number is lower. This is not a criticism of the model — it is a measurement
problem, and fixing the measurement is part of the work ahead (Stage 4).

### 2.4 The shape of the fix

```
                 CURRENT                          TARGET
  ┌────────────────────────────┐    ┌────────────────────────────────────┐
  │ slide.tiff + LABEL MASK    │    │ slide.tiff ONLY                    │
  │        ↓ k-means on mask   │    │        ↓ tissue detection (image)  │
  │   64 tiles, no coords      │    │   144 tiles + coords + tissue frac │
  │        ↓ UNI2              │    │        ↓ UNI2                      │
  │   {id}.pt : [N,1536]       │    │   features.npy + index + tiles     │
  │                            │    │   + manifest.json                  │
  │   ✗ cannot run at test     │    │   ✓ identical at train and test    │
  └────────────────────────────┘    └────────────────────────────────────┘
```

---

## Part 3 — What this phase produces

### 3.1 The pipeline and where this handbook sits

```
[1] raw .tiff  →  [2] tiles  →  [3] feature vectors  →  [4] MIL model  →  [5] grade 0–5
    20000×20000     224×224        [N, 1536]              attention          prediction
                    images                                 pooling

    └────────── THIS HANDBOOK ───────────────────┘ └────── COMES LATER ───┘
```

This handbook covers everything from the raw file to the feature vectors. Once
that half exists, model training never opens a `.tiff` again — it only reads
the files produced here.

### 3.2 What "MIL" means (so you understand why coordinates matter)

The model that will sit on top of these features is a **Multiple Instance
Learning (MIL)** model. The idea:

- You give it a **bag** — an unordered collection of feature vectors, one per
  tile, all from the same slide.
- The bag has one label (the slide's ISUP grade). The individual tiles have no
  labels.
- The model learns to pay **attention** to the informative tiles and ignore the
  rest, then pools them into one vector and predicts the grade.

Right now the bag is genuinely unordered — the model has no idea where any tile
came from. That works, but it forecloses several better model families
(TransMIL, graph-based MIL, hierarchical models) that use spatial relationships
between tiles: "these two cancerous regions are adjacent" is real information.

**This is why you must save tile coordinates.** Adding them costs you almost
nothing now. Not having them means a full re-extraction later — days of GPU
time. It is the single most expensive thing to get wrong.

### 3.3 The five things this phase must leave behind

| # | Artifact | Used by |
|---|---|---|
| 1 | Fold assignments | every experiment, forever |
| 2 | Feature set | model training |
| 3 | Tiling code | **inference on unseen slides** |
| 4 | Encoding code | **inference on unseen slides** |
| 5 | Encoder weights available offline | inference on unseen slides |

**Items 3, 4 and 5 are the ones people forget.** Feature files alone let you
train a model but not submit it, because at submission time the tiling and
encoding have to run live on slides nobody has seen. The code is as much a
deliverable as the data — that omission is exactly why the previous attempt
has no working submission.

### 3.4 The two functions everything else depends on

Eventually two functions carry the whole preprocessing pipeline. They get
called in two very different places — once offline to build the training
feature set, and once live inside the submission notebook — and those two call
sites must behave identically. Write them once, in one place, and resist the
urge to inline a "quick version" anywhere else.

Do not create these files today. They emerge in Stage 7, once the notebook
work has settled. This is what they will look like when they do.

```python
# src/tiling.py

def get_tiles(slide_path, tile_size=224, level=1, max_tiles=144,
              min_tissue_frac=0.10):
    """Cut a whole-slide image into tiles, using ONLY the image.

    Args:
        slide_path:      str, path to a .tiff
        tile_size:       int, pixel width/height of each tile
        level:           int, OpenSlide pyramid level to read tiles from
        max_tiles:       int, maximum tiles to return
        min_tissue_frac: float, discard grid cells below this tissue fraction

    Returns:
        tiles:       list of N PIL.Image, each (tile_size, tile_size), RGB,
                     out-of-bounds areas filled WHITE
        coords:      np.ndarray int32 [N, 2], LEVEL-0 (x, y) of each tile's
                     top-left corner
        tissue_frac: np.ndarray float32 [N], fraction of each tile that is tissue

    Guarantees:
        - Uses no masks, no CSVs, no labels, no provider info.
        - Deterministic: same file in => byte-identical output, every time.
        - N may be less than max_tiles. N is never 0 unless the slide is
          unreadable, in which case it returns ([], empty, empty).
    """
```

```python
# src/encode.py

def load_encoder(weights_path, device):
    """Build the UNI2-h model and load weights. Returns a model in eval mode."""

def encode_tiles(encoder, tiles, device, batch_size=32):
    """list of PIL.Image -> torch.Tensor float16 [N, 1536] on CPU.

    The resize/normalise transform lives INSIDE this function. Callers never
    apply their own preprocessing.
    """
```

**Why the transform must live inside `encode_tiles`:** if the resize and
normalisation steps are written once in the training script and again in the
inference notebook, they will eventually differ — someone changes a mean value,
someone uses a different interpolation mode — and the model will silently
degrade at submission time with no error message. One function, one definition,
no drift.

---

## Part 4 — Concepts you need before you start

### 4.1 OpenSlide, pyramid levels, and downsample factors

WSI files are stored as **image pyramids**. The same picture is saved several
times at different resolutions, stacked in one file:

```
level 0:  20000 × 20000   full resolution      downsample 1
level 1:   5000 ×  5000   quarter size         downsample 4
level 2:   1250 ×  1250   sixteenth size       downsample 16
```

The library `openslide` reads these:

```python
import openslide
slide = openslide.OpenSlide(path)

slide.level_count          # e.g. 3
slide.level_dimensions     # e.g. ((20000,20000), (5000,5000), (1250,1250))
slide.level_downsamples    # e.g. (1.0, 4.0, 16.0)  — floats, may be 3.9998
slide.properties           # dict of metadata
slide.close()
```

**Downsample factor** = how much smaller a level is than level 0. Level 1 with
downsample 4 means one pixel at level 1 covers a 4 × 4 block of level-0 pixels.

Reading at level 1 instead of level 0 gives you tiles that cover 4 × more real
tissue for the same number of pixels, and reads roughly 16 × faster. That's why
the existing code uses level 1, and you should keep it.

### 4.2 The `read_region` coordinate trap — read this twice

This is the number one source of silent bugs in WSI code.

```python
region = slide.read_region(location, level, size)
```

- **`location`** is `(x, y)` of the top-left corner, **always in level-0
  coordinates**, no matter what `level` you pass.
- **`level`** is which pyramid level to read from.
- **`size`** is `(width, height)` **in that level's pixels**.

So mixing coordinate systems is easy and produces tiles from the wrong part of
the slide with no error. Concretely:

```python
# WRONG — reads from near the top-left corner of the slide
slide.read_region((500, 500), 1, (224, 224))   # if you meant "500,500 at level 1"

# RIGHT
ds = slide.level_downsamples[1]                # 4.0
slide.read_region((int(500*ds), int(500*ds)), 1, (224, 224))
```

**Rule: store all coordinates in level-0 units, everywhere, always.** Convert
to level-0 the moment you compute a position, and never store anything else.
That's why `get_tiles` returns level-0 coordinates in its contract.

### 4.3 The RGBA / black-border problem

`read_region` returns an **RGBA** image — four channels, the fourth being
**alpha** (opacity). Areas of the requested rectangle that fall outside the
slide come back fully transparent, with RGB values of `(0, 0, 0)`.

If you write `.convert("RGB")`, PIL drops the alpha channel and you are left
with **black**. Tiles at the edge of a slide get black borders that the neural
network sees as dark tissue.

Correct handling — composite onto a white background:

```python
from PIL import Image

def rgba_to_rgb_white(region):
    """region: PIL RGBA from read_region. Returns RGB with transparent -> white."""
    bg = Image.new("RGB", region.size, (255, 255, 255))
    bg.paste(region, mask=region.split()[3])   # channel 3 is alpha
    return bg
```

White is correct because the empty area of a glass slide is white under a
scanner. This is a real bug in the current extraction code. Fix it in yours.

### 4.4 HSV colour space and why it's used for tissue detection

**RGB** describes a colour as amounts of red, green and blue. It's how images
are stored, but it's awkward for "is this pixel tissue?" because tissue can be
light pink or dark purple and those have very different RGB values.

**HSV** describes the same colour three different ways:

- **Hue (H)** — which colour it is, on a wheel. In OpenCV this runs 0–179
  (red ≈ 0, green ≈ 60, blue ≈ 120, magenta ≈ 150).
- **Saturation (S)** — how vivid it is, 0–255. Grey and white have S ≈ 0;
  strong pink has high S.
- **Value (V)** — how bright it is, 0–255. Black is 0, white is 255.

Tissue detection becomes easy in HSV:

- **Background** (empty glass) is white → very low S, very high V.
- **Tissue** (stained pink/purple) → noticeably higher S.

So `(S > threshold) AND (V < threshold)` is a decent tissue detector, and it's
what the existing `tissue_ratio_hsv` function does. You'll build on it.

Conversion in OpenCV — note the channel order:

```python
import cv2
hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)   # input must be RGB, uint8
```

If your array came from OpenCV's `imread` it's **BGR**, not RGB, and you need
`COLOR_BGR2HSV`. Arrays from PIL are RGB. Getting this backwards swaps red and
blue and quietly ruins your thresholds.

### 4.5 Morphological operations — cleaning up a binary mask

After thresholding you get a boolean image that is mostly right but speckly:
isolated true pixels in the background, isolated false pixels inside tissue.

Two operations fix this:

- **Opening** = erode then dilate. Removes small isolated *true* specks.
  Deletes noise in the background.
- **Closing** = dilate then erode. Fills small isolated *false* holes.
  Patches gaps inside tissue.

```python
kernel = np.ones((3, 3), np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel, iterations=1)   # kill specks
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)   # fill holes
```

Input must be `uint8` (0 or 1), not `bool`. Convert with `.astype(np.uint8)`.

### 4.6 Connected components

A **connected component** is a blob of touching true-pixels. In a biopsy image
each separate tissue fragment is one component, and dust specks are tiny
components.

```python
n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, 8)
# n         = number of components, INCLUDING the background (label 0)
# labels    = int array, same shape, each pixel tagged with its component id
# stats[i]  = [x, y, width, height, area] for component i
```

Component 0 is always the background — skip it. Use `stats[i, cv2.CC_STAT_AREA]`
to drop blobs smaller than some pixel count.

### 4.7 Perceptual hashing and duplicate detection

The dataset contains near-duplicate slides — the same tissue scanned twice, or
consecutive sections of the same biopsy. If a duplicate ends up in the training
set and its twin in the validation set, the model has effectively already seen
the answer, and the validation score is inflated.

A **perceptual hash** turns an image into a short bit-string such that visually
similar images produce similar bit-strings. Unlike a normal file hash (MD5), a
one-pixel change does not change the whole hash.

**Hamming distance** = the number of bit positions where two bit-strings
differ. Similar images → small Hamming distance.

```python
import imagehash
h = imagehash.phash(pil_image, hash_size=16)   # 16×16 = 256 bits
distance = h1 - h2                             # imagehash overloads minus = Hamming
```

The first-place team did exactly this, grouping image IDs by image hash before
splitting into folds. You will reproduce it in Stage 3.

### 4.8 Cross-validation, folds, and "grouped" / "stratified"

**Cross-validation (CV):** split the data into 5 equal parts (**folds**). Train
5 models. Model *k* trains on 4 folds and is tested on fold *k*. Every slide
gets predicted exactly once by a model that never saw it. Those pooled
predictions are called **out-of-fold (OOF)** predictions, and QWK computed on
them is the honest score.

**Stratified** means each fold gets roughly the same *proportion* of each
class. Without it, one fold might get twice as many grade-5 slides as another
and its score would be meaningless.

**Grouped** means all members of a group are forced into the *same* fold. We
group by duplicate cluster, so duplicates can never be split across the
train/validation boundary.

We need both at once. `sklearn.model_selection.StratifiedGroupKFold` does it:

```python
from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    ...
```

It cannot always satisfy both perfectly — groups constrain what's possible. You
must **check the result** and confirm the distributions are close enough. Task
Stage 4 tells you how.

### 4.9 fp16, fp32, and why storage size matters

A number stored as `float32` takes 4 bytes; `float16` takes 2. Neural network
features do not need 32-bit precision — 16 is plenty.

Storage for the new feature set:

```
10,616 slides × 144 tiles × 1536 numbers × 2 bytes  ≈  4.7 GB   (float16)
10,616 slides × 144 tiles × 1536 numbers × 4 bytes  ≈  9.4 GB   (float32)
```

Kaggle has limits on notebook output size and dataset size. **Use float16.**
Save with `.half()` before writing, and convert to float32 inside the training
loop (that's one line and costs nothing).

Note: the *current* features are probably float32 even though the code uses
fp16 autocast, because the last layer of the model is a LayerNorm and PyTorch
forces LayerNorm to compute in fp32. Check it in Stage 1 rather than assuming.

### 4.10 Memory-mapped arrays

Ten thousand small files is slow to read. Every file open is a separate request
to the operating system, and a training loop does 10,569 of them per epoch.

A **memory-mapped array** is one large file on disk that the operating system
lets you index like a normal array, loading only the parts you touch:

```python
import numpy as np
arr = np.load("features.npy", mmap_mode="r")   # instant, loads nothing yet
block = arr[1000:1144]                          # only now does it read from disk
```

Opening is instant regardless of file size. This turns epoch time from ~16
seconds to ~2. Across 5 folds × 20 experiments, that's the difference between
a day and an hour.

### 4.11 What UNI2-h is and what it produces

**UNI2-h** is a **foundation model for pathology** — a large neural network
(Vision Transformer, ViT-H/14, roughly 680 million parameters) trained on
enormous quantities of unlabelled pathology images. It has learned to turn a
tissue image into a numeric summary that captures its visual structure.

We use it **frozen**: we never train it, we only run it forward. Feed it a
224 × 224 tile, get back **1536 numbers**. That vector is the tile's "feature
vector" or "embedding".

Why frozen: the alternative — training the encoder end-to-end — costs roughly
4 hours per fold on a good GPU. Frozen features are extracted once and then
you train a small model in minutes. With two students on Kaggle GPU quotas,
that iteration speed is worth more than a marginal accuracy gain.

Useful context: the UNI paper reports quadratic weighted kappa of **0.946** on
PANDA using frozen features and a standard attention model, but an independent
evaluation on a different split got **0.888 ± 0.013** with the same setup. So
frozen features can plausibly reach the target, but don't treat 0.946 as
guaranteed.

**Licence note:** UNI is released under CC-BY-NC-ND — non-commercial, academic
use, no derivatives. Read the licence before uploading weights anywhere. Our
use is non-commercial research, which should be fine, but you should have
actually read it rather than assuming.

---

## Part 5 — Setting up, and how to work

### 5.1 What you have right now

```
data/
pyproject.toml
uv.lock
```

That is the correct amount of structure for today. Resist the urge to scaffold
a framework before you know what the code needs to do. Directories get created
when something needs to go in them, not before.

Add one directory now:

```
notebooks/
```

Everything in Stages 1 through 7 happens in notebooks. A `src/` directory
appears in Stage 8, when the code has stopped changing shape and you know what
belongs in it.

### 5.2 Dependencies

You need, roughly in the order you will reach for them: **openslide-python**
(reading whole-slide images — also needs the system library `openslide`
installed outside Python), **opencv-python-headless** (colour conversion,
morphology, connected components), **numpy**, **pandas**, **pyarrow** (Parquet
files), **matplotlib** (looking at things), **tqdm** (progress bars),
**scikit-learn** (the fold splitter), **ImageHash** (duplicate detection),
**torch** and **timm** (the encoder), **huggingface_hub** (fetching weights).

Add them to `pyproject.toml` as you actually need them rather than all at once.
The system-level openslide install is the one that catches people out — if
`import openslide` fails with a library-not-found error, that's what's missing,
not the Python package.

### 5.3 The one rule about where code lives

**A notebook is for looking at things. A module is for anything you run twice.**

You will violate this at first and that's fine — Stage 8 exists specifically to
clean it up. What matters is that you notice when a cell has become
infrastructure. The signal is simple: the moment you copy a function from one
notebook into another, it belongs in a module.

The previous attempt at this project lost all 65 of its trained model
checkpoints because a notebook cell wrote every one of them to the same
filename. That is what unstructured notebook work costs at scale.

### 5.4 How to run a working session

Since you are sitting together and doing one stage at a time:

- **Read the whole stage before starting it.** The judgment calls section
  usually changes how you'd approach the mechanical steps.
- **One stage per session where possible.** Stages are sized so that each ends
  with something you can point at.
- **Write the finding down before you stand up.** Every stage produces a
  sentence for the findings log. If you can't write the sentence, the stage
  isn't finished.
- **When you disagree, look at the images.** Almost every disagreement in this
  phase is settleable by displaying 20 slides.

### 5.5 Paths and platform

You need the competition data — the `train_images` directory, the
`train_label_masks` directory, and `train.csv`.

Decide early whether you are working locally or on Kaggle, because it changes
what's feasible. Locally you need roughly 400 GB of disk and your own GPU. On
Kaggle the data is already mounted, but sessions time out (historically around
12 hours), there's a weekly GPU quota (historically around 30 hours), and the
output directory has a size cap (historically 20 GB). Those numbers change —
check the current ones before planning Stage 10, which is the only stage that
strains them.

Put the data root in one variable at the top of every notebook. When you build
modules in Stage 8, that variable becomes a single module that everything else
imports. Never hard-code a path twice.

### 5.6 The findings log

Make a file — `FINDINGS.md`, at the repo root — and put these questions in it
now, unanswered:

- Why are 47 slides missing from the previous feature set?
- Median and minimum tiles per slide in the old features?
- Are Radboud and Karolinska at different microns-per-pixel?
- Do pen marks exist in this dataset, and on roughly how many slides?
- How many duplicate groups, and how big is the largest?
- What fraction of tiles overhang the slide edge?
- Does `test.csv` contain a `data_provider` column?
- Seconds per slide for tiling plus encoding, on your hardware?
- How many slides are in the hidden test set?

Each stage below answers one or more of these. The log is not administrative
overhead — it is the difference between a project that accumulates knowledge
and one that rediscovers the same facts in month three.

---

# Part 6 — THE STAGES

Each stage is written the same way:

- **The question** — what you are trying to find out. If you lose the thread
  mid-stage, come back to this line.
- **Where you are** — what is known, what isn't, and why this stage comes now.
- **The work** — the sequence of things to do, in order.
- **The judgment calls** — the decisions only you can make, and how to make
  them. This is where the actual difficulty lives; the work list is mostly
  mechanical.
- **Leave behind** — what exists when you're done, including the sentence for
  the findings log.
- **Traps** — the specific ways this stage goes wrong.

Do them in order. Several are cheap and boring. Do those anyway — nearly every
problem in the previous attempt came from skipping one.

---

## Stage 1 — Take inventory

> **The question:** what do we actually have, and does any of it surprise us?

### Where you are

You have 10,616 rows in `train.csv` and 10,616 image files on disk. You do not
yet know whether every one of them opens, how each file is built inside, how
big they are, or whether the two hospitals scanned at the same magnification.
You also do not know whether the grades in the CSV agree with the Gleason
scores written next to them.

All of that is cheap to find out now, and every later stage leans on at least
one of it. Starting anywhere else means building on guesses.

We are building the feature set from scratch. There is no earlier feature set
to inspect or compare against, so everything from here is measured from the
images themselves.

### The work

1. Load `train.csv`. Confirm the row count and the columns.

2. Build one table with a row per slide holding: the slide id, its grade, its
   Gleason score, its hospital, whether the image file exists, and whether a
   label mask exists. Recording that a mask exists is fine — an inventory is
   not tiling. Nothing that places tiles is ever allowed to read this column.

3. Open every image once and add to the same row: how many resolution levels
   the file contains, the width and height of the largest level, the shrink
   factor of each level, and the physical resolution the file reports (microns
   per pixel, plus the raw resolution tags it came from). If a file will not
   open, write the error into that row and carry on.

4. Add one rough number per slide: read the smallest, most shrunken version of
   the image and record what fraction of it is not near-white. This is **not** a
   tissue detector and it will never be used to place a tile. It is a cheap
   ranking, so that Stage 2 knows which slides are worth looking at first and
   so you can see the thin end of the dataset.

5. Save the table to disk. Every later stage loads this file instead of
   reopening ten thousand images.

6. Read the table back and check the basics: did every slide open, do all the
   files have the same number of levels and the same shrink factors, and how do
   the largest-level sizes compare between the two hospitals.

7. Compare the reported physical resolution between the two hospitals. Look at
   the spread of both, not just an average.

8. Check whether the ISUP grade and the Gleason score agree on every row, using
   the standard mapping between them. Keep the rows that disagree.

Opening 10,616 files takes roughly 20–40 minutes. Step 4 reads real pixels and
is considerably slower — time it on twenty slides and multiply before you start
the full run.

### The judgment calls

**Decide the saved format before the long run, not after.** Some of what the
slide reader hands back is not a plain number. The property list behaves like a
dictionary but stays tied to the open file, and the shrink factors arrive as a
group rather than one value. Pick the handful of values you actually want as
ordinary columns and pull them out as you go. If you drop the live objects into
the table and only think about saving at the end, you will find the table
cannot be written out, and you will run the whole loop a second time.

**The resolution question.** It is widely repeated that Radboud slides are
around 0.24 microns per pixel and Karolinska around 0.48 — a factor of two
apart. If that were true here it would matter: a tile of a fixed pixel size
would cover twice as much real tissue on one hospital's slides as the other's,
and the encoder would effectively be looking at two different zoom levels. But
two slides from this copy of the data, one from each hospital, both came back
near 0.45–0.49. That is a sample of two and may not hold. The point is that the
number is written in these files, so measure it across all 10,616 instead of
repeating the claim. If both hospitals really are at the same scale in our
copy, that removes one suspected cause of the cross-hospital gap, and it is
worth knowing early. Record what you find and move on; nothing acts on it until
much later.

**How thin is too thin.** The bottom of the "not near-white" distribution from
step 4 is the set of slides that will give a tiling routine almost nothing to
work with. If a large share of slides sit far down that distribution, that
limits how aggressive the tissue threshold in Stage 5 can be. Look at the
minimum and the lowest tenth, never the average.

**Gleason and ISUP disagreements.** The ISUP grade can be worked out from the
Gleason score, so any row where the two contradict each other is a labelling
mistake you get for free. Note also that Karolinska writes "negative" where
Radboud writes "0+0" for the same thing, which is a small sign that the two
hospitals' labelling pipelines never met. Whatever this turns up is the first
hard evidence of label noise in the dataset. Do not correct the rows. Count
them, keep the list, and mention it whenever a score looks suspiciously stuck.

### Leave behind

A slide inventory table saved to disk, and answers in the findings log to:
*does every slide open, and are the files built the same way inside?*, *are the
two hospitals at the same physical resolution?*, *how much of a typical slide
is blank, and what does the thin end look like?*, and *how many rows have a
grade that disagrees with their Gleason score?*

### Traps

- Not closing each image after you read it. Handles left open will exhaust the
  operating system's limit part-way through a long loop, and the failure
  message will not point at the cause. Close it in a way that still runs when
  the body of the loop fails.
- Keeping the property list in the table. It holds the file open even though
  you thought you were finished with the slide, which is the quiet version of
  the trap above.
- Letting one unreadable file end a forty-minute loop. Catch it per slide,
  write the error into the row, keep going.
- Building each row as a bare list and adding fewer values on the error path
  than on the success path. The row stops matching the columns and the loop
  stops with it. Use names, not positions.
- Reading the average of anything and stopping there. The minimum and the
  bottom tenth are where the problems live.
- Treating a missing resolution field as "the two hospitals are the same". It
  means you do not know yet.

---

## Stage 2 — Look at the data

> **The question:** what does this dataset actually look like, and what is
> going to break a tissue detector?

### Where you are

You have counts. You have not looked at a single slide. In Stage 5 you will
choose colour thresholds, and every one of those numbers should be chosen
because you saw the images — not because a document suggested a value.

This stage produces no artifact except knowledge and a list. It is still the
stage most worth not skipping.

### The work

1. Write a small function that displays a grid of slide thumbnails, reading the
   smallest pyramid level. You will reuse it constantly.
2. Look at 20 random Radboud slides and 20 random Karolinska slides, side by
   side. The colour difference should be obvious.
3. Look at 10 slides from each ISUP grade, 0 through 5.
4. Look at the 20 slides with the least ink on them, taken from the Stage 1
   table. These are your likely failure cases — find out what is wrong with
   each one.
5. Look at the largest and smallest slides by area.
6. Write down a list titled "visual problems in this dataset", with a rough
   count for each.

### The judgment calls

**What counts as a problem worth listing.** You are looking for: how large the
colour difference between providers is; whether pen marks exist and on roughly
how many slides; whether there are faint or over-stained slides; dust, bubbles,
out-of-focus regions and edge shadows; how many separate tissue fragments a
typical slide has; and how much of the image is empty background.

**The pen mark question specifically.** The competition data description states
that some training images have stray pen marks but that test slides are free of
them. That means pen removal is about making *training* resemble *test*, not
about surviving test — useful framing when you decide in Stage 5 whether to
build a filter at all. If you find almost no pen, skip it entirely.

**Why the thin slides matter most.** A slide that came out nearly blank either
has very little tissue on it, is faintly stained, or is broken in some way.
Each of those implies a different fix. Work out which it is by eye now, because
in Stage 5 these are the cases your tissue detection has to survive.

### Leave behind

A notebook containing the image grids, and a written list of visual problems
with rough counts. An answer in the findings log to: *do pen marks exist, and
on how many slides?*

### Traps

- Looking only at Radboud because it sorted first alphabetically.
- Looking only at random slides. The extremes teach you more.
- Reading level 0 for a thumbnail. Use the smallest level; the full-resolution
  image is gigabytes.
- Deciding this stage is optional because it produces no file.

---

## Stage 3 — Find the duplicate slides

> **The question:** which slides are near-copies of each other, and would
> therefore leak between training and validation?

### Where you are

The dataset contains slides of the same tissue scanned twice, and consecutive
sections of the same biopsy. If one copy lands in training and its twin in
validation, the model has effectively already seen the answer. Validation
scores go up, real performance doesn't, and every experiment comparison after
that is polluted.

The first-place team in the original competition handled this explicitly,
grouping image IDs by image hash before making their split. You are reproducing
that, and you must do it before Stage 4 because the fold split depends on it.

### The work

1. For every slide, read the smallest pyramid level and resize it to a fixed
   square before hashing. Compute a perceptual hash.
2. Compare every hash against every other one and collect the pairs closer than
   some Hamming distance.
3. Merge those pairs transitively into groups using union-find, so that if A
   matches B and B matches C, all three end up in one group.
4. Give every slide a group ID, including slides with no duplicate — those get
   a group of size one. Slides with no readable image still need a group so
   they don't vanish from the fold table.
5. Display the thumbnails of ten groups and look at them.
6. Adjust the threshold and repeat step 5 until the groups look right.

### The judgment calls

**The distance threshold is the whole stage.** Everything else is mechanical.
Too loose and unrelated slides get chained together into one enormous group;
too tight and you catch nothing. Both failures are visible immediately if you
look at the pictures, and invisible if you don't.

Watch the size of the largest group as your alarm. If one group swallows
hundreds of slides, union-find has chained everything transitively and the
threshold is far too loose. A healthy result has a largest group in the single
or low double digits.

**A second, independent check:** true duplicates should usually share an ISUP
grade. If most of your multi-slide groups contain wildly different grades, they
aren't duplicates.

**What this cannot fix.** The dataset's 10,616 slides come from about 2,113
patients — roughly five slides per patient — and **patient ID was never
released**. Perceptual hashing catches the visually near-identical pairs, but
two slides from the same patient that look different will still split across
folds. You cannot fully solve this. What you can do is know it's there, so that
nobody rediscovers it in month three and panics. Your eventual cross-validation
score is very slightly optimistic even after this stage, and that's the best
available.

### Leave behind

A duplicates table mapping every slide to a group. Findings log answers: *how
many groups, how large is the largest, and what threshold did we settle on and
why.*

### Traps

- Hashing without resizing to a fixed square first. Slides have different
  aspect ratios and you would be partly hashing shape rather than content.
- A pairwise comparison loop in pure Python. Ten thousand slides is 56 million
  comparisons — do it as a matrix operation, in chunks, or it takes hours.
- Setting the threshold once and never looking at the result.
- Dropping unreadable slides instead of giving them singleton groups.

---

## Stage 4 — Fix the folds

> **The question:** how will every future number in this project be measured?

### Where you are

The previous attempt used a single random 85/15 split, then tuned 65
hyperparameter trials against it, then reported that same split's best score as
the result. That is optimistic twice over, and it had duplicates on both sides.

This stage produces the most important file in the project. It is the shared
ruler. Every number either of you produces from now until the end is measured
against it, which means it can only be created once.

### The work

1. Merge the duplicate groups from Stage 3 onto the training table. Give any
   slide without a group its own.
2. Build a combined stratification key from grade and provider together, so
   that folds balance on both at once.
3. Split into five folds using a splitter that is simultaneously *grouped* by
   duplicate cluster and *stratified* by that combined key.
4. Assert that no duplicate group appears in more than one fold. If this fails,
   stop — do not proceed with the file.
5. Print grade distribution by fold and provider distribution by fold as
   proportions. Read both tables.
6. Save the file, commit it, and write in the README that it is never
   regenerated.

### The judgment calls

**Reading the balance tables.** Each column should be roughly constant down the
rows. A grade whose share swings from 0.08 in one fold to 0.15 in another means
that fold's score will differ from the others for reasons unrelated to your
model.

**When "good enough" is good enough.** A grouped, stratified splitter cannot
always satisfy both constraints — large duplicate groups genuinely constrain
what is possible. Try two or three random seeds and keep the best. Do not
over-engineer this; roughly even is the target, and chasing perfect balance is
wasted effort.

**Why five folds and not a holdout.** Five folds means every slide gets
predicted exactly once by a model that never trained on it. Pooling those gives
you a prediction for all 10,600 slides, and a score computed on that pool is
your honest number. A single holdout gives you one small sample and an
irresistible temptation to tune against it.

**Building this before any features exist is deliberate.** The fold assignment
depends only on the labels and the duplicate groups, not on features at all. So
it can be settled now, and once it is settled every number recorded from here
on is comparable with every other one. Do it later and you will be tempted to
change it after you have seen a result, which is exactly how a split stops
being honest.

### Leave behind

A committed fold assignment file covering all 10,616 slides, with the two
balance tables pasted into the README. A line in the README stating in bold
that this file is never regenerated.

### Traps

- Regenerating it later "just to fix one thing". Every previously recorded
  result becomes incomparable the moment you do.
- Stratifying on grade but not provider.
- Skipping the duplicate-leak assertion because it "should be fine".
- Filtering to only slides that have features. Include everything; filter at
  read time.

---

## Stage 5 — Detect tissue without masks

> **The question:** can we find the tissue using only the image, as well as the
> expert annotations do?

### Where you are

This is the stage that makes the whole pipeline submittable. The previous
pipeline chose where to look using `train_label_masks`, which do not exist for
test slides. Everything you build from here must work from pixels alone.

It is also the hardest stage, and the one where the two providers will first
genuinely fight you.

### The work

1. Read a slide's smallest pyramid level as an RGB array.
2. Convert to HSV and build a boolean mask from a saturation threshold, using
   the fact that empty glass is nearly white and therefore nearly unsaturated,
   while stained tissue is not.
3. Clean the mask with morphological opening and closing, then drop connected
   components below some pixel area.
4. Build a display function that shows original, mask, and
   tissue-kept-on-white side by side. Build this *before* you start tuning.
5. Tune by looking, on ten slides from each provider separately.
6. Try an automatic per-slide threshold as an alternative to a fixed one, and
   compare.
7. Decide whether pen removal is needed, based on Stage 2's count.
8. Run the detector over a large sample and compare tissue-coverage
   distributions between providers.
9. Score your detector against the label masks on a few hundred slides.

### The judgment calls

**What "good" means here, and it isn't accuracy.** Missed tissue is the
expensive error — a cancerous region you never tiled is a grade you can never
predict. Extra tissue only wastes a few tiles. So when you compare against the
label masks in step 9, optimise for **recall** of the annotated tissue, not for
overlap. A detector that finds 99% of the tissue plus some background is far
better than one that finds 92% cleanly.

**The legitimate use of masks.** Step 9 uses the label masks, and that is fine.
The rule from Part 1.6 is that masks may inform *what we learn*, never *what we
look at*. Tuning a threshold against them is the first; the tuned detector
still runs on pixels alone. Keep this visible in the code — the scoring
function lives in the notebook, never inside the tiling path.

One caveat when you do it: mask value 0 means background *or unknown*, so real
tissue can be labelled 0. If your detector finds more tissue than the mask
does, that is not automatically wrong. Look at those regions before tightening
anything.

**Fixed threshold or per-slide automatic.** A single fixed number is
predictable and easy to reason about, but may not suit both providers. An
automatic per-slide method adapts, which usually generalises better across
sites, but fails badly on slides that are almost entirely background. A safe
compromise is automatic with the result clamped into a sane band. Test both;
pick with evidence.

**Provider-specific settings are a last resort.** Prefer one setting that works
for both. If you genuinely cannot find one, remember that provider-specific
settings require knowing the provider at test time — verify that `test.csv`
carries that column before depending on it.

**Pen removal is dangerous.** Hematoxylin stains cell nuclei blue-purple. A
loosely specified "remove blue" filter deletes exactly the tissue you need.
Only build one if Stage 2 found actual pen; tune it against those specific
slides; and then verify against slides you know are clean that it removes
essentially nothing. If it eats more than a fraction of a percent of a clean
slide, it is too aggressive.

### Leave behind

A working tissue detector in a notebook, plus a 20-slide overlay grid you have
both looked at and agreed on. Findings log answers: *what thresholds, fixed or
adaptive, is pen removal on, and what is median tissue coverage per provider.*

### Traps

- Feeding a BGR array to an RGB-to-HSV conversion. Red and blue swap and every
  threshold becomes nonsense. Arrays from PIL are RGB; arrays from OpenCV's
  file reader are BGR.
- Passing a boolean array to a morphology function that expects 8-bit integers.
- Tuning on one provider only.
- A threshold so tight it excludes faint tissue at fragment edges — the
  expensive error.
- Building the pen filter before confirming there is pen.
- Forgetting that OpenCV squeezes hue into 0–179 rather than 0–359, so every
  hue threshold you read online needs halving.

---

## Stage 6 — Place the tiles

> **The question:** which 144 squares of this slide should the encoder see?

### Where you are

You can find tissue. Now you have to convert that into a specific, ordered,
reproducible list of tile positions.

The previous code clustered randomly sampled points, which was slow (up to
5,120 region reads per slide), non-deterministic (the random seed was never
set), and could place overlapping tiles. You are replacing it with something
simpler that every strong solution in the original competition used: a regular
grid, scored by tissue content, top N kept.

### The work

1. Choose which pyramid level to build the tissue mask on, such that one tile
   spans a reasonable number of mask pixels — enough that the tissue fraction
   of a grid cell is a meaningful number rather than noise.
2. Lay a regular non-overlapping grid over the tissue mask, with cell size
   equal to one tile's footprint expressed in mask pixels.
3. Score every cell by the fraction of it that is tissue. Discard cells below a
   minimum.
4. Sort by tissue fraction descending, keep the top N, then re-sort by position
   so tiles come back in reading order.
5. Convert every kept position to level-0 coordinates.
6. Read the actual tiles at those positions, compositing onto white so that
   out-of-bounds areas are white rather than black.
7. Display the resulting tiles for ten slides from each provider.
8. Display the selected positions drawn on the slide thumbnail, to confirm the
   geometry is right.
9. Assert that calling the function twice on the same slide returns identical
   coordinates.
10. Measure what fraction of selected tiles overhang the slide edge.

### The judgment calls

**Three coordinate systems are live at once** and this is where the stage is
actually hard: the tissue mask sits at one pyramid level, tiles are read at
another, and coordinates must be stored in level-0 units. One tile is
`tile_size` pixels at the reading level, which is `tile_size × downsample`
level-0 pixels, which is that divided by the mask's downsample in mask pixels.
Get this wrong and you get tiles from the wrong part of the slide, with no
error message. Part 4.2 is worth rereading before you start.

**Why 144 tiles rather than 64.** Two reasons. Coverage: 64 tiles of 224 pixels
at level 1 covers roughly 51 million level-0 pixels, a fraction of a
20,000 × 20,000 slide, so you are more likely to miss the small region that
determines the grade. And augmentation: the model will later want to sample a
random 64 of your 144 each epoch, which is free data augmentation and one of
the bigger available wins — but only if there is a surplus to sample from.
Storage at fp16 is about 4.7 GB, which is acceptable.

**Determinism is a requirement, not a nicety.** Sorting by tissue fraction
alone leaves ties broken by insertion order. That happens to be stable here,
but relying on it is fragile — make the tie-breaking explicit by sorting on
position as a secondary key. The reason this matters is Stage 9: you will
verify that features computed live at inference match features computed offline
during training, and any nondeterminism makes that check fail.

**The overhang measurement in step 10 settles an open question.** Grid cells
get clipped to the tissue mask bounds, but tiles are read as a full square from
the cell's top-left corner, so cells at the right and bottom edges can extend
past the slide. Whether that actually happens, and how often, is empirical.
Measure it rather than assuming — then you know whether the white-compositing
step is load-bearing or merely insurance.

**Reading tile counts as feedback.** If most slides find far fewer than your
cap, your minimum tissue fraction is too high or your detector is too tight.
Median tile count close to the cap is what you want.

### Leave behind

A working tile selector, tile grids and position maps you have both looked at,
and a passing determinism assertion. Findings log answers: *median tiles per
slide, and what fraction of tiles overhang the edge.*

### Traps

- Passing level-1 coordinates to a region read that expects level-0 ones.
- A mask level so coarse that a tile spans only a handful of mask pixels —
  guard against it explicitly.
- Slides with fewer pyramid levels than you assumed. Clamp the level.
- Dropping the alpha channel instead of compositing onto white, giving edge
  tiles black borders that the encoder reads as dark tissue.
- Leaving determinism to luck.

---

## Stage 7 — Get the encoder running

> **The question:** can we turn tiles into feature vectors, both online and
> offline?

### Where you are

You have tiles. Now you need the 1536 numbers per tile, and — critically — you
need to be able to produce them **without an internet connection**, because a
Kaggle submission runs with networking disabled.

That second requirement is why this is its own stage. It is trivially easy to
get the online path working, ship everything else, and discover in month three
that the submission notebook cannot load the model.

### The work

1. Build the model with the correct architecture arguments and pull the
   pretrained weights from HuggingFace.
2. Define the resize and normalise transform **once**, inside the encoding
   function. Not in the caller.
3. Run a batch of tiles through and confirm the output shape, dtype, and that
   there are no non-finite values.
4. Force the output to fp16 explicitly rather than relying on what the model
   returns — see Part 4.9 on why the dtype may not be what you expect.
5. Save the weights to a local file and add a second loading path that reads
   from it instead of from the network.
6. Verify that the offline path produces the same outputs as the online path on
   the same tiles.
7. Upload the weights somewhere the submission environment can reach them, and
   test loading them with networking actually switched off.
8. Time the whole thing: seconds per slide for tiling plus encoding.

### The judgment calls

**Why the transform must live in exactly one place.** If preprocessing is
written once in the extraction code and again in the submission notebook, they
will eventually diverge — someone changes a mean value, someone uses a
different interpolation mode — and the model degrades silently with no error.
One function, one definition. This is the single most important structural
decision in the whole preprocessing half.

**Testing offline loading now, not later.** Step 7 feels premature. Do it
anyway. It is a two-hour task in Stage 7 and a project-blocking crisis in
Stage 12.

**The timing number decides your architecture.** Multiply seconds per slide by
the test set size — roughly 1,000 images in the hidden test set — and compare
against the notebook time limit. Leave 40% headroom. If it's tight, you have
three dials before you touch anything structural: fewer tiles at inference only
(training keeps all 144, inference uses the top 96), a larger encoder batch
size, or dropping from UNI2-h to the smaller UNI. Test the accuracy cost of the
third before committing to it.

**Read the licence.** UNI is released under CC-BY-NC-ND — non-commercial, no
derivatives. Academic use is very likely fine, but "no derivatives" is worth
reading carefully before you fine-tune anything or publish, and you should read
the actual text rather than take anyone's summary.

### Leave behind

Working online and offline encoder paths, weights available to the submission
environment, and a verified match between the two. Findings log answers:
*seconds per slide, and the extrapolated total for the test set.*

### Traps

- Assuming the model returns fp16 because you used mixed precision. The final
  layer normalisation is computed in fp32, so the output probably is too.
- Discovering the offline path is broken on submission day.
- Two copies of the preprocessing transform.
- Timing on a warm cache and getting an unrealistically good number.

---

## Stage 8 — Move the code out of the notebook

> **The question:** which parts of this have stopped changing?

### Where you are

Seven stages of notebook work have produced code you now depend on. Some of it
is stable — the tissue detector and tile selector have settled. Some is still
exploratory. This stage separates them.

Doing this *now*, before extraction, is deliberate. Stage 9 needs to import the
same functions twice from two different places, and Stage 12 needs to import
them into a submission notebook. Both are painful if the code lives in cells.

### The work

1. Create a `src/` directory.
2. Move the settled functions into modules: path configuration, tiling
   (including tissue detection), encoding, and the display helpers.
3. Leave the exploratory notebook cells where they are. Not everything needs to
   move.
4. Rewrite the notebooks to import from the modules rather than defining
   functions inline.
5. Re-run the Stage 5 and 6 visual checks through the imported versions, to
   confirm nothing broke in the move.

### The judgment calls

**What moves and what stays.** The test is whether you have run it twice, or
whether you copied it between notebooks. Anything that passes either test
moves. Anything you wrote once to look at something stays.

**Keep the tiling module clean.** It must import nothing that touches labels,
masks, provider information, or any CSV. That constraint is what makes the
whole pipeline submittable, and it's easy to violate accidentally by importing
a convenience function that reads the training table. If you find yourself
wanting to, that's a signal the function belongs elsewhere.

**Do not build a configuration framework.** A module holding paths and
constants is enough. YAML config systems, experiment registries and sweep
orchestrators are solutions to problems you do not yet have.

### Leave behind

A `src/` directory containing the stable pipeline, notebooks that import from
it, and the visual checks re-run and still passing.

### Traps

- Moving everything, including the throwaway exploration.
- Refactoring and changing behaviour in the same commit. If the visual checks
  change, you won't know which of the two caused it.
- Letting the tiling module acquire a dependency on the training labels.

---

## Stage 9 — Pilot, then prove it reproduces

> **The question:** does the new pipeline work, and does inference reproduce
> training exactly?

### Where you are

You are one stage away from spending a day of GPU time on extraction. Do not
spend it before checking that the thing you're about to run 10,616 times is
correct.

There are two separate checks here and they catch different bugs.

### The work

1. Choose about 2,000 slides, stratified by grade and provider.
2. Extract features for those, in the format you intend to use for the full
   run.
3. Train a simple model on the pilot features, using the folds from Stage 4,
   and record the score. Then extract a second pilot set over the same slides
   with the tiles placed on a plain evenly spaced grid, ignoring tissue
   entirely, train the same model on that, and compare the two scores.
4. Separately: pick five slides that already have extracted features. For each,
   run tiling and encoding fresh from the `.tiff`, and compare the result
   against what is stored — both the feature values and the coordinates.
5. Save that comparison as a test you can re-run.

### The judgment calls

**What the pilot comparison is for.** The evenly spaced grid is a deliberately
poor way to choose tiles — it spends most of them on blank glass. It exists as
a floor. If choosing tiles by tissue does not beat that floor clearly, then the
tissue step is not doing its job and you need to find out why before spending a
day of GPU time. You are not chasing a particular number here; you are checking
that the pipeline runs end to end and that each part of it earns its place.

If it does collapse, debug it now, while it costs a day rather than a week.

**The reproduction check is the more important of the two.** It proves that the
features your submission notebook will compute live are the same features the
model was trained on. If the coordinates differ, your tiling isn't
deterministic. If the coordinates match but the features don't, you have two
copies of the preprocessing transform. Both of those are silent failures that
would otherwise surface as an unexplained gap between local scores and
leaderboard scores.

**Re-run it every time.** Any change to tiling, tile count, level, encoder, or
preprocessing invalidates the previous check. Make it cheap enough to run
casually.

### Leave behind

A pilot feature set, the two scores from the comparison, and a saved
reproduction test that passes.

### Traps

- Skipping the pilot because the code "looks right".
- Reading the pilot score as if it were the final score. It comes from a
  fifth of the data.
- Running the reproduction check only once, at the start.
- Comparing against features extracted with different settings and concluding
  the code is broken.

---

## Stage 10 — Run the full extraction

> **The question:** none — this is the one stage that is pure execution.

### Where you are

Everything before this existed to make sure you only do this once. Budget
somewhere between 18 and 38 hours of GPU time depending on your hardware and
tile count.

### The work

1. Split the slide list into parts by a stride, so parallel runs on different
   machines or accounts never collide.
2. Write one file per slide, and skip slides whose file already exists.
3. Overlap the CPU tiling with the GPU encoding using a data loader with worker
   processes.
4. Record which parts are complete as you go.
5. Collect the list of slides that produced zero tiles and investigate them
   individually.

### The judgment calls

**Resumability is not optional.** Sessions die — Kaggle's time out, local
machines crash. Per-slide files with a skip-if-exists check mean a dead session
costs you nothing. Writing directly into one large array cannot be resumed, so
don't; pack into the final format afterwards, in Stage 11.

**Tiling is CPU work and encoding is GPU work.** Run them naively in sequence
and the GPU idles for most of the run. Worker processes preparing the next
slides while the GPU handles the current one is the single biggest speedup
available here, often several times.

**Slides with zero tiles are a finding, not noise.** Look at each one. It is
either a genuinely tiny biopsy, a faintly stained slide your detector missed,
or a corrupt file — and each implies something different.

### Leave behind

A per-slide feature file for every slide in the fold table, or a documented
list of the ones that could not be produced and why.

### Traps

- Random splitting instead of a stride, so two runs duplicate work.
- No resume check, so a session timeout costs the whole run.
- Zero-worker data loading, leaving the GPU idle.
- Silently ignoring empty slides.

---

## Stage 11 — Pack and verify

> **The question:** is this feature set complete, self-describing, and fast to
> read?

### Where you are

You have ten thousand small files. That is slow to read — every file open is a
separate request to the operating system, and a training loop does 10,600 of
them per epoch. And a feature set nobody checked is how slides go missing
without anyone noticing for months.

### The work

1. Count the total tiles across all files so you can pre-allocate.
2. Write one large memory-mappable array, plus an index mapping each slide to
   its row range, plus a table of per-tile coordinates and tissue fractions.
3. Write a manifest recording the encoder, feature dimension, dtype, tile size,
   level, tile cap, tissue method, slide and tile counts, coordinate space,
   date, and code commit.
4. Include in the manifest an explicit flag recording that label masks were not
   used.
5. Write a verification script that checks: every expected slide is present, no
   slide has zero tiles, the index ranges are contiguous with no gaps, the array
   shape matches the index, the dtype is what the manifest claims, the per-tile
   table has the right number of rows, there are no non-finite values, and the
   no-masks flag is set.
6. Run it. Fix anything it catches.

### The judgment calls

**The manifest is not paperwork.** The moment a second feature set exists,
results from the two become indistinguishable without one. Bump the version
string every time tiling, tile count, level or encoder changes, and change the
directory name with it. Keep old versions on disk until every result derived
from them has been superseded.

**The no-masks flag is a one-line proof of submittability.** Making it a
verification failure rather than a comment turns the most expensive mistake in
this project's history into something a script catches.

**Contiguity matters more than it looks.** If the index has gaps or overlaps,
slides silently get each other's features and nothing errors. The assertion is
cheap; the bug is not.

### Leave behind

A packed feature set with index, per-tile table and manifest, and a
verification script that exits clean.

### Traps

- Packing before extraction is complete, producing a partial set that looks
  finished.
- Storing fp32 and doubling your disk usage for no benefit.
- Omitting coordinates. They cost about a quarter of one percent of the
  storage, and not having them means a full re-extraction to get them back.
- Skipping verification because the extraction "looked fine".

---

## Stage 12 — Get a real number

> **The question:** what does this pipeline actually score on data nobody has
> seen?

### Where you are

Everything so far is unmeasured against reality. Late submissions to the
original competition are still scored, which means you can get a genuine
external number — but only by submitting a working algorithm that runs the
whole pipeline live.

### The work

1. Build a submission notebook that, for each test slide, runs tiling, runs
   encoding, runs the model, and writes a prediction.
2. Make it handle the case where the test directory does not exist, since it
   won't during editing.
3. Attach the offline encoder weights from Stage 7.
4. Submit, even if the model on top is crude.
5. Compare the resulting score against your cross-validation number.

### The judgment calls

**Submit early and submit something bad.** A score of 0.85 from a working
pipeline is worth more right now than a score of 0.92 you cannot obtain. The
gap between your local number and the leaderboard number is information you
cannot get any other way, and it calibrates everything you do afterwards.

**Nothing is written to disk except the prediction file.** During training you
cached features because you were going to train fifty times and didn't want to
re-run the encoder fifty times. At test time there is no cache — it is a
straight loop. Roughly forty lines of code.

**Expect the local and external numbers to differ**, and treat a gap as
information about generalisation rather than as a bug. If the gap is large,
that is the domain-shift problem showing up, and it is worth understanding
before optimising anything else.

### Leave behind

A working submission notebook and a real external score. Findings log answer:
*what is the gap between our cross-validation number and the external one?*

### Traps

- A notebook that crashes when the test directory is absent.
- Relying on a network fetch for the weights.
- Waiting until the model is good. The pipeline is what you're testing.

---

## Stage 13 — What comes after

Do none of these until Stages 1 through 12 are complete and there is an
external number. Ordered by expected value.

**Resolution matching between providers.** If Stage 1 confirmed the roughly
two-fold difference in microns per pixel, make both providers' tiles cover the
same physical area rather than the same pixel count. This directly attacks the
domain-shift problem from Part 1.5, and it may be worth more than everything
else on this list. It also generalises: a pipeline that handles two scanners at
different scales handles a third by changing a configuration value.

**Stain normalisation.** Transform every tile's colours toward a common
reference so the providers look alike. Measure it — it does not always help,
and it costs time at inference.

**Multi-scale tiles.** Extract at two pyramid levels and supply both. The
coarser level gives tissue architecture, the finer gives cell morphology.
Doubles storage and extraction time.

**Richer feature vectors at zero extra GPU cost.** The encoder currently
returns only its pooled output, but the full forward pass produces all tokens.
Keeping the pooled vector concatenated with the mean of the patch tokens
doubles the feature dimension for the same compute — you simply discard less of
what you already computed. Storage doubles, so make it a flag and measure
whether the extra dimensions earn their keep.

**A second encoder for ensemble diversity.** Same tiles, different model. Cheap
because all the tiling work carries over, and two encoders whose errors differ
give a reliable small gain when averaged.

**Auxiliary mask labels.** The legitimate use of `train_label_masks`: for each
tile, compute the fraction of its pixels belonging to each annotated class, and
supply those as extra training targets. The model learns "this tile is
cancerous" explicitly rather than inferring it from a slide-level number. Safe
because it is a training-time label that never needs to exist at test time.
Note the asymmetry from Part 1.6 — Radboud masks distinguish Gleason patterns,
Karolinska masks only benign versus cancerous, so this is either Radboud-only
or collapsed to three shared classes.

**The fine-tuned CNN branch.** The frozen-feature route probably tops out
around 0.91–0.92. The proven path beyond that adds a second, architecturally
different branch: a small CNN fine-tuned end-to-end on tile-concatenation
images. The tiling work is reused unchanged; what changes is that tiles get
stitched into one image rather than encoded separately. Two branches with
uncorrelated errors, blended, is the realistic route to the target.

---

# Part 7 — File format reference

### Fold assignments

One row per slide, covering all 10,616. Columns: slide ID, ISUP grade, data
provider, duplicate group, fold number 0–4.

Generated once. Never regenerated.

### Packed feature set

Four files in one directory:

**The feature array.** fp16, shaped [total tiles, feature dimension]. Row order
matches the index. Memory-mappable, so opening is instant regardless of size
and only the rows you touch get read.

**The index.** One row per slide: slide ID, first row, one-past-last row, tile
count. Ranges are contiguous with no gaps — the verification script asserts
this, because gaps mean slides silently receive each other's features.

**The per-tile table.** One row per tile: slide ID, tile index within that
slide, x and y of the tile's top-left corner in **level-0 coordinates**, and
tissue fraction. Tile *k* of a slide lives at the index's first row plus *k*.

Coordinates are always level-0. Never store anything else — mixing coordinate
systems is the most common bug in whole-slide-image code.

**The manifest.** Version, encoder name, feature dimension, dtype, tile size,
pyramid level, tile cap, minimum tissue fraction, a description of the tissue
method, an explicit false flag for whether label masks were used, slide count,
tile count, coordinate space, creation date, and code commit hash.

### Intermediate extraction format

During extraction, one file per slide holding the features, coordinates, tissue
fractions, and the settings used. Exists only so extraction is resumable; the
packing step consumes it. Nothing downstream reads this format.

---

# Part 8 — Troubleshooting

| Symptom | Likely cause | Where to look |
|---|---|---|
| Tiles have black borders | Alpha channel dropped instead of composited | Part 4.3 |
| Tiles come from the wrong part of the slide | Level coordinates passed where level-0 expected | Part 4.2 |
| Tissue mask is all false | Wrong colour channel order, or threshold too high | Part 4.4 |
| Tissue mask is all true | Threshold too low, or an unusually dark slide | Stage 5 |
| "Too many open files" | Slide handles not closed inside a loop | Stage 1 |
| Morphology function errors | Boolean array passed where 8-bit expected | Part 4.5 |
| Tiling returns different results on reruns | Unbroken ties in the sort, or unseeded randomness | Stage 6 |
| Very few tiles on many slides | Minimum tissue fraction too high, or detector too tight | Stage 6 |
| One duplicate group has hundreds of slides | Hash threshold far too loose | Stage 3 |
| Encoder fetch fails at submission | Networking disabled; needs the offline path | Stage 7 |
| Reproduction test fails on features | Two copies of the preprocessing transform | Stage 9 |
| Reproduction test fails on coordinates | Non-deterministic tile ordering | Stage 6 |
| Session dies mid-extraction | Time limit; needs per-slide resumable output | Stage 10 |
| Out of disk during extraction | Storing fp32 instead of fp16 | Part 4.9 |
| HSV image looks like wrong colours | Displaying HSV channels as if they were RGB | Part 4.4 |

---

# Part 9 — Glossary

**Bag** — the collection of feature vectors from one slide, given to the model
as a single unlabelled set.

**Connected component** — a blob of touching pixels of the same value in a
binary image.

**Downsample factor** — how many times smaller a pyramid level is than level 0.

**Feature vector / embedding** — the list of numbers a neural network produces
to summarise an image. Here, 1536 numbers per tile.

**Fold** — one of the five equal parts the data is split into for
cross-validation.

**Foundation model** — a large network pretrained on huge unlabelled data,
reused as a feature extractor without further training. UNI2-h is one.

**fp16 / fp32** — 16-bit and 32-bit floating point. fp16 uses half the storage.

**Frozen** — used forward-only, never trained.

**Grouped split** — a split where all members of a group are forced into the
same fold. Here the groups are duplicate clusters.

**Hamming distance** — the number of positions at which two bit-strings differ.

**HSV** — a colour representation using Hue, Saturation and Value instead of
Red, Green and Blue. Easier for "is this tissue?" thresholds.

**ISUP grade** — the label, an integer 0–5, where 0 is no cancer and 5 is most
aggressive.

**Level (pyramid level)** — one resolution stored inside a whole-slide image
file. Level 0 is full resolution; higher numbers are smaller.

**Memory-mapped array** — a large file the operating system lets you index like
an array, loading only what you touch.

**MIL (Multiple Instance Learning)** — learning from bags of unlabelled items
where only the bag has a label.

**Morphological opening / closing** — mask cleanup operations. Opening removes
specks; closing fills holes.

**MPP (microns per pixel)** — how much physical tissue one pixel covers.

**OOF (out-of-fold)** — a prediction made by a model that never trained on that
slide. The honest score.

**OpenSlide** — the library for reading whole-slide image formats.

**Otsu's method** — automatic threshold selection from an image histogram.

**Perceptual hash** — an image fingerprint where visually similar images give
similar fingerprints.

**QWK (Quadratic Weighted Kappa)** — the competition metric. Penalises errors
by the square of the distance. 1.0 is perfect, 0.0 is random.

**Stratified split** — a split where each fold has similar class proportions.

**Tile / patch** — a small square cut out of a whole-slide image.

**Union-find (disjoint-set)** — the algorithm that merges pairs into groups
transitively.

**WSI (whole slide image)** — a digitised microscope slide, typically
20,000 × 20,000 pixels or larger.

---

# Part 10 — The five rules

1. **The tiling code touches nothing but the image file.** No masks, no CSVs,
   no labels, no provider information. If it needs anything else, it is wrong
   and it cannot be submitted.

2. **The fold assignment is never regenerated.** Changing it makes every
   previously recorded result incomparable.

3. **Every feature set gets a version string and a manifest.** The moment two
   feature sets exist, results become ambiguous without one. Every recorded
   result names the version that produced it.

4. **Look at the images.** Every threshold is chosen because you saw the
   pictures, not because a document suggested a number.

5. **A notebook is for looking at things. A module is for anything you run
   twice.**

---

## Working order

| Stage | What | Roughly |
|---|---|---|
| 1 | Take inventory | half a day |
| 2 | Look at the data | half a day |
| 3 | Find duplicates | one day |
| 4 | Fix the folds | half a day |
| 5 | Detect tissue without masks | two to three days |
| 6 | Place the tiles | two days |
| 7 | Get the encoder running | one day |
| 8 | Move code out of the notebook | half a day |
| 9 | Pilot and prove reproduction | one day plus GPU |
| 10 | Run the full extraction | mostly waiting |
| 11 | Pack and verify | half a day |
| 12 | Get a real number | two to three days |
| 13 | Everything after | the rest of the project |

Stage 4 is worth reaching quickly. Once the fold assignment exists and is fixed
in place, every number produced afterwards is comparable with every other one,
and modelling work can run alongside the rest of this list. Until it exists, no
number anyone produces means anything.
