# Prostate Cancer Grade Assessment

A deep learning research project for automated Gleason grading of prostate cancer from Whole Slide Images (WSIs). Built on the [PANDA (Prostate cANcer graDe Assessment) dataset](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment) — the largest publicly available prostate histopathology dataset, containing 10,616 digitised H&E stained biopsies from Radboud University Medical Center and Karolinska Institute.

**Primary metric:** Quadratic Weighted Kappa (QWK).
**Target:** Exceed QWK 0.94 on the local validation set, surpassing all known public PANDA competition results.

---

## Progress

| Phase | Status | Result |
|---|---|---|
| **1 — Tile Extraction** | ✅ Complete | 382,117 tiles · 49 GB · 0 failures |
| **2 — Data Analysis & Audit** | ✅ Complete | 260 noise candidates flagged · Augmentation set |
| **3A — EfficientNet Baseline** | 🔜 Next | Target: QWK > 0.88 |
| **3B — DINOv2 + ABMIL** | ⏳ Pending | Target: QWK > 0.92 |
| **3C — TransMIL Upgrade** | ⏳ Pending | If 3B plateaus |
| **4 — Evaluation** | ⏳ Pending | — |

---

## What This Project Does

The pipeline takes a prostate biopsy Whole Slide Image and outputs an **ISUP grade (0–5)**, corresponding to the Gleason grading system used in clinical diagnosis. The architecture is a two-stage, research-grade pipeline that reflects current state-of-the-art practice in computational pathology.

**Stage 1 — Tissue-Aware Tile Extraction.** Each full-resolution TIFF slide is scanned at 20× magnification (~0.5 µm/px). A global tissue mask is derived from the HSV saturation channel using Otsu thresholding. The slide is then divided into a 256×256 grid and every candidate tile is scored by the fraction of its area classified as tissue. The top 36 most tissue-dense tiles per slide are retained. This eliminates blank glass regions and ensures the model trains exclusively on diagnostically relevant tissue.

**Stage 2 — Foundation Model Feature Extraction.** A pretrained foundation model (DINOv2 ViT-S/14, or UNI if institutional access is available) encodes each 256×256 tile into a dense feature vector. This step runs once, offline, and the embeddings are saved to disk — removing the need for repeated backbone forward passes during training experiments.

**Stage 3 — Multiple Instance Learning Aggregation.** A lightweight attention-based MIL head aggregates the 36 tile embeddings for each slide into a single slide-level representation. The attention mechanism learns to assign higher weight to diagnostically critical tiles — those containing dense Gleason pattern 4/5 nuclei — while downweighting benign or stromal regions.

**Stage 4 — Ordinal Classification.** The slide representation passes through an ordinal regression head that predicts the conditional probability of each grade threshold, respecting the biological ordering of cancer severity. This is fundamentally more appropriate than standard cross-entropy, which treats all grade errors as equally penalised.

---

## Repository Structure

```
Prostate-Cancer-Grading/
├── README.md                        # this file
├── plan.md                          # concise phase checklist
├── pyproject.toml                   # dependencies managed via uv
│
├── data/                            # not tracked by git
│   ├── train_images/                # raw PANDA BigTIFF slides (~500 GB)
│   ├── train_label_masks/           # pixel-level annotations (reserved)
│   ├── train.csv                    # slide metadata
│   ├── tiles/                       # extracted PNG tiles (49 GB)
│   ├── manifest.csv                 # one row per tile — master record
│   ├── skipped_slides.csv           # slides with zero usable tissue
│   └── failed_slides.log            # slides that raised exceptions
│
├── notebooks/
│   ├── 01_tile_extraction.ipynb     # ✅ Complete
│   ├── 02_data_analysis.ipynb       # 🔜 Next
│   ├── 03_train_efficientnet.ipynb  # ⏳ Pending
│   ├── 04_train_dinov2_abmil.ipynb  # ⏳ Pending
│   └── 05_evaluate.ipynb            # ⏳ Pending
│
└── src/
    ├── dataset.py
    ├── losses.py
    ├── train.py
    ├── metrics.py
    └── models/
        ├── efficientnet.py
        └── dinov2_abmil.py
```

> The `data/` directory is excluded from version control. Raw TIFF slides, extracted tile PNGs, and saved model weights are not committed to this repository.

---

## Requirements

### Hardware

A CUDA-capable GPU is required for model training. 8 GB of VRAM is the recommended minimum. Tile extraction is CPU-bound and requires no GPU, but benefits from fast storage — an NVMe SSD is strongly recommended given the volume of data.

For the raw dataset (~500 GB) plus extracted tiles (~50 GB), plan for at least 600 GB of available disk space.

### Software

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) — the package manager used in this project (handles virtual environments and dependency resolution)
- Git

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. All platform-specific instructions are below.

### Linux / macOS

Clone the repository and enter the project directory:

```bash
git clone https://github.com/mirza-ahsan/Prostate-Cancer-Grading.git
cd Prostate-Cancer-Grading
```

Install uv (skip if already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the virtual environment and install all dependencies:

```bash
uv sync
```

Register the Jupyter kernel so notebooks use the project environment:

```bash
uv run python -m ipykernel install --user --name prostate-cancer-grading --display-name "Prostate Cancer Grading"
```

### Windows

Open PowerShell and clone the repository:

```powershell
git clone https://github.com/mirza-ahsan/Prostate-Cancer-Grading.git
cd Prostate-Cancer-Grading
```

Install uv (skip if already installed):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create the virtual environment and install all dependencies:

```powershell
uv sync
```

Register the Jupyter kernel:

```powershell
uv run python -m ipykernel install --user --name prostate-cancer-grading --display-name "Prostate Cancer Grading"
```

### GPU Configuration

The default `pyproject.toml` targets CUDA 13.0. If you are on an older driver, a different CUDA version, or a CPU-only machine, update the index URL in `pyproject.toml` before running `uv sync`. The [PyTorch installation page](https://pytorch.org/get-started/locally/) provides the correct index URL for every platform and CUDA version.

**macOS (Apple Silicon):** Remove the CUDA index URL from `pyproject.toml` entirely and use the standard PyTorch wheel index. PyTorch will automatically detect and use MPS acceleration on M-series chips.

If you prefer standard `pip` over uv, create a virtual environment manually, activate it, and run `pip install -e .` from the project root.

### Downloading the Dataset

The PANDA dataset is available from the [Kaggle competition page](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data). A Kaggle account and acceptance of the competition rules are required. The Kaggle CLI is the most convenient download method — install it with pip, place your `kaggle.json` API token in `~/.kaggle/`, and run:

```bash
kaggle competitions download -c prostate-cancer-grade-assessment -p data/
cd data && unzip prostate-cancer-grade-assessment.zip
```

Place the extracted files under `data/` so the paths match the structure shown in the repository layout above.

---

## Running on Limited Hardware (Kaggle Free Tier)

If your machine does not have a capable GPU, insufficient RAM, or inadequate disk space, the entire pipeline can be run on **Kaggle's free tier**, which provides:

- A Tesla T4 GPU (16 GB VRAM) at no cost
- The PANDA dataset already attached — no download required
- Up to 12 hours of continuous GPU runtime per session

To use Kaggle, upload the notebooks from this repository to a new Kaggle notebook, attach the official PANDA dataset via the dataset panel, and install any additional dependencies at the top of each notebook. The notebooks are written to be self-contained and require no modification to run on Kaggle.

**Google Colab** is an alternative if you have Google Drive space for the dataset, though attaching 500 GB of data requires a paid plan. Kaggle is the recommended free option for this specific dataset.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` ≥ 2.13 | Core deep learning framework |
| `torchvision` ≥ 0.28 | Image transforms and pretrained model utilities |
| `tifffile` ≥ 2026.7 | BigTIFF whole-slide image reading via pyramidal series API |
| `imagecodecs` ≥ 2026.6 | TIFF decompression (JPEG2000, LZW) required by tifffile |
| `opencv-python-headless` ≥ 5.0 | HSV colour conversion and Otsu thresholding |
| `scikit-learn` ≥ 1.9 | Stratified train/val split and metric utilities |
| `tqdm` ≥ 4.66 | Progress bars for the multi-hour extraction loop |
| `matplotlib` / `seaborn` | Training curve and evaluation visualisation |
| `ipykernel` (dev) | Jupyter notebook kernel registration |

---

## Phase 1 — Tile Extraction (Complete)

**Notebook:** `notebooks/01_tile_extraction.ipynb`

### Design Decisions

**Resolution — Level 0 only.** The PANDA dataset is stored as BigTIFF pyramids with multiple resolution levels. The pipeline reads exclusively at Level 0 (full resolution, ~0.5 µm/px, ~20× magnification). This is the only level where Gleason patterns 3, 4, and 5 are morphologically distinguishable. Reading at Level 1 on smaller Radboud biopsies yields fewer than 36 candidate tile positions — insufficient to represent the slide — making Level 0 necessary for both datasets.

**Global Otsu thresholding from the thumbnail.** Computing the tissue mask on the full-resolution slide requires converting the entire multi-gigabyte image to HSV colour space, allocating a second equally large array in memory. Instead, the saturation histogram is computed from the lowest-resolution pyramid level (typically a few megabytes), which reliably captures the bimodal tissue/background distribution. The resulting threshold is applied per-tile during the grid scan at negligible memory cost. This is an approximation — downsampling smooths the histogram slightly — but the practical difference for tissue detection is negligible.

**Coordinate-only scan, deferred pixel copying.** During the grid scan, only (coverage, x, y) coordinate tuples are collected for candidate tiles — no pixel data is copied. After scoring all positions, the top 36 by coverage are selected and only those 36 tiles are physically copied from the image array. This caps the memory overhead of the selection step at approximately 7 MB regardless of how densely a slide is covered with tissue, avoiding the hidden memory leak that would occur from copying every tissue-positive tile upfront before sorting.

**Resumable loop with per-slide disk writes.** The manifest CSV is opened in append mode and written after each slide completes. If the process is interrupted — by a system event, memory pressure, or manual intervention — the manifest on disk accurately reflects exactly the slides that have been successfully processed. On restart, the notebook reads the existing manifest and skip log from disk (not from memory) and skips already-processed slides. This makes the multi-hour extraction run safely interruptible without losing any work or producing duplicate records.

**Non-destructive directory creation.** Output directories are created with `exist_ok=True` and the tile directory is never deleted when the notebook is re-run. This is what makes resumability meaningful — if re-run after a crash, existing tiles are preserved and the loop simply skips the slides already recorded in the manifest.

### Results

| Metric | Value |
|---|---|
| Total tiles saved | **382,117** |
| Total disk usage | **49 GB** |
| Slides with tiles | **10,615 / 10,616** |
| Slides skipped (zero tissue) | **1** |
| Slides failed (exception) | **0** |
| Mean tissue coverage per tile | **97.25%** |
| Median tissue coverage | **98.48%** |

**Split × Grade breakdown:**

| | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Grade 5 |
|---|---|---|---|---|---|---|
| **Train** | 83,263 | 76,788 | 38,664 | 35,784 | 35,956 | 35,241 |
| **Val** | 20,808 | 19,188 | 9,684 | 8,928 | 8,993 | 8,820 |

**Provider breakdown:**

| | Karolinska | Radboud |
|---|---|---|
| **Train** | 156,559 | 149,137 |
| **Val** | 39,816 | 36,605 |

The split is **stratified by ISUP grade** at the slide level with `SEED=42`. No slide contributes tiles to both the training and validation sets.

---

## What We Know About the SOTA

Understanding precisely what the competition winners did — and where the field has moved in the four years since — determines what we need to build to exceed their results.

### PANDA Competition Results (2020, top QWK ~0.94)

| Place | Team | Backbone | Tile Size | N Tiles | Defining Technique |
|---|---|---|---|---|---|
| 1st | PND (yukkyo, kentaroy47) | EfficientNet-B0/B1 | 192×192 | 64 | Label denoising via OOF cross-validation |
| 2nd | Save the Prostate | EfficientNet | 256×256 | 36–49 | Tile-grid concatenation |
| Top overall | Multiple teams | EfficientNet variants | 192–256 | 36–64 | 8-fold D4 TTA at inference |

**What actually won first place** was not the architecture — it was label denoising. The winning team used 5-fold cross-validation to collect out-of-fold (OOF) predictions for every slide, then identified slides where the model's prediction sharply contradicted the ground truth label. Particularly Grade 0 (benign) slides that the model consistently predicted as high-grade were flagged as likely containing unannotated cancerous foci and removed or downweighted in the final training mix. This directly addressed PANDA's well-documented label noise problem and delivered the largest single gain over the baseline.

**Loss function.** All top teams used ordinal loss rather than standard cross-entropy. The most common implementations were a cumulative sigmoid head (predicting the probability of each grade threshold separately) and MSE on the ordinal target. Huber loss was used specifically for Radboud slides to suppress the influence of outlier labels.

**Stain normalization.** Top competition solutions deliberately avoided Macenko or Vahadane stain normalization. Instead they relied on aggressive colour jitter augmentation — random brightness, contrast, hue, and saturation — to force the model to become stain-invariant. This proved more robust and far cheaper computationally than explicit normalization.

**Test-time augmentation (TTA).** The D4 symmetry group — 4 rotations (0°, 90°, 180°, 270°) combined with 2 flip states — produces 8 augmented versions of each slide. Averaging logits across all 8 versions at inference consistently added 0.005–0.01 QWK with no training cost.

### Post-Competition SOTA (2022–2025)

The field has moved entirely to a two-stage pipeline: a large pretrained foundation model extracts patch embeddings, and a lightweight MIL aggregator converts those embeddings into a slide-level prediction. This decouples representation learning from task adaptation and consistently outperforms the competition-era tile-concatenation approaches.

**Foundation models currently used in computational pathology:**

| Model | Developer | Pretraining | Access |
|---|---|---|---|
| **UNI** | Mahmood Lab (Harvard/MGB) | 100k+ WSIs via DINOv2 self-supervised learning | HuggingFace — gated, institutional email required |
| **UNI2-h** | Mahmood Lab | Larger scale successor to UNI | HuggingFace — gated |
| **CONCH** | Mahmood Lab | 1.17M pathology image-caption pairs (vision-language) | HuggingFace — gated |
| **Prov-GigaPath** | Microsoft / Providence | 1.3 billion patches, whole-slide context modeling | HuggingFace |
| **Virchow** | Paige / Microsoft | 1.5M WSIs from Memorial Sloan Kettering | HuggingFace (Apache 2.0) |
| **DINOv2 ViT-S/14** | Meta AI | 142M natural images, self-supervised | Fully open via `torch.hub` |

UNI and CONCH are consistently top-ranked in prostate-specific benchmarks but require an institutional research email for access. DINOv2 is the strongest fully open alternative and still substantially outperforms ImageNet-supervised CNNs on pathology tasks.

**MIL aggregators, ranked for prostate grading:**

| Aggregator | Spatial Context | Notes |
|---|---|---|
| **TransMIL** | Self-attention across all tile embeddings | Current SOTA for prostate; captures long-range tissue architecture |
| **ABMIL** | Per-tile scalar attention weight | Fast, interpretable, excellent baseline |
| **HIPT** | Hierarchical (256→4096→WSI) | Most thorough; very memory intensive |
| **DSMIL** | Dual local+global streams | Solid baseline; generally below TransMIL on large datasets |

**Current performance ceiling:** DINOv2 or UNI paired with TransMIL achieves QWK of 0.94–0.97 in curated validation cohorts. The majority of the improvement over competition-era results comes from encoder quality rather than aggregator design.

---

## Phase 2 — Data Analysis and Quality Audit (Complete)

**Notebook:** [`notebooks/02_data_analysis.ipynb`](notebooks/02_data_analysis.ipynb)

Before any model is built, the data must be understood. Phase 2 involves four audits:

**Class balance audit.** Tiles per ISUP grade per split are counted and visualised. PANDA is heavily skewed toward grades 0 and 1 (~52% of all tiles). This imbalance must be quantified before deciding on a class weighting strategy for training.

**Provider audit.** Karolinska and Radboud use different scanners and staining protocols, producing slides with visibly different colour profiles. Sample tile grids are displayed separately for each provider and each ISUP grade. If the colour profiles are substantially different, the model trained on combined data may learn scanner-specific cues rather than tissue morphology, which would appear as a systematic gap between provider-stratified QWK scores. This audit determines whether explicit stain normalization is necessary or whether colour jitter augmentation is sufficient.

**Label noise check.** Inspired by the 1st-place PANDA solution: Grade 0 (benign) slides with unusually high and spatially uniform tissue coverage across all 36 tiles are flagged as potential mislabels. The expectation is that a truly benign biopsy would contain varying tissue densities, while a slide with dense, homogeneous tissue more typical of cancer would have anomalously high and consistent tissue coverage scores. Flagged slides are candidates for soft-label treatment or exclusion.

**Tile quality inspection.** For each ISUP grade (0–5), a grid of sample tiles from both providers is displayed. The goal is to visually confirm that the tiles contain coherent, grade-appropriate tissue morphology and not scanner artifacts, pen marks, or processing debris.

---

## Phase 3A — EfficientNet Baseline

**Notebook:** `notebooks/03_train_efficientnet.ipynb`

The EfficientNet model serves as the performance floor — a replication of the competition-era approach close enough to confirm the data pipeline is correct before the more complex foundation model architecture is built.

**Architecture.** An EfficientNetV2-S backbone from the `timm` library (ImageNet pretrained) processes a 6×6 grid of the 36 extracted tiles concatenated into a single image. A global average pooling layer feeds into a two-layer classification head with GELU activation and dropout. The output is an ordinal sigmoid head rather than a softmax — each of the 5 output units predicts the probability that the slide's grade exceeds a given threshold, directly optimising for the ordinal structure of the ISUP system.

**Training strategy.** Two-stage fine-tuning: the backbone is frozen for the first 10 epochs while the head is trained at a higher learning rate, then the full model is unfrozen and fine-tuned at a lower rate with cosine annealing. Early stopping monitors validation QWK with a patience of 5 epochs.

**Label denoising.** After the first cross-validation fold completes, slides where the model's OOF prediction differs from the ground truth by two or more grades are identified and downweighted. This is the specific technique that determined the 1st-place finish in the competition.

**Expected performance:** QWK 0.88–0.91.

---

## Phase 3B — DINOv2 + ABMIL

**Notebook:** `notebooks/04_train_dinov2_abmil.ipynb`

This is the primary architecture. The two-stage approach — offline embedding extraction followed by lightweight MIL training — is both more principled and more efficient than the end-to-end CNN pipeline.

**Feature extraction.** DINOv2 ViT-S/14 processes each 256×256 tile and produces a 384-dimensional CLS token embedding. This step runs once on all 382,117 tiles and the resulting embeddings (approximately 700 MB total) are saved to disk. All subsequent training experiments operate on these embeddings directly, with no repeated backbone inference.

**Why DINOv2.** Self-supervised pretraining on 142 million images produces representations that transfer exceptionally well to H&E tissue morphology. ViT patch-based attention is particularly well-suited to tissue analysis because the structural regularity and spatial coherence of histological patterns are exactly the signals that pairwise patch attention captures. Empirically, DINOv2 features outperform ImageNet-supervised CNNs on computational pathology benchmarks by a substantial margin without any pathology-specific pretraining.

**ABMIL aggregation.** The attention mechanism learns a scalar importance weight for each of the 36 tile embeddings per slide. The slide representation is the weighted sum of tile embeddings — a learnable version of simple averaging that gives the model the ability to concentrate on the most diagnostically relevant regions. Training only the ABMIL head (frozen backbone) on pre-extracted embeddings converges in under 30 minutes on a modern GPU.

**Ordinal loss.** A Grade 3 slide produces a binary target vector of [1, 1, 1, 0, 0], where each element represents whether the grade exceeds threshold k. Five sigmoid units are trained independently on these binary targets. The predicted grade at inference is the sum of units exceeding 0.5, automatically producing an integer prediction aligned with the ISUP scale.

**Optional backbone fine-tuning.** After ABMIL convergence, partial unfreezing of the DINOv2 backbone (last 4 transformer blocks only, at a substantially reduced learning rate) can recover additional QWK. This requires gradient checkpointing and a smaller batch size to fit within 8 GB of VRAM and is only warranted if the frozen-backbone ABMIL plateaus below the target.

**Expected performance:** QWK 0.91–0.93 (frozen backbone) / 0.93–0.95 (partial fine-tune).

---

## Phase 3C — TransMIL Upgrade

If ABMIL plateaus below the primary target, the aggregation head is replaced with TransMIL. Rather than assigning each tile an independent attention weight, TransMIL runs a Transformer encoder across all 36 tile embeddings before pooling. This allows the model to capture pairwise spatial relationships between tiles — the equivalent of the holistic scan a pathologist performs across the entire biopsy before committing to a grade. Tiles containing Gleason pattern 4 adjacent to pattern 3 carry different diagnostic weight than isolated pattern 4 regions, and TransMIL is designed to learn exactly this kind of inter-tile context.

**Expected additional gain over ABMIL:** +0.01–0.02 QWK.

---

## Phase 4 — Evaluation

**Notebook:** `notebooks/05_evaluate.ipynb`

### Metrics

| Metric | Purpose |
|---|---|
| **QWK (primary)** | Penalises large ordinal errors more than small ones — the PANDA competition standard |
| Overall accuracy | Interpretable comparison floor |
| Per-grade accuracy (0–5) | Identifies which grades each model struggles with — grades 2–3 are hardest to distinguish |
| **Provider-stratified QWK** | Separate QWK for Karolinska and Radboud — a gap indicates scanner-specific overfitting rather than genuine tissue understanding |
| Confusion matrix | Full error distribution across all grade pairs |

### Test-Time Augmentation

8-fold D4 TTA is applied at inference: 4 rotations combined with 2 flip states produce 8 augmented views per slide. Logits are averaged before the final prediction. This consistently adds 0.005–0.01 QWK with no training cost.

### Interpretability

**Attention heatmaps.** For a sample of slides across all 6 grades, the 36 tiles are displayed in their original spatial arrangement with colour encoding for the ABMIL attention weights. Tiles with the highest attention (those the model found most diagnostically informative) should visually correspond to regions of dense, architecturally complex cancer — Gleason 4 cribriform patterns or Gleason 5 single infiltrating cells — rather than benign stroma or empty glass.

**Grad-CAM overlays (EfficientNet).** Spatial activation maps are overlaid on individual tile images to show which sub-regions within a tile drove the classification decision. This provides a second interpretability signal complementary to the tile-level attention weights.

---

## Ranked Path to Beating QWK 0.94

| Technique | Expected QWK Gain | Evidence |
|---|---|---|
| Label denoising — Grade 0 OOF check | +0.01–0.02 | 1st-place PANDA solution |
| DINOv2 features over EfficientNet | +0.03–0.05 | Post-competition pathology benchmarks |
| ABMIL aggregation over tile averaging | +0.01–0.02 | ABMIL paper; computational pathology benchmarks |
| Ordinal loss over cross-entropy | +0.01–0.02 | Multiple published comparisons (2022–2023) |
| 8-fold TTA at inference | +0.005–0.01 | All top PANDA solutions |
| TransMIL over ABMIL | +0.01–0.02 | Comparative MIL benchmarks |
| UNI backbone over DINOv2 (if access obtained) | +0.02–0.04 | Foundation model benchmark leaderboards |

---

## Milestone Targets

| Milestone | Criterion |
|---|---|
| **Phase 1** ✅ | `manifest.csv` covers all non-corrupted slides; 0 unaccounted; 0 failed |
| **Phase 2** ✅ | 260 label noise candidates flagged; provider colour gap quantified; jitter augmentation decided |
| **Phase 3A** | EfficientNet baseline QWK **> 0.88** |
| **Phase 3B** | DINOv2 + ABMIL QWK **> 0.92** |
| **Primary target** | Any model QWK **≥ 0.94** on local validation |
| **Stretch target** | QWK **≥ 0.96** via TransMIL or UNI backbone |

---

## Reproducibility

- Random seed `42` is set in all notebooks and passed explicitly to NumPy, PyTorch, and scikit-learn.
- The train/validation split is performed once in Notebook 01 and encoded in `manifest.csv`. All downstream notebooks read from the manifest and never re-split.
- All hyperparameters are declared in a configuration cell at the top of each training notebook. No magic numbers appear in the model or training code.
- Model checkpoints are saved with epoch number and validation QWK in the filename for unambiguous identification.
- `failed_slides.log` is reviewed and confirmed empty before any training run begins.

---

## References

- [PANDA Kaggle Competition](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment)
- [Artificial intelligence for diagnosis and Gleason grading of prostate cancer — Bulten et al., *Nature Medicine* 2022](https://www.nature.com/articles/s41591-021-01620-2)
- [1st Place PANDA Solution — yukkyo & kentaroy47](https://github.com/kentaroy47/Kaggle-PANDA-1st-place-solution)
- [DINOv2: Learning Robust Visual Features without Supervision — Oquab et al., 2023](https://arxiv.org/abs/2304.07193)
- [Towards a General-Purpose Foundation Model for Computational Pathology (UNI) — Chen et al., *Nature Medicine* 2024](https://www.nature.com/articles/s41591-024-02857-3)
- [A General-Purpose AI Copilot for Computational Pathology (CONCH) — Lu et al., 2024](https://www.nature.com/articles/s41591-024-02856-4)
- [Attention-Based Deep Multiple Instance Learning — Ilse et al., ICML 2018](https://arxiv.org/abs/1802.04712)
- [TransMIL: Transformer based Correlated MIL for WSI Classification — Shao et al., NeurIPS 2021](https://arxiv.org/abs/2106.00908)
- [A Whole-Slide Foundation Model for Pathology (Virchow) — Vorontsov et al., *Nature Medicine* 2024](https://www.nature.com/articles/s41591-024-02897-9)
- [EfficientNetV2: Smaller Models and Faster Training — Tan & Le, 2021](https://arxiv.org/abs/2104.00298)
