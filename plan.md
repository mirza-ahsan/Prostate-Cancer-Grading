# Experiment Plan — Quick Reference

Full documentation, setup instructions, SOTA research, and architectural details are in [README.md](README.md).

---

## Phase Status

| Phase | Status | Key Output |
|---|---|---|
| **1 — Tile Extraction** | ✅ Complete | 382,117 tiles, 49 GB, `data/manifest.csv` |
| **2 — Data Analysis** | 🔜 Next | Audit manifest, inspect tiles, confirm augmentation strategy |
| **3A — EfficientNet Baseline** | ⏳ Pending | Target QWK > 0.88 |
| **3B — DINOv2 + ABMIL** | ⏳ Pending | Target QWK > 0.92 |
| **3C — TransMIL Upgrade** | ⏳ Pending | If 3B plateaus below 0.92 |
| **4 — Evaluation** | ⏳ Pending | Provider-stratified QWK, attention heatmaps |

---

## Immediate Next Steps

1. Open `notebooks/02_data_analysis.ipynb` (create it).
2. Load `data/manifest.csv`.
3. Audit class balance, provider colour gap, and Grade 0 label noise candidates.
4. Display sample tile grids per ISUP grade and provider.
5. Decide augmentation strategy before writing any training code.

---

## Key Numbers

- **10,615 slides** covered (10,616 total — 1 had zero tissue, logged).
- **382,117 tiles** — mean tissue coverage **97.25%**.
- **0 failed slides** — extraction was clean.
- **Split:** 80/20 stratified by ISUP grade, SEED=42.
- **Tile size:** 256×256, Level 0 (~0.5 µm/px, ~20× magnification).

---

## Target

Beat QWK 0.94 on local validation (above all known public PANDA competition results).
Stretch goal: QWK ≥ 0.96 via TransMIL or UNI backbone.