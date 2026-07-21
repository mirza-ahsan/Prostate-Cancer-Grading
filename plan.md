# Experiment Plan — Quick Reference

Full documentation, setup instructions, SOTA research, and architectural details are in [README.md](README.md).

---

## Phase Status

| Phase | Status | Key Output |
|---|---|---|
| **1 — Tile Extraction** | ✅ Complete | 382,117 tiles, 49 GB, `data/manifest.csv` |
| **2 — Data Analysis** | ✅ Complete | Data audited, 260 label noise candidates flagged, jitter confirmed |
| **3A — EfficientNet Baseline** | 🔜 Next | Target QWK > 0.88 |
| **3B — DINOv2 + ABMIL** | ⏳ Pending | Target QWK > 0.92 |
| **3C — TransMIL Upgrade** | ⏳ Pending | If 3B plateaus below 0.92 |
| **4 — Evaluation** | ⏳ Pending | Provider-stratified QWK, attention heatmaps |

---

## Immediate Next Steps

1. Open `notebooks/03_train_efficientnet.ipynb` (create it).
2. Implement a PyTorch `Dataset` that reads `data/manifest.csv` and loads the tiles.
3. Apply aggressive colour jitter augmentation, following Phase 2 decisions.
4. Build the baseline EfficientNetV2-S model with an ordinal sigmoid regression head.
5. Train the model and evaluate against the QWK > 0.88 baseline target.

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