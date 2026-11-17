"""Slide-level grading on a packed feature set: gated ABMIL, the protocols, and scoring.

Notebook 9's ABMIL head, unchanged, moved here because every feature set we
compare (the baseline and each stain normalisation) is trained and scored the
same way. Labels are used to train and score, never to choose tiles.

Protocols
---------
- "mixed": the Stage 4 folds; train on four, test on the fifth, pooled over the
  five; seed SEED.
- "radboud_to_karolinska" and "karolinska_to_radboud" (CROSS_SITE): train on
  every slide of one hospital, test on every slide of the other, once per seed
  in SEEDS. Folds are not needed: no duplicate group spans both hospitals.
- "radboud_to_karolinska_folds": Stage 9's fold-based cross-site version, kept
  only to show Stage 9 reproduces.

Functions
---------
1. load_bags: every slide's features from the packed set, copied into RAM.
2. GatedABMIL, fit_abmil, predict_abmil: the model, its fixed training, and
   its predictions.
3. split, cross_validate: the fold-based protocols; cross_site: the
   hospital-to-hospital protocol, one model per seed.
4. qwk, seed_summary, grade_distribution, score_table: scores, the spread over
   seeds, and the predicted-grade distribution, which shows over- or
   under-grading.
5. paired_bootstrap, paired_bootstrap_seeds: 95% intervals for the difference
   between two runs on the same slides.
6. save_predictions: per-slide predictions with grade, hospital and fold, next
   to the feature set.
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from torch import nn

from src.encode import DEVICE, FEATURE_DIM
from src.features import FEATURES_DIR, predictions_file

GRADES = list(range(6))
CLASSES = len(GRADES)

HIDDEN = 512
ATTENTION = 256
DROPOUT = 0.25
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 20

SEED = 0
SEEDS = list(range(5))
HOSPITALS = ("radboud", "karolinska")
CROSS_SITE = {"radboud_to_karolinska": ("radboud", "karolinska"), "karolinska_to_radboud": ("karolinska", "radboud")}
FOLD_PROTOCOLS = ("mixed", "radboud_to_karolinska_folds")
BOOTSTRAP_DRAWS = 2000


def load_bags(packed, slide_ids=None):
    """Each slide's features as a float16 CPU tensor, in a dict by slide_id.

    packed is what src.features.open_packed returns. The rows are copied out of
    the memory map (np.array makes a real copy), so training reads RAM, not
    disk. slide_ids limits it to some slides; None takes all.
    """
    wanted = None if slide_ids is None else set(slide_ids)
    return {
        slide_id: torch.from_numpy(np.array(packed.features[start:end]))
        for slide_id, start, end in zip(packed.index["slide_id"], packed.starts, packed.ends)
        if wanted is None or slide_id in wanted
    }


class GatedABMIL(nn.Module):
    """Gated attention MIL (Ilse et al. 2018): one slide's tiles [n, in_dim] in, grade logits [classes] out.

    Notebook 9's model, unchanged, so Stage 9's scores reproduce exactly.
    1. embed: each tile through a linear layer (in_dim -> hidden), ReLU, dropout.
    2. attn_v and attn_u: the two branches of the gated attention, each a linear
       layer (hidden -> attention) with dropout; tanh on one, sigmoid on the other.
    3. attn_w: their product, through a linear layer, gives one score per tile.
    4. The softmax of the scores over the tiles weights them into one slide vector.
    5. classify: a linear layer from the slide vector to one logit per grade. No
       sigmoid or softmax: cross-entropy takes raw logits.
    The layers are created in this order on purpose: with the seed set, it is the
    order the weights are initialised in, which must match notebook 9's.
    """

    def __init__(self, in_dim=FEATURE_DIM, hidden=HIDDEN, attention=ATTENTION, classes=CLASSES, dropout=DROPOUT):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attn_v = nn.Sequential(nn.Linear(hidden, attention), nn.Tanh(), nn.Dropout(dropout))
        self.attn_u = nn.Sequential(nn.Linear(hidden, attention), nn.Sigmoid(), nn.Dropout(dropout))
        self.attn_w = nn.Linear(attention, 1)
        self.classify = nn.Linear(hidden, classes)

    def forward(self, tiles):
        h = self.embed(tiles)
        scores = self.attn_w(self.attn_v(h) * self.attn_u(h)).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        return self.classify(weights @ h)


def fit_abmil(bags, train_ids, grades, seed=SEED, epochs=EPOCHS, device=DEVICE):
    """Train a GatedABMIL on the given slides and return it.

    1. Seed PyTorch with seed (torch.manual_seed), and make a separate
       generator, seeded the same, for the shuffling. The model is created after
       seeding: that order is what makes a run repeat exactly on one GPU.
    2. AdamW (LR, WEIGHT_DECAY) and a cosine learning rate over every step
       (epochs x training slides).
    3. Each epoch, visit the training slides in a shuffled order, one slide per
       step: cross-entropy of its logits against its grade, backward, optimiser
       step, schedule step.
    grades follow the order of train_ids; they are copied into a new array
    first, because pandas may hand back a read-only one.
    """
    torch.manual_seed(seed)
    order = torch.Generator().manual_seed(seed)
    model = GatedABMIL().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs * len(train_ids))
    grades = torch.tensor(np.array(grades), device=device)

    model.train()
    for _ in range(epochs):
        for i in torch.randperm(len(train_ids), generator=order).tolist():
            logits = model(bags[train_ids[i]].to(device).float())
            loss = nn.functional.cross_entropy(logits[None], grades[i:i + 1])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
    return model


def predict_abmil(model, bags, ids, device=DEVICE):
    """The predicted grade (argmax of the logits) of each slide in ids, in eval mode, as a numpy array."""
    model.eval()
    with torch.inference_mode():
        return np.array([model(bags[s].to(device).float()).argmax().item() for s in ids])


def split(labels, fold, protocol):
    """The training and test slide IDs of one fold under a fold-based protocol, in the order of labels.

    "mixed" trains on the other folds and tests on this one;
    "radboud_to_karolinska_folds" also keeps only Radboud in training and only
    Karolinska in test (Stage 9's cross-site protocol).
    """
    train = labels["fold"] != fold
    test = labels["fold"] == fold
    if protocol == "radboud_to_karolinska_folds":
        train &= labels["provider"] == "radboud"
        test &= labels["provider"] == "karolinska"
    elif protocol != "mixed":
        raise ValueError(f"unknown fold protocol {protocol!r}; expected one of {FOLD_PROTOCOLS}")
    return labels.index[train].tolist(), labels.index[test].tolist()


def cross_validate(bags, labels, protocol, seed=SEED, progress=None):
    """Train and predict on each of the five folds; the pooled test predictions, a Series by slide_id."""
    predictions = {}
    folds = sorted(labels["fold"].unique())
    for fold in progress(folds, desc=protocol) if progress else folds:
        train_ids, test_ids = split(labels, fold, protocol)
        model = fit_abmil(bags, train_ids, labels.loc[train_ids, "isup_grade"].to_numpy(), seed=seed)
        predictions.update(zip(test_ids, predict_abmil(model, bags, test_ids)))
    return pd.Series(predictions, name="predicted").sort_index()


def cross_site(bags, labels, train_hospital, test_hospital, seeds=SEEDS, progress=None):
    """Train on every slide of one hospital, test on every slide of the other, once per seed.

    Returns a table indexed by the test slides (sorted), one column of
    predicted grades per seed: seed_0, seed_1, ...
    """
    train_ids = labels.index[labels["provider"] == train_hospital].tolist()
    test_ids = sorted(labels.index[labels["provider"] == test_hospital])
    grades = labels.loc[train_ids, "isup_grade"].to_numpy()
    columns = {}
    for seed in progress(seeds, desc=f"{train_hospital} to {test_hospital}") if progress else seeds:
        model = fit_abmil(bags, train_ids, grades, seed=seed)
        columns[f"seed_{seed}"] = predict_abmil(model, bags, test_ids)
    return pd.DataFrame(columns, index=pd.Index(test_ids, name="slide_id"))


def qwk(truth, predicted):
    """Quadratic weighted kappa with grades 0-5 as the fixed label set, so a grade nobody predicted still counts."""
    return cohen_kappa_score(truth, predicted, labels=GRADES, weights="quadratic")


def _columns(predicted):
    """A run's predictions as a 2-D array [slides, seeds]: one column for a Series, one per seed for a table."""
    values = np.asarray(predicted)
    return values[:, None] if values.ndim == 1 else values


def seed_summary(truth, by_seed):
    """QWK of each seed's predictions, and their mean, standard deviation (between seeds), minimum and maximum."""
    scores = [qwk(truth, column) for column in _columns(by_seed).T]
    return {"per_seed": scores, "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "min": float(np.min(scores)), "max": float(np.max(scores))}


def grade_distribution(truth, predicted):
    """Share of slides per grade, true against predicted (all seeds pooled), as a table indexed by grade."""
    truth = np.asarray(truth)
    pooled = _columns(predicted).ravel()
    return pd.DataFrame({
        "true": np.bincount(truth, minlength=CLASSES) / len(truth),
        "predicted": np.bincount(pooled, minlength=CLASSES) / len(pooled),
    }, index=pd.Index(GRADES, name="grade"))


def paired_bootstrap(truth, first, second, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """95% interval of qwk(first) - qwk(second) over slides drawn with replacement, seeded."""
    return paired_bootstrap_seeds(truth, first, second, draws, seed)


def paired_bootstrap_seeds(truth, first_by_seed, second_by_seed, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """95% interval of the difference in mean QWK over seeds between two runs on the same slides.

    1. Draw the slides with replacement, draws times, with a seeded generator.
    2. On each draw, the mean QWK of the first run's seeds minus that of the
       second run's (one seed each reduces to the plain paired bootstrap).
    3. The 2.5th and 97.5th percentiles of those differences.
    Slide sampling and seed noise end up in one interval.
    """
    truth, first, second = np.asarray(truth), _columns(first_by_seed), _columns(second_by_seed)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        sample = rng.integers(0, len(truth), len(truth))
        differences.append(np.mean([qwk(truth[sample], c[sample]) for c in first.T])
                           - np.mean([qwk(truth[sample], c[sample]) for c in second.T]))
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high)


def score_table(labels, predictions):
    """One row per protocol and slide group: n, and QWK (for several seeds: their mean, spread, minimum, maximum).

    predictions maps protocol to its predictions (a Series, or a table with one
    column per seed). The mixed protocol is also scored per hospital.
    """
    rows = []
    for protocol, predicted in predictions.items():
        groups = {"all": predicted.index}
        if protocol == "mixed":
            groups.update({h: predicted.index[labels.loc[predicted.index, "provider"] == h] for h in HOSPITALS})
        for group, ids in groups.items():
            summary = seed_summary(labels.loc[ids, "isup_grade"], predicted.loc[ids])
            rows.append({"protocol": protocol, "slides": group, "n": len(ids), "seeds": len(summary["per_seed"]),
                         "qwk": summary["mean"], "std": summary["std"], "min": summary["min"], "max": summary["max"]})
    return pd.DataFrame(rows)


def save_predictions(version, name, labels, predicted, features_dir=FEATURES_DIR):
    """Save a run's predictions with each slide's grade, hospital and fold; returns the path.

    predicted is a Series (saved as column "predicted") or a table with one
    column per seed. The file is predictions_file(version, name), e.g. name
    "abmil-mixed"; its folder is made if needed.
    """
    columns = predicted.to_frame("predicted") if isinstance(predicted, pd.Series) else predicted
    table = labels.loc[columns.index, ["isup_grade", "provider", "fold"]].join(columns)
    path = predictions_file(version, name, features_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table.to_parquet(path)
    return path
