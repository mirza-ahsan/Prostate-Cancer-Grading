"""Stain normalisation of a slide's tiles, before encoding (handbook Stage 13).

Five methods (Reinhard in two versions), each in two steps per slide:

1. Fit: estimate the slide's stain statistics from the tissue pixels of all
   of its tiles together. One estimate per slide, not per tile, because a
   tile with little tissue gives a noisy estimate.
2. Apply: map every tile from the slide's statistics to a fixed target.

The target is fixed per method (TARGETS, chosen in notebook 13 from reference
slides and checked by eye). At inference nothing but the slide's own tiles and
these constants is used: no masks, labels or hospital.

Methods
-------
- reinhard (Reinhard et al. 2001): match the mean and standard deviation of
  each LAB channel to the target's. Every pixel is mapped, as in the original,
  glass included: notebook 13 shows this tints glass differently per hospital.
- reinhard_tissue: the same statistics and mapping, but only tissue pixels
  are mapped; glass and anything else is copied unchanged, as retinex does.
- macenko (Macenko et al. 2009): stain vectors from the angular extremes of
  the tissue optical densities in their main plane; each pixel's stain
  concentrations, scaled so the slide's 99th percentile matches the target's,
  are recombined with the target's stain vectors.
- vahadane (Vahadane et al. 2016): stain vectors from sparse non-negative
  dictionary learning on the tissue optical densities; then the same
  concentration scaling and recombination as macenko, so the two differ only
  in how the stain vectors are estimated. Concentrations use least squares,
  not a per-pixel sparse solve, as TIAToolbox does.
- retinex (Hoque et al. 2021, ported from Hamza's implementation in the
  FRSG repo): per tile, colour deconvolution with fixed H&E vectors and a
  structure-aware illumination correction of each stain; then each stain's
  density is scaled so the slide's 95th percentile matches the target's.
  Non-tissue pixels are left as they are.

Tissue pixels are our own rule from src.tiling: saturation above
SATURATION_CUT, minus pen ink. A slide with fewer than MIN_TISSUE_PIXELS
tissue pixels cannot be fitted: its tiles are returned unchanged and the
result says so, so callers can count these instead of missing them.

normalise_slide(method, tiles) is the one entry point used by extraction and
by inference; calibrate(method, slides) computes a target from reference
slides. Tiles go in and come out as 224 x 224 RGB PIL images, as encode_tiles
takes them.
"""

import warnings

import cv2
import numpy as np
from PIL import Image
from scipy.linalg import solveh_banded
from sklearn.decomposition import DictionaryLearning
from sklearn.exceptions import ConvergenceWarning

from src.tiling import INK_HUE_THRESH, INK_SAT_THRESH, SATURATION_CUT

METHODS = ("reinhard", "reinhard_tissue", "macenko", "vahadane", "retinex")
MIN_TISSUE_PIXELS = 2000
FIT_PIXELS = 200_000
SEED = 0

# Macenko and Vahadane (shared optical-density settings)
IO = 240.0
OD_BETA = 0.15
MACENKO_ALPHA = 1.0
MAX_C_PERCENTILE = 99
VAHADANE_PIXELS = 5_000
VAHADANE_LAMBDA = 0.1

# Retinex (Hamza's values; paper Table 2 where it gives them)
RETINEX = {"od_thresh": 0.1, "gamma": 0.6, "sigma": 2.0, "eps": 1e-4, "th": 0.95, "mu": 0.1,
           "solve_size": 56}

# Targets per method: the median over 30 Radboud slides (seed 0), calibrated in notebook 13, to 4 decimals.
TARGETS = {
    "reinhard": {"mean": [175.9338, 154.062, 123.7651], "std": [28.9135, 8.5007, 4.6925]},
    "reinhard_tissue": {"mean": [175.9338, 154.062, 123.7651], "std": [28.9135, 8.5007, 4.6925]},
    "macenko": {"vectors": [[0.5286, 0.1998], [0.724, 0.8615], [0.4432, 0.4669]], "max_c": [1.7614, 1.1988]},
    "vahadane": {"vectors": [[0.4659, 0.2673], [0.7496, 0.8124], [0.4701, 0.5182]], "max_c": [2.1416, 1.4531]},
    "retinex": {"extremes": [0.4293, 0.2131]},
}


# --- shared pieces -----------------------------------------------------------------------------------------


def to_arrays(tiles):
    """PIL tiles to one uint8 array [n, h, w, 3]."""
    return np.stack([np.asarray(tile, dtype=np.uint8) for tile in tiles]) if len(tiles) else np.zeros(
        (0, 224, 224, 3), np.uint8)


def tissue_mask(arrays):
    """Tissue pixels of RGB tiles [n, h, w, 3]: saturation above SATURATION_CUT, minus pen ink (src.tiling's rule)."""
    hsv = cv2.cvtColor(arrays.reshape(-1, arrays.shape[2], 3), cv2.COLOR_RGB2HSV).reshape(arrays.shape)
    hue, sat = hsv[..., 0], hsv[..., 1]
    ink = (sat > INK_SAT_THRESH) & (hue > INK_HUE_THRESH[0]) & (hue < INK_HUE_THRESH[1])
    return (sat > SATURATION_CUT) & ~ink


def optical_density(arrays):
    """Optical density -log((I + 1) / IO) per pixel, float32, same shape."""
    return -np.log((arrays.astype(np.float32) + 1.0) / IO)


def sample_rows(rows, limit, seed=SEED):
    """At most limit rows, chosen with a fixed seed so the same slide always gives the same fit."""
    if len(rows) <= limit:
        return rows
    return rows[np.random.default_rng(seed).choice(len(rows), limit, replace=False)]


# --- Reinhard ----------------------------------------------------------------------------------------------


def reinhard_fit(arrays, mask):
    lab = cv2.cvtColor(arrays.reshape(-1, arrays.shape[2], 3), cv2.COLOR_RGB2LAB).reshape(arrays.shape)
    pixels = lab[mask].astype(np.float64)
    return {"mean": pixels.mean(axis=0), "std": pixels.std(axis=0)}


def reinhard_apply(arrays, stats, target, mask=None):
    """Map each LAB channel from the slide's mean and std to the target's; with mask, only tissue pixels change."""
    lab = cv2.cvtColor(arrays.reshape(-1, arrays.shape[2], 3), cv2.COLOR_RGB2LAB).reshape(arrays.shape)
    scale = np.asarray(target["std"]) / np.maximum(stats["std"], 1e-6)
    mapped = (lab.astype(np.float32) - stats["mean"]) * scale + np.asarray(target["mean"])
    mapped = np.clip(np.rint(mapped), 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(mapped.reshape(-1, arrays.shape[2], 3), cv2.COLOR_LAB2RGB).reshape(arrays.shape)
    return rgb if mask is None else np.where(mask[..., None], rgb, arrays)


# --- Macenko and Vahadane ----------------------------------------------------------------------------------


def order_stains(vectors):
    """Unit stain vectors as columns [3, 2], haematoxylin first (it absorbs more red than eosin)."""
    vectors = vectors / np.linalg.norm(vectors, axis=0, keepdims=True)
    return vectors if vectors[0, 0] >= vectors[0, 1] else vectors[:, ::-1]


def macenko_vectors(od):
    """Macenko stain vectors from tissue optical densities [m, 3]."""
    _, eigenvectors = np.linalg.eigh(np.cov(od.T))
    plane = eigenvectors[:, 1:3]
    projected = od @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.percentile(angles, [MACENKO_ALPHA, 100 - MACENKO_ALPHA])
    first = plane @ np.array([np.cos(low), np.sin(low)])
    second = plane @ np.array([np.cos(high), np.sin(high)])
    vectors = np.stack([first, second], axis=1)
    vectors *= np.sign(vectors.sum(axis=0, keepdims=True))  # optical densities are positive
    return order_stains(vectors)


def vahadane_vectors(od):
    """Vahadane stain vectors: sparse non-negative dictionary learning on tissue optical densities [m, 3].

    The inner lasso solves stop at their iteration limit and warn that they did
    not converge. That is expected and silenced here, only for this fit: the
    vectors agree to cosine 1.000 with a fit on four times the pixels
    (FINDINGS), and on a full extraction the warning was printed 573,869 times.
    """
    learner = DictionaryLearning(n_components=2, alpha=VAHADANE_LAMBDA, max_iter=100, fit_algorithm="cd",
                                 transform_algorithm="lasso_cd", positive_code=True, positive_dict=True,
                                 random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        learner.fit(sample_rows(od, VAHADANE_PIXELS))
    return order_stains(learner.components_.T.copy())


def concentrations(od_rows, vectors):
    """Least-squares stain concentrations [2, m] of optical densities [m, 3] on stain vectors [3, 2]."""
    return np.linalg.lstsq(vectors, od_rows.T, rcond=None)[0]


def deconvolution_fit(arrays, mask, method):
    od = optical_density(arrays)
    tissue = od[mask & (od > OD_BETA).all(axis=-1)].astype(np.float64)
    if len(tissue) < MIN_TISSUE_PIXELS:
        return None
    tissue = sample_rows(tissue, FIT_PIXELS)
    vectors = macenko_vectors(tissue) if method == "macenko" else vahadane_vectors(tissue)
    max_c = np.percentile(concentrations(tissue, vectors), MAX_C_PERCENTILE, axis=1)
    return {"vectors": vectors, "max_c": max_c}


def deconvolution_apply(arrays, stats, target):
    od = optical_density(arrays).reshape(-1, 3).astype(np.float64)
    c = concentrations(od, stats["vectors"])
    c *= (np.asarray(target["max_c"]) / np.maximum(stats["max_c"], 1e-6))[:, None]
    rgb = IO * np.exp(-(np.asarray(target["vectors"]) @ c)).T
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8).reshape(arrays.shape)


# --- Retinex (ported from Hamza's FRSG implementation) -----------------------------------------------------

LN10 = np.log(10.0)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


V_H = _unit([0.65, 0.70, 0.29])  # Ruifrok and Johnson H&E vectors, and an orthogonal residual channel
V_E = _unit([0.07, 0.99, 0.11])
V_B = _unit(np.cross(V_H, V_E))
STAIN_V = np.stack([V_H, V_E])
STAIN_MINV = np.linalg.inv(np.stack([V_H, V_E, V_B])).astype(np.float32)


def _solve_refine(t0, wh, wv, mu):
    """Solve (I + mu (Dh' Wh Dh + Dv' Wv Dv)) t = t0 with a banded Cholesky (5-point stencil)."""
    n_r, n_c = t0.shape
    wh_f, wv_f = (mu * wh).ravel(), (mu * wv).ravel()
    diag = 1.0 + wh_f + wv_f
    diag[1:] += wh_f[:-1]
    diag[n_c:] += wv_f[:-n_c]
    ab = np.zeros((n_c + 1, n_r * n_c))
    ab[n_c, :] = diag
    ab[n_c - 1, 1:] = -wh_f[:-1]
    ab[0, n_c:] = -wv_f[:-n_c]
    return solveh_banded(ab, t0.ravel(), lower=False).reshape(n_r, n_c)


def _refine_illumination(t_init):
    """Structure-aware refinement of an illumination map (relative total variation weights), on a coarse grid."""
    h, w = t_init.shape
    n = RETINEX["solve_size"]
    t0 = cv2.resize(t_init.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA).astype(np.float64)
    dh = np.zeros_like(t0)
    dh[:, :-1] = t0[:, 1:] - t0[:, :-1]
    dv = np.zeros_like(t0)
    dv[:-1, :] = t0[1:, :] - t0[:-1, :]
    wh = 1.0 / (np.abs(cv2.GaussianBlur(dh, (0, 0), RETINEX["sigma"])) + RETINEX["eps"])
    wv = 1.0 / (np.abs(cv2.GaussianBlur(dv, (0, 0), RETINEX["sigma"])) + RETINEX["eps"])
    wh[:, -1] = 0.0
    wv[-1, :] = 0.0
    t = _solve_refine(t0, wh, wv, RETINEX["mu"])
    return np.clip(cv2.resize(t.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR), 1e-3, 1.0)


def _retinex_channel(density, vector):
    """One stain's density [h, w] through illumination correction (gamma, then fusion); returns the new density."""
    single = np.exp(-LN10 * density[..., None] * vector.astype(np.float32))
    illumination = _refine_illumination(single.max(axis=-1)) ** RETINEX["gamma"]
    illumination = illumination[..., None]
    corrected = np.clip(single / illumination, 0.0, 1.0)
    fused = single * illumination + corrected * (1.0 - illumination)
    return (-np.log10(np.maximum(fused, 1e-6)) @ vector.astype(np.float32)).astype(np.float32)


def _retinex_stage1(array, mask):
    od = -np.log10((array.astype(np.float32) + 1.0) / 256.0)
    c = (od.reshape(-1, 3) @ STAIN_MINV).reshape(array.shape)
    tissue = mask & (od.max(axis=-1) >= RETINEX["od_thresh"])
    processed = np.stack([_retinex_channel(np.maximum(c[..., k], 0.0), STAIN_V[k]) for k in range(2)])
    return {"processed": processed, "residual": c[..., 2].astype(np.float32), "tissue": tissue}


def retinex_fit(arrays, mask):
    stage1 = [_retinex_stage1(array, m) for array, m in zip(arrays, mask)]
    values = np.concatenate([s["processed"][:, s["tissue"]] for s in stage1], axis=1)
    if values.shape[1] < MIN_TISSUE_PIXELS:
        return None
    values = sample_rows(values.T, FIT_PIXELS).T
    return {"extremes": np.percentile(values, 100 * RETINEX["th"], axis=1), "stage1": stage1}


def retinex_apply(arrays, stats, target):
    scale = (np.asarray(target["extremes"], np.float32) / np.maximum(stats["extremes"], 1e-6))[:, None, None]
    out = np.empty_like(arrays)
    for i, (array, s) in enumerate(zip(arrays, stats["stage1"])):
        c = s["processed"] * scale.astype(np.float32)
        od = (c[0][..., None] * V_H.astype(np.float32) + c[1][..., None] * V_E.astype(np.float32)
              + s["residual"][..., None] * V_B.astype(np.float32))
        rgb = np.clip(np.power(10.0, -od) * 256.0 - 1.0, 0, 255)
        out[i] = np.rint(np.where(s["tissue"][..., None], rgb, array)).astype(np.uint8)
    return out


# --- entry points ------------------------------------------------------------------------------------------


def fit_slide(method, arrays):
    """A slide's stain statistics from all its tiles [n, h, w, 3]; None if it has too little tissue to fit."""
    mask = tissue_mask(arrays)
    if method in ("reinhard", "reinhard_tissue"):
        return reinhard_fit(arrays, mask) if mask.sum() >= MIN_TISSUE_PIXELS else None
    if method in ("macenko", "vahadane"):
        return deconvolution_fit(arrays, mask, method)
    if method == "retinex":
        return retinex_fit(arrays, mask)
    raise ValueError(f"unknown stain method {method!r}; expected one of {METHODS}")


def target_of(method, stats):
    """The part of a slide's statistics that a target holds (drops per-tile working data)."""
    keys = {"reinhard": ("mean", "std"), "reinhard_tissue": ("mean", "std"), "macenko": ("vectors", "max_c"),
            "vahadane": ("vectors", "max_c"), "retinex": ("extremes",)}[method]
    return {key: np.asarray(stats[key]).tolist() for key in keys}


def normalise_slide(method, tiles, target=None):
    """Normalise one slide's tiles. Returns (tiles, fitted): PIL tiles, and False if they came back unchanged.

    1. Fit the slide's statistics on all its tiles' tissue pixels.
    2. If the slide has too little tissue, return the tiles unchanged with
       fitted False.
    3. Otherwise map every tile to the target (TARGETS[method] unless given).
    """
    target = TARGETS[method] if target is None else target
    if target is None:
        raise ValueError(f"no target set for {method}; calibrate it in notebook 13 first")
    if not len(tiles):
        return list(tiles), False
    arrays = to_arrays(tiles)
    stats = fit_slide(method, arrays)
    if stats is None:
        return list(tiles), False
    if method == "reinhard":
        out = reinhard_apply(arrays, stats, target)
    elif method == "reinhard_tissue":
        out = reinhard_apply(arrays, stats, target, tissue_mask(arrays))
    elif method == "retinex":
        out = retinex_apply(arrays, stats, target)
    else:
        out = deconvolution_apply(arrays, stats, target)
    return [Image.fromarray(array) for array in out], True


def calibrate(method, slides):
    """A target for method: the median of the per-slide statistics over reference slides (lists of tiles).

    Slides with too little tissue are skipped. For macenko and vahadane the
    median stain vectors are re-normalised to unit length. Returns the target
    and how many slides it was fitted on.
    """
    fitted = [fit_slide(method, to_arrays(tiles)) for tiles in slides]
    fitted = [target_of(method, stats) for stats in fitted if stats is not None]
    target = {key: np.median(np.array([f[key] for f in fitted]), axis=0) for key in fitted[0]}
    if "vectors" in target:
        target["vectors"] = target["vectors"] / np.linalg.norm(target["vectors"], axis=0, keepdims=True)
    return {key: value.tolist() for key, value in target.items()}, len(fitted)
