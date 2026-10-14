"""Pictures for checking tile placement by eye.

show_positions draws where the tiles landed; show_tiles draws the tiles
themselves. Notebook 6 uses both, and every later check of the tiling (the
Stage 9 pilot, the full extraction) should look at the same two pictures.

Both take slide ids and read the slides from paths.IMAGE_DIR.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from src.paths import slide_path
from src.tiling import READ_LEVEL, TILE_SIZE, get_tiles, read_smallest_level, rgba_to_rgb_white


def show_positions(slide_ids, titles=None, ncols=5, suptitle=None):
    """Chosen tile positions drawn as outlined squares on each slide's smallest level.

    1. Lay out a grid of panels, one per slide, all frames switched off.
    2. For each slide, read its smallest level and run get_tiles.
    3. Positions are level-0 pixels and the picture is the smallest level, so
       divide by the smallest level's downsample to draw them. One tile is
       TILE_SIZE pixels at READ_LEVEL, which is TILE_SIZE x the read level's
       downsample in level-0 pixels, divided by the smallest level's downsample
       on this picture.
    4. Caption each panel with its title (the slide id by default) and its tile
       count.

    A coordinate mistake shows up here as squares that drift off the tissue or
    bunch in a corner, which no number reveals.
    """
    slide_ids = list(slide_ids)
    titles = list(titles) if titles is not None else [s[:8] for s in slide_ids]
    nrows = math.ceil(len(slide_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 3.2))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()

    for ax, slide_id, title in zip(axes, slide_ids, titles):
        image, ds = read_smallest_level(slide_path(slide_id))
        _, coords, _ = get_tiles(slide_path(slide_id))
        side = TILE_SIZE * ds[READ_LEVEL] / ds[-1]
        ax.imshow(rgba_to_rgb_white(image))
        for x, y in coords:
            ax.add_patch(patches.Rectangle((x / ds[-1], y / ds[-1]), side, side,
                                           fill=False, edgecolor="red", linewidth=0.5))
        ax.set_title(f"{title}\n{len(coords)} tiles", fontsize=7)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def show_tiles(slide_id, n=24, ncols=8, suptitle=None):
    """The first n tiles of a slide, in reading order, captioned with their tissue fraction.

    A level mix-up (real image from the wrong place) or a dropped transparency
    channel (black edges) only shows up in the tiles themselves.
    """
    tiles, _, fracs = get_tiles(slide_path(slide_id))
    tiles, fracs = tiles[:n], fracs[:n]
    nrows = max(1, math.ceil(len(tiles) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.3, nrows * 1.45))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.set_axis_off()
    for ax, tile, frac in zip(axes, tiles, fracs):
        ax.imshow(tile)
        ax.set_title(f"{frac:.2f}", fontsize=6)
    fig.suptitle(suptitle or slide_id[:8], fontsize=11)
    fig.tight_layout()
    return fig
