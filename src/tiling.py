"""Tile selection: slide file in, tiles out.

get_tiles(slide_path) is the entry point. It finds the tissue on the slide's
smallest pyramid level, lays a grid of tile-sized cells over it, keeps the
cells with the most tissue, and reads those tiles from READ_LEVEL.

This module reads the slide file and nothing else: no masks, no CSVs, no
labels, no hospital. That is what lets it run on test slides at submission
time. It imports only image libraries, and must stay that way; notebook 6
checks it.

Three coordinate systems meet here. The mask is the smallest level, about 16
times smaller than full resolution. Tiles are read at level 1, 4 times
smaller. read_region takes its position in level-0 pixels always, whatever
level it reads, and its size in the pixels of the level it reads. Positions
are returned in level-0 pixels as (x, y).

Settings
--------
Tile
1. TILE_SIZE is the side of one tile in pixels, as the encoder receives it.
2. READ_LEVEL is the pyramid level tiles are read from.

Tissue detector, decided in Stage 5 (notebook 5)
3. SATURATION_CUT: pixels more colourful than this are tissue candidates.
4. MIN_CC_AREA: blobs smaller than this, in pixels at the smallest level, are
   dust rather than tissue.
5. INK_HUE_THRESH and INK_SAT_THRESH describe pen ink: strongly coloured and
   green-to-blue. OpenCV hue runs 0-179, not 0-359.

Tile placement, decided in Stage 6 (notebook 6)
6. TILE_PER_SLIDE_CAP is the most tiles one slide may get.
7. MIN_FRAC is the tissue floor: a cell must be at least this share tissue to
   be a candidate.
8. PAD_EDGES scores the cells that run past the right and bottom edges as if
   the missing part were empty (True), instead of dropping them (False).

Changing any of these changes every feature extracted with them.
"""

import cv2
import numpy as np
import openslide
from PIL import Image

TILE_SIZE = 224
READ_LEVEL = 1

SATURATION_CUT = 15
MIN_CC_AREA = 40
INK_HUE_THRESH = (30, 135)
INK_SAT_THRESH = 60

TILE_PER_SLIDE_CAP = 144
MIN_FRAC = 0.05
PAD_EDGES = True


def rgba_to_rgb_white(region):
    """Turn the RGBA image from read_region into RGB, with
    transparency white.

    read_region returns four channels, the fourth being opacity, and anything
    outside the slide comes back fully transparent. Dropping the fourth channel
    with .convert("RGB") would leave those areas black, which reads as very dark
    tissue. Pasting onto white is right, because empty glass is white under a
    scanner. With padded edges some tiles do reach past the slide, so this
    matters.

    1. Make a white RGB image the same size.
    2. Paste the region on top, using its opacity channel as the paste mask: opaque
       pixels keep their colour, transparent ones stay white.
    """
    background = Image.new("RGB", region.size, "white")
    background.paste(region, mask=region.split()[3])
    return background


def read_smallest_level(slide_path):
    """Read a slide's smallest pyramid level, whole.

    Returns the image and the downsample of every level, for example
    (1, 4, 16.0022). The downsamples are needed to turn mask pixels into level-0
    pixels and to size one tile on the mask.

    1. Open the slide.
    2. Read its smallest level in full, the only level small enough to read whole.
       The position (0, 0) is level-0, as read_region always expects.
    3. Keep the downsamples before closing.
    4. Close the slide in a finally block, so it is closed even if the read fails.
       A loop over thousands of slides that leaves files open dies with an error
       that does not name the cause.
    """
    slide = openslide.OpenSlide(slide_path)
    try:
        lvl_2_img = slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[slide.level_count - 1])
        ds = slide.level_downsamples
    finally:
        slide.close()
    return lvl_2_img, ds


def tissue_mask(rgb_arr):
    """The tissue detector decided in Stage 5, with the settings it chose.

    Takes an RGB array from the smallest level and returns a True/False array of
    the same height and width, True where there is tissue. Uses the image and
    nothing else.

    1. Convert RGB to HSV: hue (which colour), saturation (how strong) and value
       (how bright). COLOR_RGB2HSV, because these arrays come from PIL, which
       stores red first.
    2. sat_mask marks every pixel more colourful than SATURATION_CUT. Empty glass
       is nearly colourless; stained tissue is not.
    3. ink_mask marks pen: strongly coloured pixels in the green-to-blue part of
       the wheel. It is removed from the tissue, so a cell full of pen stroke never
       outranks a cell of glands. A cell holding tissue and ink still keeps its
       tissue.
    4. Combine the two and convert to 0/1 bytes; the clean-up functions need uint8,
       not True/False.
    5. Opening, a shrink then a grow, removes isolated specks. Closing, a grow then
       a shrink, fills small holes. Both use a 3x3 square. iterations is passed by
       name: the fourth positional argument of morphologyEx is the output array,
       not the iteration count.
    6. Label every blob of touching pixels, and keep only blobs of at least
       MIN_CC_AREA pixels. Blob 0 is the background.
    7. keep[labels] turns the per-blob decision back into a per-pixel mask.
    """
    hsv_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
    sat_mask = hsv_arr[..., 1] > SATURATION_CUT
    ink_mask = (hsv_arr[..., 1] > INK_SAT_THRESH) & (hsv_arr[..., 0] > INK_HUE_THRESH[0]) & (hsv_arr[..., 0] < INK_HUE_THRESH[1])
    mask = (sat_mask & ~ink_mask).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    for i in range(1, count):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= MIN_CC_AREA
    return keep[labels]


def score_grid(mask, cell_size, pad):
    """Score every cell of a grid laid over the mask.

    Args:
        mask:      True/False tissue mask at the smallest level
        cell_size: side of one cell in mask pixels, the footprint of one tile
        pad:       what to do with cells that run past the right or bottom edge

    Returns:
        fractions: one tissue share per cell, 0 to 1
        coords:    each cell's top-left corner as mask (row, col), in the same
                   order as fractions

    1. Walk the mask in steps of cell_size, row by row, left to right.
    2. Padded: a cell that runs past an edge is filled out to full size with
       non-tissue, so it is scored against the full cell area and kept. Its tile
       will read past the slide edge, which the white compositing fills.
    3. Trimmed: a partial cell at the right edge is skipped, and the partial
       bottom row is dropped.
    4. A cell's score is the share of its pixels that are tissue.
    """
    h, w = mask.shape
    grid = []
    grid_coords = []
    score = lambda cell: (np.sum(np.ravel(cell)).item() / (cell.shape[0] * cell.shape[1]))

    for i in range(0, h, cell_size):
        grid.append([])
        for j in range(0, w, cell_size):
            bottom_edge = i + cell_size
            right_edge = j + cell_size
            if pad:
                added_rows = 0
                added_cols = 0
                if bottom_edge > h:
                    added_rows = bottom_edge - h
                    bottom_edge = h
                if right_edge > w:
                    added_cols = right_edge - w
                    right_edge = w
                actual_cell = mask[i:bottom_edge, j:right_edge]
                actual_cell = np.concatenate((actual_cell, np.zeros((bottom_edge - i, added_cols), bool)), 1)
                actual_cell = np.concatenate((actual_cell, np.zeros((added_rows, cell_size), bool)), 0)
                grid_coords.append((i, j))
                grid[-1].append(score(actual_cell))

            else:
                if bottom_edge > h:
                    grid.pop()
                    break
                if right_edge > w:
                    continue
                actual_cell = mask[i:bottom_edge, j:right_edge]
                grid_coords.append((i, j))
                grid[-1].append(score(actual_cell))

    return np.array(grid).flatten(), np.array(grid_coords)


def select_cells(fractions, min_frac, cap):
    """Keep the best cells above the floor, up to the cap.

    Args:
        fractions: the (fractions, coords) pair from score_grid
        min_frac:  the tissue floor
        cap:       the most cells to keep

    Returns the kept cells' (row, col) and their fractions, in reading order.

    1. Keep the cells at or above the floor.
    2. Rank them by fraction, highest first. np.lexsort sorts by its last key
       first, so the keys read backwards: fraction, then row, then column. Ties
       are broken by position, never by the order the arrays happened to be built
       in, so the same slide always gives the same cells.
    3. Keep the first cap of them.
    4. Put the kept cells back into reading order, row then column. The order of
       the tiles is then fixed by position alone, and neighbours on the slide sit
       next to each other in the list.
    """
    frac, coords = fractions
    mask = frac >= min_frac
    candidate, coords_candidates = frac[mask], coords[mask]
    arg_count = min(cap, candidate.shape[0])
    final_candidates = np.lexsort((coords_candidates[:, 1], coords_candidates[:, 0], -candidate))[:arg_count]
    selected_frac, selected_coords = candidate[final_candidates], coords_candidates[final_candidates]
    reading_order = np.lexsort((selected_coords[:, 1], selected_coords[:, 0]))
    return selected_coords[reading_order], selected_frac[reading_order]


def cells_to_level0(rows, cols, mask_downsample):
    """Convert mask (row, col) to level-0 pixels.

    Multiply by the smallest level's real downsample and round. The real value is
    used, not 16, because some slides report 16.0022, and over a 30,000-pixel slide
    that difference moves a tile by a few pixels.
    """
    row = (np.round(rows * mask_downsample)).astype(np.uint32)
    col = (np.round(cols * mask_downsample)).astype(np.uint32)
    return row, col


def read_tiles(slide, coords, level, tile_size=TILE_SIZE):
    """Read one tile per position from an open slide.

    Args:
        slide:     an open OpenSlide handle; the caller closes it
        coords:    level-0 (x, y) of each tile's top-left corner
        level:     the pyramid level to read from
        tile_size: the side of a tile in that level's pixels

    read_region takes the position in level-0 pixels and the size in the pixels of
    the level being read. Each tile is composited onto white and returned as a
    PIL RGB image, in the order of coords.
    """
    region_grid = []

    for x, y in coords:
        region = slide.read_region((int(x), int(y)), level, (tile_size, tile_size))
        region_array = rgba_to_rgb_white(region)
        region_grid.append(region_array)

    return region_grid


def get_tiles(slide_path, min_frac=None, pad=None, cap=None):
    """Choose and read the tiles for one slide. It reads the slide file
    and nothing else.

    Args:
        slide_path: path of the slide's .tiff
        min_frac, pad, cap: the floor, edge handling and cap. Left as None,
            they are MIN_FRAC, PAD_EDGES and TILE_PER_SLIDE_CAP. Only the
            notebook 6 sweep passes other values; extraction and submission
            never do.

    Returns:
        tiles:       list of TILE_SIZE x TILE_SIZE RGB images, in reading order
        coords:      int32 array [n, 2], level-0 (x, y) of each tile
        tissue_frac: float32 array [n], each tile's tissue share on the mask
    A slide with no tissue gives an empty list and empty arrays, not an error.

    1. Read the smallest level and the downsamples.
    2. Find the tissue on it.
    3. Size one cell: a tile is TILE_SIZE pixels at READ_LEVEL, which is
       TILE_SIZE x that level's downsample in level-0 pixels, divided by the
       smallest level's downsample on the mask. About 56 mask pixels. Stop if it
       comes out under 8: the pyramid is not what this code expects.
    4. Score the grid and keep the best cells.
    5. Convert the kept cells to level-0 pixels, and swap (row, col) into (x, y):
       x is the column, y the row, which is the order read_region takes.
    6. Open the slide, read the tiles, and close it in a finally block.
    """
    min_frac = MIN_FRAC if min_frac is None else min_frac
    pad = PAD_EDGES if pad is None else pad
    cap = TILE_PER_SLIDE_CAP if cap is None else cap

    lvl_2_img, ds = read_smallest_level(slide_path)

    our_mask = tissue_mask(np.asarray(rgba_to_rgb_white(lvl_2_img)))

    cell_size = round(TILE_SIZE * ds[READ_LEVEL] / ds[-1])
    if cell_size < 8:
        raise ValueError(f"tile spans only {cell_size} mask pixels")

    fractions = score_grid(our_mask, cell_size, pad)

    coords, tissue_frac = select_cells(fractions, min_frac, cap)

    rows, cols = cells_to_level0(coords[:, 0], coords[:, 1], ds[-1])
    coords_level0 = np.stack([cols, rows], axis=1).astype(np.int32)

    slide = openslide.OpenSlide(slide_path)
    try:
        tiles = read_tiles(slide, coords_level0, READ_LEVEL)
    finally:
        slide.close()

    return tiles, coords_level0, tissue_frac.astype(np.float32)
