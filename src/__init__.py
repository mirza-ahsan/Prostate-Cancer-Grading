"""The settled parts of the pipeline, imported by the notebooks and, later, by
the extraction and submission code.

paths    where the data, derived files, feature sets and weights are, locally and on Kaggle
tiling   slide file in, tiles out; reads nothing but the image
encode   UNI2-h loading (online and offline) and tiles-to-features
features packing and verifying a feature set, and opening a packed one (memory-mapped)
display  pictures for checking tile placement by eye
"""
