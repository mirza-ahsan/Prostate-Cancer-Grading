"""The encoder: tiles in, UNI2-h feature vectors out.

There are two ways to load UNI2-h, and one way to use it:

- load_encoder_online() asks HuggingFace for the model by name. It needs a
  HuggingFace login, because UNI2-h is gated. It is how the weights reach our
  machine the first time.
- load_encoder_offline(path) builds the empty network from UNI2H_KWARGS and
  fills it from a local weights file. No step asks the internet for anything.
  This is the path every Kaggle notebook uses.
- prepare(model) makes either one ready to run, and encode_tiles(model, tiles)
  turns tiles into features.

The pixel transform (scale to 0-1, normalise with MEAN and STD) lives inside
encode_tiles and nowhere else. Two copies of it would drift apart one day and
quietly spoil the features.

Precision: on the GPU the model runs in fp16. It halves the memory (1.4 GB
instead of 2.7 GB) and is the same everywhere, locally and on Kaggle, so
training and submission features come from identical arithmetic. Notebook 7
measured the cost against full precision: cosine similarity at least 0.99998.

Settings
--------
1. HF_MODEL is UNI2-h's name on HuggingFace.
2. ARCHITECTURE and UNI2H_KWARGS describe the network. They come from the
   UNI2-h model card and must match the weights exactly; they are not ours to
   tune.
3. FEATURE_DIM is the length of one tile's feature vector.
4. MEAN and STD are the colour normalisation the model was trained with.
   Notebook 7 checks them against the model's own settings.
5. DEVICE is the GPU if there is one.
6. BATCH_SIZE is how many tiles go through the model at once. 32 fits a 4 GB
   laptop GPU with room to spare.

UNI2-h is licensed CC-BY-NC-ND: the weights file is never shared, and only
ever reaches Kaggle as a private dataset.
"""

import numpy as np
import timm
import torch

from src.tiling import TILE_SIZE

HF_MODEL = "hf-hub:MahmoodLab/UNI2-h"
ARCHITECTURE = "vit_giant_patch14_224"
UNI2H_KWARGS = {
    "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24, "init_values": 1e-5,
    "embed_dim": 1536, "mlp_ratio": 2.66667 * 2, "num_classes": 0, "no_embed_class": True,
    "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU,
    "reg_tokens": 8, "dynamic_img_size": True,
}
FEATURE_DIM = 1536
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32


def load_encoder_online():
    """UNI2-h from HuggingFace, on the CPU, in fp32 (full precision).

    timm reads the model name, fetches the weights (or reuses the local
    HuggingFace cache) and builds the network with UNI2H_KWARGS.
    """
    return timm.create_model(HF_MODEL, pretrained=True, **UNI2H_KWARGS)


def load_encoder_offline(weights_path):
    """UNI2-h from a local weights file, on the CPU. Never touches the network.

    1. Build the empty network from ARCHITECTURE and UNI2H_KWARGS.
       pretrained=False, so timm fetches nothing.
    2. Read the weights file. weights_only=True reads plain tensors only, which
       is the safe way to open a file of weights.
    3. Load them into the network. load_state_dict is strict by default: a
       missing or unexpected weight is an error, not a silent gap.
    """
    model = timm.create_model(ARCHITECTURE, pretrained=False, **UNI2H_KWARGS)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model


def prepare(model):
    """Make a loaded model ready to run: evaluation mode, fp16 on the GPU.

    1. eval() switches off training-only behaviour, so the same tile always
       gives the same features.
    2. On the GPU, convert to fp16 before moving it, so the full-precision copy
       never has to fit on the card. On the CPU it stays fp32, since CPUs run
       fp16 slowly.
    """
    model = model.eval()
    if DEVICE == "cuda":
        model = model.half()
    return model.to(DEVICE)


def encode_tiles(encoder, tiles, batch_size=None):
    """Turn tiles into feature vectors.

    Args:
        encoder:    a model from prepare()
        tiles:      list of TILE_SIZE x TILE_SIZE RGB images, from get_tiles
        batch_size: tiles per forward pass; None means BATCH_SIZE

    Returns:
        float16 tensor [len(tiles), FEATURE_DIM] on the CPU, in the order of
        tiles. No tiles gives an empty [0, FEATURE_DIM] tensor.

    There is no resize: tiles are already TILE_SIZE x TILE_SIZE, and a tile of
    any other size is a bug upstream, so it stops with an error instead.

    1. Read the device and precision off the model itself, so the input always
       matches it.
    2. For each batch: check every tile is the right size and RGB, stack them
       into one array, and move it to the model's device.
    3. The transform: reorder to (tile, channel, height, width), scale to 0-1,
       subtract MEAN and divide by STD, then convert to the model's precision.
    4. Run the model without tracking gradients (inference_mode), since
       nothing is being trained.
    5. Convert the output to fp16 on the CPU explicitly, whatever the model
       returned. Its last layer is a LayerNorm, which PyTorch may compute in
       fp32.
    """
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    weight = next(encoder.parameters())
    mean = torch.tensor(MEAN, device=weight.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=weight.device).view(1, 3, 1, 1)

    features = [torch.empty((0, FEATURE_DIM), dtype=torch.float16)]
    for start in range(0, len(tiles), batch_size):
        batch = tiles[start:start + batch_size]
        if any(tile.size != (TILE_SIZE, TILE_SIZE) or tile.mode != "RGB" for tile in batch):
            raise ValueError(f"encode_tiles takes {TILE_SIZE}x{TILE_SIZE} RGB tiles")
        pixels = torch.from_numpy(np.stack([np.asarray(tile) for tile in batch])).to(weight.device)
        pixels = pixels.permute(0, 3, 1, 2).float() / 255
        pixels = ((pixels - mean) / std).to(weight.dtype)
        with torch.inference_mode():
            output = encoder(pixels)
        features.append(output.to(device="cpu", dtype=torch.float16))
    return torch.cat(features)
