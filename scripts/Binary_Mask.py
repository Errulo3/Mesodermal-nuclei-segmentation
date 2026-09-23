#!/usr/bin/env python
# ======================================================
# Binary mask generation from 3D multi-channel TIFFs
# (Otsu threshold + cleanup + size-selective hole filling)
#
# Builds a binary mCD8-GFP mask from channel 0 of each
# TIFF stack. The mask is used downstream to select
# mesodermal cells (see vnc_distance.py).
# ======================================================

from pathlib import Path

import numpy as np
from tifffile import imread, imwrite
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes


# ----------------------
# Paths (repo-relative)
# ----------------------
BASE_DIR = Path(__file__).resolve().parent.parent

input_folder = BASE_DIR / "data" / "raw"
output_folder = BASE_DIR / "results" / "binary_masks"

output_folder.mkdir(parents=True, exist_ok=True)


# ----------------------
# User parameters
# ----------------------
channel_to_use = 0     # channel index in (Z, C, Y, X); 0 = mCD8-GFP
min_size = 50          # minimum object size in pixels
max_hole_size = 200    # fill only holes smaller than this (in pixels)


# ----------------------
# Find TIFF files
# ----------------------
tiff_files = sorted(
    f for f in input_folder.iterdir()
    if f.suffix.lower() in (".tif", ".tiff")
)

if not tiff_files:
    raise RuntimeError(f"No TIFF files found in {input_folder}")

print(f"Found {len(tiff_files)} TIFF files")


# ======================================================
# Processing loop
# ======================================================
for idx, filename in enumerate(tiff_files, start=1):
    print(f"\n[{idx}/{len(tiff_files)}] Processing: {filename.name}")

    img = imread(filename)
    print(f"  Image shape: {img.shape}")

    # ----------------------
    # Extract channel
    # ----------------------
    if img.ndim == 4:  # (Z, C, Y, X)
        ch = img[:, channel_to_use]
    elif img.ndim == 3:  # (Z, Y, X)
        ch = img
    else:
        print("  ❌ Unsupported image shape, skipping")
        continue

    # ----------------------
    # Binary mask creation
    # ----------------------
    binary_mask = np.zeros_like(ch, dtype=bool)

    for z in range(ch.shape[0]):
        slice_img = ch[z]

        if slice_img.max() == 0:
            continue

        try:
            thresh = threshold_otsu(slice_img)
            mask = slice_img > thresh
        except ValueError:
            mask = slice_img > 0

        # Remove small objects
        mask = remove_small_objects(mask, min_size=min_size)

        # Fill only small holes
        mask = remove_small_holes(mask, area_threshold=max_hole_size)

        binary_mask[z] = mask

    # ----------------------
    # Save mask
    # ----------------------
    output_name = f"mask-{filename.stem}.tif"
    output_path = output_folder / output_name

    imwrite(output_path, (binary_mask.astype(np.uint8) * 255))

    print(f"  ✅ Saved: {output_name}")


# ======================================================
# Done
# ======================================================
print("\nProcessing complete.")
print(f"Masks saved in: {output_folder}")
