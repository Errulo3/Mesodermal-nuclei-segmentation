#!/usr/bin/env python
# ======================================================
# Elav / Mef2 intensity quantification
#
# For each raw stack, this script:
#   1. Loads the raw image, the binary mCD8-GFP mask
#      (from Binary_Mask.py), and the Cellpose nuclear
#      labels (from segmentation.py).
#   2. Computes, for every label, the fraction of its
#      voxels that fall inside the binary mask.
#   3. Splits labels into "overlapping" (>= overlap_thresh)
#      and "non-overlapping", applying a size filter.
#   4. Reports mean Elav and Mef2 intensity per group,
#      using non-overlapping labels as the baseline.
#
# Outputs (per file + a global summary) go to
# results/analysis/.
# ======================================================

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage


# ========== PATHS ==========
BASE_DIR = Path(__file__).resolve().parent.parent

input_folder        = BASE_DIR / "data" / "raw"
binary_masks_folder = BASE_DIR / "results" / "binary_masks"
labels_raw_folder   = BASE_DIR / "results" / "labels_raw"
output_folder       = BASE_DIR / "results" / "analysis"

output_folder.mkdir(parents=True, exist_ok=True)


# ========== PARAMETERS ==========
overlap_thresh = 0.5     # fraction of label voxels inside the binary mask
min_cell_size  = 250     # minimum label size in voxels
max_cell_size  = 4000    # maximum label size in voxels


# ========== FILE LIST ==========
input_files = sorted(
    f for f in input_folder.iterdir()
    if f.suffix.lower() in (".tif", ".tiff")
)

print(f"📁 Processing {len(input_files)} files from {input_folder}")


# ========== MAIN LOOP ==========
summary_data = []

for filename in input_files:
    print(f"\n{'=' * 60}")
    print(f"🚀 Processing: {filename.name}")
    start_file = time.time()

    base_name = filename.stem
    img_path          = filename
    binary_mask_path  = binary_masks_folder / f"mask-{base_name}.tif"
    labels_path       = labels_raw_folder   / f"labels_raw-{base_name}.tif"

    elav_non_csv     = output_folder / f"elav_non_overlap_{base_name}.csv"
    non_overlap_path = output_folder / f"non_overlap_{base_name}.tif"
    overlap_dual_csv = output_folder / f"elav_mef2_overlap_{base_name}.csv"
    stats_csv        = output_folder / f"stats_{base_name}.csv"
    all_overlap_path = output_folder / f"all_overlap_{base_name}.tif"

    # ========== LOAD DATA ==========
    print("  Loading data...")
    t0 = time.time()
    with tifffile.TiffFile(img_path) as tif:
        img = tif.asarray()
    binary_mask = tifffile.imread(binary_mask_path) > 0
    raw_labels  = tifffile.imread(labels_path)
    print(f"    Load time: {time.time() - t0:.1f}s")
    print(f"    Image shape: {img.shape}, Labels: {raw_labels.shape}")

    # ========== EXTRACT CHANNELS (Elav = index 1, Mef2 = index 2) ==========
    elav_ch = img[:, 1, :, :].astype(np.float32)
    mef2_ch = img[:, 2, :, :].astype(np.float32)

    # ========== OVERLAP CALCULATION ==========
    print("  Calculating overlaps...")
    t0 = time.time()
    unique_labels = np.unique(raw_labels)
    unique_labels = unique_labels[unique_labels != 0]
    print(f"    Total labels: {len(unique_labels)}")

    overlap_sums = ndimage.sum(
        binary_mask.astype(np.float32), labels=raw_labels, index=unique_labels
    )
    label_sizes = ndimage.sum(
        np.ones_like(raw_labels, dtype=np.float32),
        labels=raw_labels, index=unique_labels,
    )
    overlap_fractions = overlap_sums / label_sizes

    # ========== SIZE FILTER + OVERLAP SPLIT ==========
    size_mask           = (label_sizes >= min_cell_size) & (label_sizes <= max_cell_size)
    overlapping_mask    = (overlap_fractions >= overlap_thresh) & size_mask
    overlapping_labels  = unique_labels[overlapping_mask]
    non_overlapping_labels = unique_labels[~overlapping_mask & size_mask]
    print(f"    Size filter: kept {size_mask.sum()}, removed {(~size_mask).sum()}")
    print(f"    Overlapping: {len(overlapping_labels)}, "
          f"Non-overlapping: {len(non_overlapping_labels)}")
    print(f"    Overlap calc time: {time.time() - t0:.1f}s")

    # ========== NON-OVERLAPPING LABELS (baseline) ==========
    print("  Processing non-overlapping labels...")
    t0 = time.time()
    if len(non_overlapping_labels) > 0:
        elav_sums_non = ndimage.sum(elav_ch, labels=raw_labels, index=non_overlapping_labels)
        sizes_non     = label_sizes[~overlapping_mask & size_mask]

        avg_elav_non_overlap = elav_sums_non.sum() / sizes_non.sum()
        std_elav_non_overlap = np.std(elav_sums_non / sizes_non)

        with open(elav_non_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "mean_elav_intensity", "overlap_status"])
            for lab, intensity in zip(non_overlapping_labels, elav_sums_non / sizes_non):
                writer.writerow([lab, intensity, "non_overlapping"])

        non_overlap_mask = np.isin(raw_labels, non_overlapping_labels)
        non_overlap_labels_tiff = raw_labels.copy()
        non_overlap_labels_tiff[~non_overlap_mask] = 0
        tifffile.imwrite(non_overlap_path, non_overlap_labels_tiff.astype(np.uint16))

        print(f"    💾 Non-overlapping labels: {non_overlap_path.name}")
        print(f"    Baseline Elav average: {avg_elav_non_overlap:.2f}")
    else:
        avg_elav_non_overlap = 0
        std_elav_non_overlap = 0
        with open(elav_non_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["No non-overlapping labels found"])
        tifffile.imwrite(non_overlap_path, np.zeros_like(raw_labels, dtype=np.uint16))
        print("    ⚠️ No non-overlapping labels for baseline")
    print(f"    Non-overlap processing time: {time.time() - t0:.1f}s")

    # ========== OVERLAPPING LABELS ==========
    print("  Processing overlapping labels...")
    t0 = time.time()

    file_summary = {
        "filename": filename.name,
        "base_name": base_name,
        "total_labels": len(unique_labels),
        "overlapping_labels": len(overlapping_labels),
        "non_overlapping_labels": len(non_overlapping_labels),
        "avg_elav_non_overlap": avg_elav_non_overlap,
        "std_elav_non_overlap": std_elav_non_overlap,
        "avg_elav_overlap": 0,
        "std_elav_overlap": 0,
        "avg_mef2_overlap": 0,
        "std_mef2_overlap": 0,
        "overlap_threshold_used": overlap_thresh,
    }

    if len(overlapping_labels) > 0:
        elav_sums_overlap = ndimage.sum(elav_ch, labels=raw_labels, index=overlapping_labels)
        mef2_sums_overlap = ndimage.sum(mef2_ch, labels=raw_labels, index=overlapping_labels)
        sizes_overlap     = label_sizes[overlapping_mask]

        avg_overlapping_elav = elav_sums_overlap.sum() / sizes_overlap.sum()
        avg_overlapping_mef2 = mef2_sums_overlap.sum() / sizes_overlap.sum()
        std_overlapping_elav = np.std(elav_sums_overlap / sizes_overlap)
        std_overlapping_mef2 = np.std(mef2_sums_overlap / sizes_overlap)

        file_summary.update({
            "avg_elav_overlap": avg_overlapping_elav,
            "std_elav_overlap": std_overlapping_elav,
            "avg_mef2_overlap": avg_overlapping_mef2,
            "std_mef2_overlap": std_overlapping_mef2,
        })

        with open(overlap_dual_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "label", "mean_elav_intensity",
                "mean_mef2_intensity", "overlap_fraction",
            ])
            for i, lab in enumerate(overlapping_labels):
                writer.writerow([
                    lab,
                    elav_sums_overlap[i] / sizes_overlap[i],
                    mef2_sums_overlap[i] / sizes_overlap[i],
                    overlap_fractions[overlapping_mask][i],
                ])

        all_overlap_mask = np.isin(raw_labels, overlapping_labels)
        all_overlap_labels = raw_labels.copy()
        all_overlap_labels[~all_overlap_mask] = 0
        tifffile.imwrite(all_overlap_path, all_overlap_labels.astype(np.uint16))
    else:
        with open(overlap_dual_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["No overlapping labels found"])
        tifffile.imwrite(all_overlap_path, np.zeros_like(raw_labels, dtype=np.uint16))

    # ========== PER-FILE STATS CSV ==========
    with open(stats_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["statistic", "value"])
        for key, val in file_summary.items():
            if key in ("filename", "base_name"):
                continue
            writer.writerow([key, val])

    summary_data.append(file_summary)
    print(f"  ✅ File complete in {time.time() - start_file:.1f} seconds")


# ========== GLOBAL SUMMARY ==========
print(f"\n{'=' * 60}")
print("📊 Creating summary file...")

summary_csv = output_folder / "overlap_analysis_summary.csv"

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    column_order = [
        "filename", "base_name",
        "total_labels", "overlapping_labels", "non_overlapping_labels",
        "avg_elav_non_overlap", "std_elav_non_overlap",
        "avg_elav_overlap", "std_elav_overlap",
        "avg_mef2_overlap", "std_mef2_overlap",
        "overlap_threshold_used",
    ]
    summary_df = summary_df[column_order]
    summary_df.to_csv(summary_csv, index=False)
    print(f"  💾 Summary saved to: {summary_csv}")
    print(f"  📈 Processed {len(summary_df)} files")

    if len(summary_df) > 0 and summary_df["overlapping_labels"].sum() > 0:
        print("\n  📈 OVERALL AVERAGES (across all files):")
        print(f"    - Avg Elav in overlapping labels: {summary_df['avg_elav_overlap'].mean():.2f}")
        print(f"    - Avg Mef2 in overlapping labels: {summary_df['avg_mef2_overlap'].mean():.2f}")
        print(f"    - Total overlapping labels: {summary_df['overlapping_labels'].sum()}")
        print(f"    - Total non-overlapping labels: {summary_df['non_overlapping_labels'].sum()}")
else:
    print("  ⚠️ No summary data to save")

print(f"\n{'=' * 60}")
print("🎉 All files processed!")
print(f"📁 Outputs saved to: {output_folder}")
