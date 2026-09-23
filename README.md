# Mesodermal-nuclei-segmentation

3D segmentation and quantification of nuclei in the *Drosophila* ventral
nerve cord (VNC) during mesoderm condensation, using a fine-tuned
Cellpose 4 model.

---

## Overview

This repository contains a four-step image-analysis pipeline:

1. **`segmentation.py`** — runs 3D Cellpose inference on multi-channel
   fluorescence stacks to produce nuclear labels.
2. **`Binary_Mask.py`** — builds a binary mCD8-GFP mask from the mesodermal
   membrane channel, used to identify the GFP-positive tissue.
3. **`Elav_Mef2_quantifications.py`** — quantifies mean Elav and Mef2
   intensities inside vs. outside the binary mask, using non-overlapping
   nuclei as a baseline.
4. **`vnc_distance.py`** — combines the nuclear labels and the binary mask
   to compute distances of selected nuclei to the tissue boundary,
   producing an Excel/CSV report. **This step is interactive:** you
   manually select which labeled regions to keep (e.g. VNC vs PNS).

The pipeline also ships with a **fine-tuned Cellpose 4 model**
(`models/cellpose_nuclei_model`, ~581 MB, stored via Git LFS).

---

## Channel layout

Input TIFF stacks are expected in `(Z, C, Y, X)` order with at least 3 channels:

| Channel index | Marker | Role |
|---|---|---|
| 0 (1st) | **mCD8-GFP** | Mesodermal membrane marker → binary mask → used to select mesodermal cells |
| 1 (2nd) | **Elav** | Neural nuclei marker |
| 2 (3rd) | **Mef2** | Muscle nuclei marker |

- `segmentation.py` merges **Elav (ch 1) + Mef2 (ch 2)** before inference.
- `Binary_Mask.py` thresholds **mCD8-GFP (ch 0)**.
- `Elav_Mef2_quantifications.py` measures **Elav (ch 1)** and **Mef2 (ch 2)**.
- `vnc_distance.py` reads only the labels + binary mask (no raw channels).

---

## Repository structure

```
.
├── .gitattributes                       # Git LFS tracking rules
├── .gitignore                           # ignores raw data, results, caches
├── README.md
├── requirements.txt
├── models/
│   └── cellpose_nuclei_model            # fine-tuned Cellpose model (Git LFS)
└── scripts/
    ├── segmentation.py                  # step 1: Cellpose 3D segmentation
    ├── Binary_Mask.py                   # step 2: binary mCD8-GFP mask
    ├── Elav_Mef2_quantifications.py     # step 3: intensity quantification
    └── vnc_distance.py                  # step 4: distance analysis (interactive)
```

---

## Pipeline

```
data/raw/                          ← input TIFF stacks (Z, C, Y, X)
   ├──► segmentation.py            ──►  results/labels_raw/
   └──► Binary_Mask.py             ──►  results/binary_masks/
                                                 │
   results/labels_raw/  ────┬────────────────────┤
                            │                    │
                            ▼                    ▼
              Elav_Mef2_quantifications.py   vnc_distance.py
                            │                (interactive: select VNC)
                            ▼                    │
                  results/analysis/              ▼
                  ├── overlap_analysis_summary.csv
                  ├── stats_<name>.csv       results/vnc_distance/
                  ├── elav_non_overlap_<name>.csv
                  ├── elav_mef2_overlap_<name>.csv
                  ├── non_overlap_<name>.tif
                  └── all_overlap_<name>.tif
                                             ├── vnc_distance_report.xlsx
                                             ├── vnc_distance_report.csv
                                             ├── VNC masks (.tif)
                                             └── manual selections
```

---

## Requirements

- Python 3.9+
- **Cellpose 4.0.8** (GPU strongly recommended — 3D inference is very slow on CPU)
- See `requirements.txt` for the full dependency list.

Install:

```bash
git clone https://github.com/Errulo3/Mesodermal-nuclei-segmentation.git
cd Mesodermal-nuclei-segmentation

# Ensure the LFS model is downloaded (usually automatic on clone)
git lfs pull

# Create an environment (conda shown; venv works too)
conda create -n cellpose-seg python=3.10
conda activate cellpose-seg
pip install -r requirements.txt
```

> **GPU note:** Cellpose uses CUDA automatically if PyTorch was installed
> with CUDA support, and MPS on Apple Silicon. To force CPU, change
> `gpu=True` to `gpu=False` in `scripts/segmentation.py`.

---

## Usage

### 0. Prepare the data

Place your multi-channel TIFF stacks in:

```
data/raw/
```

Each stack must be `(Z, C, Y, X)` with at least 3 channels
(see [Channel layout](#channel-layout) above).

### 1. Nuclear segmentation

```bash
python scripts/segmentation.py
```

- Reads all `.tif` / `.tiff` files from `data/raw/`
- Merges Elav + Mef2, normalizes, runs 3D Cellpose inference
- Writes `labels_raw-<name>.tif` to `results/labels_raw/`
- **Checkpointed:** safe to interrupt and resume. Progress is stored in
  `results/labels_raw/checkpoint_segmentation.json` and removed once all
  files are processed.

Key parameters (in `SEG_PARAMS`):

| Parameter | Value | Notes |
|---|---|---|
| `diameter` | `None` | Let Cellpose estimate |
| `channels` | `[0, 0]` | Channels already merged before inference |
| `do_3D` | `True` | Full 3D segmentation |
| `z_axis` | `0` | Z is the first axis |
| `channel_axis` | `3` | Single-channel input, channel last |
| `resample` | `False` | Faster, skips second high-res pass |
| `flow_threshold` | `0.4` | Higher = faster |
| `cellprob_threshold` | `-4` | Higher = faster |
| `min_size` | `20` | Discard objects smaller than 20 px |

### 2. Binary mCD8-GFP mask

```bash
python scripts/Binary_Mask.py
```

- Reads all TIFFs from `data/raw/`
- For each Z-slice: Otsu threshold on **channel 0 (mCD8-GFP)**, remove
  objects smaller than `min_size`, fill holes smaller than `max_hole_size`
- Writes binary masks as `mask-<name>.tif` (uint8, values 0 / 255) to
  `results/binary_masks/`

Key parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `channel_to_use` | `0` | mCD8-GFP channel |
| `min_size` | `50` | Remove objects smaller than this (px) |
| `max_hole_size` | `200` | Fill holes smaller than this (px) |

### 3. Elav / Mef2 intensity quantification

```bash
python scripts/Elav_Mef2_quantifications.py
```

- Reads raw stacks from `data/raw/`, binary masks from `results/binary_masks/`,
  and nuclear labels from `results/labels_raw/`
- For every label, computes the fraction of its voxels that fall inside
  the binary mask
- Splits labels into **overlapping** (`fraction >= overlap_thresh`) and
  **non-overlapping**, after applying a size filter
- Measures mean **Elav** and **Mef2** intensity in both groups, using
  non-overlapping labels as the baseline
- Writes per-file outputs and a global summary to `results/analysis/`:

```
results/analysis/
├── overlap_analysis_summary.csv          # one row per input file
├── stats_<name>.csv                      # per-file summary stats
├── elav_non_overlap_<name>.csv           # per-label intensities (baseline group)
├── elav_mef2_overlap_<name>.csv          # per-label intensities (overlapping group)
├── non_overlap_<name>.tif                # label mask (non-overlapping only)
└── all_overlap_<name>.tif                # label mask (overlapping only)
```

Key parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `overlap_thresh` | `0.5` | Fraction of a label that must overlap the mask to count as overlapping |
| `min_cell_size` | `250` | Minimum label size (voxels) |
| `max_cell_size` | `4000` | Maximum label size (voxels) |

### 4. VNC distance analysis (interactive)

```bash
python scripts/vnc_distance.py
```

- Reads nuclear labels from `results/labels_raw/`
- Reads binary masks from `results/binary_masks/`
- **Interactively prompts you to select** which labeled regions to keep
  (e.g. VNC vs PNS)
- Writes to `results/vnc_distance/`:
  - `vnc_distance_report.xlsx` — formatted Excel report
  - `vnc_distance_report.csv` — plain CSV report
  - VNC masks (`.tif`)
  - Manual selection state (for resuming)

Key parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `overlap_thresh` | `0.5` | Fraction of a label that must overlap the mask to be considered inside the tissue |
| `min_size` | `3000` | Minimum size (voxels) of a labeled region to be considered |

---

## Model

- **File:** `models/cellpose_nuclei_model`
- **Size:** ~581 MB
- **Stored with:** Git LFS (see `.gitattributes`)
- **Framework:** Cellpose 4.0.8

If you clone without Git LFS installed, the model file will be a ~130-byte
text pointer instead of the binary. Fix with:

```bash
git lfs install
git lfs pull
```

---

## Data & results policy

`.gitignore` excludes:

- `data/raw/` — raw microscopy data
- `results/` — generated labels, masks, reports
- `.DS_Store`, `__pycache__/`, `*.pyc`

Only code, the model, and configuration live in this repo.

---

## License

TBD.
