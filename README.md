# Mesodermal-nuclei-segmentation

3D segmentation of Elav/Mef2 double-labeled nuclei in the *Drosophila* VNC
during mesoderm condensation, using a fine-tuned Cellpose 4 model.

---

## Overview

This repository contains:

- A **fine-tuned Cellpose 4.0.8 model** (`models/cellpose_nuclei_model`,
  581 MB, stored via Git LFS) trained to segment nuclei in 3D from
  two-channel fluorescence stacks (Elav + Mef2).
- A **batch segmentation script** (`scripts/segmentation.py`) that:
  1. Loads multi-channel TIFF stacks,
  2. Merges the Elav (channel 0) and Mef2 (channel 1) channels,
  3. Normalizes intensities (1–99 percentile rescale to 0–1),
  4. Runs 3D Cellpose inference on the full volume,
  5. Saves 16-bit label masks as `labels_raw-<name>.tif`,
  6. Maintains a JSON checkpoint so interrupted runs can resume.

---

## Repository structure


