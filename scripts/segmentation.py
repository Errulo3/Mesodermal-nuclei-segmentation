#!/usr/bin/env python
import os
import json
import tifffile
import numpy as np
from cellpose import models
from skimage import exposure
import time

# ========== PATHS ==========
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE_DIR / "models" / "cellpose_nuclei_model"
input_folder = BASE_DIR / "data" / "raw"
labels_folder = BASE_DIR / "results" / "labels"

labels_folder.mkdir(parents=True, exist_ok=True)

# ========== PROCESSING SETTINGS ==========
# REMOVED chunk_size - processing entire volume at once

# ========== SIMPLIFIED PARAMETERS (ONLY COMPATIBLE ONES) ==========
SEG_PARAMS = {
    'diameter': None,
    'channels': [0, 0],
    'do_3D': True,
    'z_axis': 0,
    'channel_axis': 3,
    
    # --- SPEED OPTIMIZATIONS (ALL COMPATIBLE) ---
    'resample': False,        # 2-3x faster - no second high-res pass
    
    # --- QUALITY & SPEED BALANCE ---
    'flow_threshold': 0.4,     # Higher = faster (was 0.2)
    'cellprob_threshold': -4,   # Higher = faster
    'min_size': 20,            # Remove objects smaller than 20 pixels
}

# ========== PRINT CONFIGURATION ==========
print("=" * 60)
print("🔬 CELLPOSE BATCH SEGMENTATION (FULL VOLUME - NO CHUNKING)")
print("=" * 60)
print(f"📁 Model file: {MODEL_FILE}")
print(f"📁 Input folder: {input_folder}")
print(f"📁 Output folder: {labels_folder}")
print(f"⚡ Processing: ENTIRE 3D VOLUME AT ONCE")
print(f"⚡ resample: {SEG_PARAMS['resample']} (FASTER)")
print(f"⚡ flow_threshold: {SEG_PARAMS['flow_threshold']}")
print(f"⚡ cellprob_threshold: {SEG_PARAMS['cellprob_threshold']}")
print("=" * 60)

# ========== CHECKPOINT FOR RESUMING ==========
CHECKPOINT_FILE = os.path.join(labels_folder, "checkpoint_segmentation.json")

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoint = json.load(f)
    processed_files = checkpoint.get('processed_files', [])
    print(f"📚 Resuming from checkpoint. Already processed: {len(processed_files)} files")
else:
    processed_files = []
    checkpoint = {'processed_files': []}
    print("🆕 Starting new batch (no checkpoint found)")

# ========== GET FILES TO PROCESS ==========
all_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]
files_to_process = [f for f in all_files if f not in processed_files]

print(f"\n📊 Total images found: {len(all_files)}")
print(f"📊 Already processed: {len(processed_files)}")
print(f"📊 Remaining to process: {len(files_to_process)}")

if len(files_to_process) == 0:
    print("\n✅ All files already processed! Nothing to do.")
    exit(0)

# ========== LOAD MODEL (ONCE) ==========
print(f"\n🔄 Loading fine-tuned model...")
try:
    model = models.CellposeModel(gpu=True, pretrained_model=MODEL_FILE)
    print(f"✅ Model loaded successfully on GPU!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ========== PROCESS EACH FILE SEQUENTIALLY ==========
print("\n" + "=" * 60)
print("🚀 STARTING PROCESSING (FULL VOLUME EACH FILE)")
print("=" * 60)
print("💡 Tip: Script saves checkpoint after each image. You can stop anytime and resume later.\n")

start_time = time.time()
batch_start_time = start_time

for idx, filename in enumerate(files_to_process, 1):
    print(f"\n{'─' * 60}")
    print(f"🔬 [{idx}/{len(files_to_process)}] Processing: {filename}")
    print(f"{'─' * 60}")
    
    try:
        # Build paths
        img_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        labels_path = os.path.join(labels_folder, f"labels_raw-{base_name}.tif")
        
        # Load image
        print(f"  📂 Loading image...")
        img = tifffile.imread(img_path)
        print(f"  📐 Image shape: {img.shape}")
        
        # Merge channels 1 and 2 (indices 1 and 2)
        if len(img.shape) >= 4 and img.shape[1] >= 3:
            ch1 = img[:, 1, :, :]  # Channel 1 (index 1)
            ch2 = img[:, 2, :, :]  # Channel 2 (index 2)
            merged = (ch1.astype(np.float32) + ch2.astype(np.float32)) / 2
            print(f"  🔄 Merged channels 1 & 2")
        else:
            print(f"  ⚠️ Image doesn't have 3+ channels, using first channel")
            if len(img.shape) >= 4:
                merged = img[:, 0, :, :].astype(np.float32)
            else:
                merged = img.astype(np.float32)
        
        # Normalize to 0-1 range
        p_low, p_high = np.percentile(merged, [1, 99])
        merged_rescaled = exposure.rescale_intensity(merged, in_range=(p_low, p_high), out_range=(0, 1))
        merged_rescaled = merged_rescaled.astype(np.float32)
        merged_eval = merged_rescaled[..., np.newaxis]
        
        # ========== PROCESS ENTIRE VOLUME AT ONCE (NO CHUNKING) ==========
        print(f"  🧠 Running 3D segmentation on FULL VOLUME (all {merged_eval.shape[0]} slices)...")
        image_start = time.time()
        
        # Single call to model.eval for the entire volume
        masks_all, _, _ = model.eval(merged_eval, **SEG_PARAMS)
        
        image_elapsed = time.time() - image_start
        num_objects = masks_all.max()
        
        print(f"  ✅ Segmentation complete!")
        print(f"  ⏱️  Time: {image_elapsed/60:.2f} minutes")
        print(f"  🔢 Objects found: {num_objects}")
        
        # Save segmentation mask
        print(f"  💾 Saving mask...")
        tifffile.imwrite(labels_path, masks_all.astype(np.uint16))
        
        print(f"  ✅ Saved to: {labels_path}")
        
        # Update checkpoint
        processed_files.append(filename)
        checkpoint['processed_files'] = processed_files
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  📝 Checkpoint updated")
        
        # Show batch progress
        elapsed_this_batch = time.time() - batch_start_time
        avg_time_per_image = elapsed_this_batch / idx
        remaining_images = len(files_to_process) - idx
        estimated_remaining = avg_time_per_image * remaining_images
        
        print(f"  📊 Batch progress: {idx}/{len(files_to_process)} images")
        print(f"  ⏱️  Estimated remaining: {estimated_remaining/60:.1f} minutes")
        
        # Check if approaching time limit (optional)
        if elapsed_this_batch > 25 * 60:  # 25 minutes
            print(f"\n  ⚠️  Running for {elapsed_this_batch/60:.1f} minutes. Consider stopping here (Ctrl+C) and resuming next batch.")
            print(f"  Next image would start from: {files_to_process[idx] if idx < len(files_to_process) else 'None'}")
            
    except KeyboardInterrupt:
        print(f"\n  ⏹️ User interrupted. Checkpoint saved. Run again to resume.")
        break
    except Exception as e:
        print(f"  ❌ ERROR processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        print(f"  ⚠️ Skipping file and continuing...")
        continue

# ========== FINAL SUMMARY ==========
total_time = time.time() - start_time
successful = len(processed_files) - (len(all_files) - len(files_to_process))

print("\n" + "=" * 60)
print("🎉 BATCH PROCESSING COMPLETE!")
print("=" * 60)
print(f"✅ Processed in this batch: {successful} images")
print(f"📊 Total processed so far: {len(processed_files)}/{len(all_files)} images")
print(f"⏱️  This batch time: {total_time/60:.1f} minutes")
if successful > 0:
    print(f"⚡ Average per image: {(total_time/successful)/60:.1f} minutes")
print(f"📁 Results saved to: {labels_folder}")
print(f"📝 Checkpoint saved to: {CHECKPOINT_FILE}")

if len(processed_files) < len(all_files):
    remaining = len(all_files) - len(processed_files)
    print(f"\n📌 {remaining} images remaining. Run the script again to continue.")
    if len(processed_files) < len(all_files):
        print(f"   Next image will be: {all_files[len(processed_files)]}")
else:
    print(f"\n🎉 ALL {len(all_files)} IMAGES COMPLETE!")
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"🗑️  Checkpoint file removed (all images complete)")

print("=" * 60)