#!/usr/bin/env python

import os
import gc
import time
import pickle
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from matplotlib.patches import Rectangle
from matplotlib import colors
import warnings
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage import morphology, measure

warnings.filterwarnings('ignore')

# ========== PATHS ==========
from pathlib import Path

# Repository root (one level above this scripts/ folder)
BASE_DIR = Path(__file__).resolve().parent.parent

# Input: Cellpose nuclear labels produced by the segmentation pipeline
labels_raw_folder = BASE_DIR / "results" / "labels"

# Input: binary embryo masks used to identify labels overlapping the GFP-positive tissue
# Place these masks in this folder and name them: mask-{base_name}.tif
binary_masks_folder = BASE_DIR / "data" / "embryo_masks"

# Output: VNC masks, checkpoints, manual selections, and distance reports
vnc_output_folder = BASE_DIR / "results" / "vnc_distance"
vnc_output_folder.mkdir(parents=True, exist_ok=True)

# ========== PARAMETERS ==========
overlap_thresh = 0.5
min_size = 3000
dilation_radius = 3
num_components_to_keep = 4
connect_distance = 20

checkpoint_file = os.path.join(vnc_output_folder, "processing_checkpoint.pkl")
manual_selection_file = os.path.join(vnc_output_folder, "manual_selections.pkl")

print(f"📁 Processing Elav-labeled Drosophila embryos...")
print(f"  Step 1: Generate VNC masks (keeping {num_components_to_keep} biggest objects)")
print(f"  Step 2: MANUAL SELECTION - view components and choose which to keep")
print(f"  Step 3: Connect nearby components (distance < {connect_distance} pixels)")
print(f"  Step 4: Calculate distances from overlapping labels to VNC")
print(f"  Checkpoint: {checkpoint_file}")

# Get all label files
all_label_files = [f for f in os.listdir(labels_raw_folder) if f.startswith('labels_raw-') and f.endswith('.tif')]
all_label_files.sort()
print(f"Found {len(all_label_files)} files to process")

# ========== CHECKPOINT HANDLING ==========
processed_files = set()
report_data = []
manual_selections = {}

if os.path.exists(manual_selection_file):
    print(f"\n📌 Found manual selection file. Loading previous selections...")
    with open(manual_selection_file, 'rb') as f:
        manual_selections = pickle.load(f)
    print(f"   Loaded {len(manual_selections)} manual selections")

if os.path.exists(checkpoint_file):
    print(f"\n📌 Found checkpoint file. Loading previous progress...")
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
        processed_files = checkpoint['processed_files']
        report_data = checkpoint['report_data']
    print(f"   Resuming from file {len(processed_files)}/{len(all_label_files)}")
else:
    print(f"\n📌 No checkpoint found. Starting fresh...")

label_files_to_process = [f for f in all_label_files if f not in processed_files]
print(f"   Files to process now: {len(label_files_to_process)}")

# ========== PRE-COMPUTE STRUCTURING ELEMENT ==========
print(f"\n🔧 Pre-computing structuring element...")
struct_elem = morphology.ball(dilation_radius) if dilation_radius > 0 else None

# ========== FUNCTION TO CONNECT NEARBY COMPONENTS ==========
def connect_nearby_components(mask, max_distance=20):
    """
    Connect components that are within max_distance of each other
    """
    if not np.any(mask):
        return mask
    
    labeled_mask, num_components = ndimage.label(mask)
    
    if num_components <= 1:
        return mask
    
    centroids = []
    for i in range(1, num_components + 1):
        comp_mask = (labeled_mask == i)
        z, y, x = np.where(comp_mask)
        if len(z) > 0:
            centroids.append([np.mean(z), np.mean(y), np.mean(x)])
        else:
            centroids.append([0, 0, 0])
    
    centroids = np.array(centroids)
    
    from scipy.spatial.distance import pdist, squareform
    
    if num_components > 1:
        distances = squareform(pdist(centroids))
        connected = np.zeros((num_components, num_components), dtype=bool)
        connected[distances < max_distance] = True
        
        from scipy.sparse.csgraph import connected_components
        n_clusters, labels = connected_components(connected, directed=False)
        
        if n_clusters == 1:
            return mask
        
        new_mask = np.zeros_like(mask)
        for cluster_id in range(n_clusters):
            cluster_components = np.where(labels == cluster_id)[0] + 1
            if len(cluster_components) == 1:
                new_mask |= (labeled_mask == cluster_components[0])
            else:
                cluster_mask = np.zeros_like(mask)
                for comp in cluster_components:
                    cluster_mask |= (labeled_mask == comp)
                struct_small = morphology.ball(3)
                dilated = morphology.dilation(cluster_mask, struct_small)
                new_mask |= dilated
        
        return new_mask
    else:
        return mask

# ========== INTERACTIVE VISUALIZATION - SIMPLIFIED FOR SPEED ==========
def show_components_composite(component_info, z_stack_size, file_name):
    """
    Show ALL components in a single image with different colors and numbers
    component_info contains: {'display_label': 1-4, 'original_label': original_component_number, ...}
    """
    if len(component_info) == 0:
        return []
    
    if len(component_info) == 1:
        print(f"  ℹ️ Only 1 component found - automatically keeping it")
        return [component_info[0]['original_label']]
    
    # Create a figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left plot: Combined view with all components
    ax1 = axes[0]
    
    # Get the middle Z-slice that contains all components
    all_z = []
    for comp in component_info:
        z_coords, _, _ = np.where(comp['mask'])
        all_z.extend(z_coords)
    
    if len(all_z) > 0:
        # Find the Z-slice with most components visible
        from collections import Counter
        z_counts = Counter(all_z)
        best_z = max(z_counts, key=z_counts.get)
        
        # Create a blank image
        comp_masks = [comp['mask'] for comp in component_info]
        if len(comp_masks) > 0:
            h, w = comp_masks[0].shape[1], comp_masks[0].shape[2]
            combined = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Color map for different components
            colors_list = [
                [255, 0, 0],    # Red
                [0, 255, 0],    # Green
                [0, 0, 255],    # Blue
                [255, 255, 0],  # Yellow
                [255, 0, 255],  # Magenta
                [0, 255, 255],  # Cyan
            ]
            
            # Overlay each component with different color
            for idx, comp in enumerate(component_info):
                comp_mask = comp['mask'][best_z, :, :]
                color = colors_list[idx % len(colors_list)]
                
                # Add colored overlay
                for c in range(3):
                    combined[:, :, c][comp_mask] = color[c]
                
                # Find centroid for label placement
                z_coords, y_coords, x_coords = np.where(comp['mask'])
                if len(y_coords) > 0 and len(x_coords) > 0:
                    # Use the slice's centroid
                    slice_mask = comp['mask'][best_z, :, :]
                    if np.any(slice_mask):
                        y_center = np.mean(np.where(slice_mask)[0])
                        x_center = np.mean(np.where(slice_mask)[1])
                    else:
                        # Fallback to overall centroid
                        y_center = np.mean(y_coords)
                        x_center = np.mean(x_coords)
                    
                    # Draw label with background (show the display number 1-4)
                    label_text = f"#{comp['display_label']}"
                    ax1.text(x_center, y_center, label_text, 
                            color='white', fontsize=16, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7))
            
            ax1.imshow(combined)
            ax1.set_title(f"All Components (Z-slice: {best_z})\nColors indicate different components", fontsize=12)
            ax1.axis('off')
    
    # Right plot: Component info table
    ax2 = axes[1]
    ax2.axis('off')
    
    # Create info table
    info_text = f"File: {file_name}\n\n"
    info_text += f"{'#':<6} {'Original ID':<12} {'Area':<10} {'Z-start':<10} {'Z-end':<10} {'Z-range':<10}\n"
    info_text += "-" * 65 + "\n"
    
    for comp in component_info:
        info_text += f"#{comp['display_label']:<5} {comp['original_label']:<12} {comp['area']:<10} {comp['z_min']:<10.0f} {comp['z_max']:<10.0f} {comp['z_max']-comp['z_min']:<10.0f}\n"
    
    # Add color legend
    info_text += "\n" + "-" * 65 + "\n"
    info_text += "Colors:\n"
    for idx, comp in enumerate(component_info):
        color_name = ['Red', 'Green', 'Blue', 'Yellow', 'Magenta', 'Cyan'][idx % 6]
        info_text += f"  #{comp['display_label']}: {color_name}\n"
    
    info_text += "\n" + "-" * 65 + "\n"
    info_text += "TIP: Keep components that START at z=0 (bottom)\n"
    info_text += "     Exclude components that START in middle (z>0)\n"
    info_text += "     Type the DISPLAY numbers (1-4) to exclude"
    
    ax2.text(0.1, 0.9, info_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.show(block=False)
    
    # Print component info in console
    print(f"\n  📋 Components found:")
    for comp in component_info:
        print(f"       #{comp['display_label']} (original ID: {comp['original_label']}): area={comp['area']}, z_range={comp['z_min']:.0f}-{comp['z_max']:.0f}")
    
    print(f"\n  ✏️  Enter DISPLAY numbers to EXCLUDE (PNS):")
    print(f"     - Type numbers separated by commas (e.g., '2,4')")
    print(f"     - Type 'all' to keep all components")
    print(f"     - Type 'none' to keep none (rare)")
    print(f"     - Type 'auto' for automatic Z-based filtering (keeps bottom 25%)")
    print(f"     - Type 'largest' to keep only the largest component")
    
    user_input = input("  Enter choice: ").strip()
    
    # Close the figure properly
    plt.close('all')
    
    # Parse user input - return ORIGINAL labels
    if user_input.lower() == 'all' or user_input == '':
        return [c['original_label'] for c in component_info]
    elif user_input.lower() == 'none':
        return []
    elif user_input.lower() == 'largest':
        return [component_info[0]['original_label']]
    elif user_input.lower() == 'auto':
        # Automatic filtering: keep components that start in bottom 25%
        bottom_threshold = z_stack_size * 0.25
        keep = [c['original_label'] for c in component_info if c['z_min'] < bottom_threshold]
        if len(keep) == 0:
            # If none in bottom, keep the one with smallest z_min
            keep = [min(component_info, key=lambda x: x['z_min'])['original_label']]
        print(f"  🔧 Auto-selected: keeping components {keep}")
        return keep
    else:
        try:
            # Parse display numbers (1-4) to exclude
            exclude_display = [int(x.strip()) for x in user_input.split(',') if x.strip()]
            # Convert display numbers to original labels
            keep = []
            for comp in component_info:
                if comp['display_label'] not in exclude_display:
                    keep.append(comp['original_label'])
            if len(keep) == 0:
                print(f"  ⚠️ No components kept! Keeping largest as fallback.")
                keep = [component_info[0]['original_label']]
            return keep
        except:
            print(f"  ⚠️ Invalid input. Keeping all components.")
            return [c['original_label'] for c in component_info]

def get_component_info(mask, labeled_mask, components_to_keep):
    """
    Get info about top components for manual selection
    Maps original component labels to simple 1-4 display numbers
    """
    if not np.any(mask):
        return []
    
    # Get component info for the selected components only
    component_info = []
    for idx, comp_label in enumerate(components_to_keep):
        comp_mask = (labeled_mask == comp_label)
        if not np.any(comp_mask):
            continue
        
        # Get Z-positions
        z_coords, y_coords, x_coords = np.where(comp_mask)
        z_mean = np.mean(z_coords)
        z_min = np.min(z_coords)
        z_max = np.max(z_coords)
        area = len(z_coords)
        
        component_info.append({
            'display_label': idx + 1,  # Simple 1-4 numbering for display
            'original_label': comp_label,  # The actual component number from labeled_mask
            'area': area,
            'z_mean': z_mean,
            'z_min': z_min,
            'z_max': z_max,
            'mask': comp_mask
        })
    
    # Sort by area (largest first) and re-assign display labels
    component_info.sort(key=lambda x: x['area'], reverse=True)
    for idx, comp in enumerate(component_info):
        comp['display_label'] = idx + 1
    
    return component_info

# ========== PROCESS SINGLE FILE WITH MANUAL SELECTION ==========
def process_single_file(label_file, struct_elem, file_index, total_files):
    """
    Process a single file with manual component selection
    """
    print(f"\n[{file_index}/{total_files}] Processing: {label_file}")
    
    try:
        base_name = label_file.replace('labels_raw-', '').replace('.tif', '')
        
        # Check if we already have a manual selection for this file
        if base_name in manual_selections:
            print(f"  📌 Using saved manual selection: {manual_selections[base_name]}")
            keep_labels = manual_selections[base_name]
            manual_mode = False
        else:
            manual_mode = True
        
        labels_path = os.path.join(labels_raw_folder, label_file)
        mask_path = os.path.join(binary_masks_folder, f"mask-{base_name}.tif")
        vnc_mask_path = os.path.join(vnc_output_folder, f"vnc_mask_{base_name}.tif")
        
        if not os.path.exists(labels_path):
            print(f"  ❌ Labels file not found: {labels_path}")
            return {'filename': base_name, 'error': 'labels_file_missing', 'success': False}
        if not os.path.exists(mask_path):
            print(f"  ❌ Mask file not found: {mask_path}")
            return {'filename': base_name, 'error': 'mask_file_missing', 'success': False}
        
        labels_size = os.path.getsize(labels_path) / (1024**3)
        mask_size = os.path.getsize(mask_path) / (1024**3)
        print(f"  📁 Labels size: {labels_size:.2f} GB, Mask size: {mask_size:.2f} GB")
        
        print(f"  📖 Loading labels...")
        raw_labels = tifffile.imread(labels_path)
        print(f"  📖 Loading mask...")
        embryo_mask = tifffile.imread(mask_path) > 0
        
        print(f"  📊 Labels shape: {raw_labels.shape}")
        
        unique_labels = np.unique(raw_labels)
        unique_labels = unique_labels[unique_labels != 0]
        
        if len(unique_labels) == 0:
            print(f"  ⚠️ No labels found")
            return {'filename': base_name, 'error': 'no_labels', 'success': False}
        
        print(f"  🔄 Calculating overlaps...")
        overlap_sums = ndimage.sum(embryo_mask.astype(np.float32), 
                                   labels=raw_labels, index=unique_labels)
        label_sizes = ndimage.sum(np.ones_like(raw_labels, dtype=np.float32), 
                                 labels=raw_labels, index=unique_labels)
        overlap_fractions = overlap_sums / label_sizes
        
        non_overlap_labels = unique_labels[overlap_fractions < overlap_thresh]
        print(f"  🎯 Non-overlapping labels: {len(non_overlap_labels)}/{len(unique_labels)}")
        
        if len(non_overlap_labels) == 0:
            print(f"  ⚠️ No non-overlapping labels found, creating empty mask")
            empty_mask = np.zeros_like(raw_labels, dtype=bool)
            tifffile.imwrite(vnc_mask_path, (empty_mask * 255).astype(np.uint8))
            vnc_mask = empty_mask
            components_kept = 0
        else:
            print(f"  🎨 Creating initial mask...")
            initial_mask = np.isin(raw_labels, non_overlap_labels)
            
            # Find connected components
            labeled_mask, num_components = ndimage.label(initial_mask)
            print(f"  🔗 Found {num_components} connected components")
            
            if num_components > 0:
                # Get component sizes
                component_sizes = ndimage.sum(initial_mask.astype(np.float32), 
                                             labeled_mask, range(1, num_components + 1))
                
                # Filter by minimum size
                valid_mask = component_sizes >= min_size
                valid_components = np.where(valid_mask)[0] + 1
                valid_sizes = component_sizes[valid_mask]
                
                print(f"  📏 Components >={min_size}: {len(valid_components)}")
                
                if len(valid_components) > 0:
                    # Keep top N components by size
                    n_to_keep = min(num_components_to_keep, len(valid_components))
                    
                    if len(valid_components) > n_to_keep:
                        top_indices = np.argpartition(valid_sizes, -n_to_keep)[-n_to_keep:]
                        top_sorted_idx = np.argsort(valid_sizes[top_indices])[::-1]
                        components_to_keep = valid_components[top_indices[top_sorted_idx]]
                    else:
                        components_to_keep = valid_components
                    
                    print(f"  ✅ Keeping {len(components_to_keep)} largest components")
                    
                    # Get component info for manual selection (maps to 1-4 display numbers)
                    component_info = get_component_info(initial_mask, labeled_mask, components_to_keep)
                    
                    if manual_mode and len(component_info) > 1:
                        # Show composite view with all components
                        print(f"\n  🖱️  Opening composite view...")
                        keep_original_labels = show_components_composite(component_info, raw_labels.shape[0], base_name)
                        # Save selection (these are the ORIGINAL component labels from labeled_mask)
                        manual_selections[base_name] = keep_original_labels
                        with open(manual_selection_file, 'wb') as f:
                            pickle.dump(manual_selections, f)
                        print(f"  💾 Saved manual selection: {keep_original_labels}")
                    elif not manual_mode:
                        # Use saved selection
                        keep_original_labels = manual_selections.get(base_name, components_to_keep)
                        print(f"  📌 Using saved selection: {keep_original_labels}")
                    else:
                        # Only one component or no manual mode
                        keep_original_labels = components_to_keep
                    
                    # Create final mask with selected components
                    # Use the ORIGINAL labeled_mask with the original component labels
                    vnc_mask = np.zeros_like(raw_labels, dtype=bool)
                    for label in keep_original_labels:
                        vnc_mask |= (labeled_mask == label)
                    
                    components_kept = len(keep_original_labels)
                    print(f"  ✅ Final components kept (original IDs): {keep_original_labels}")
                    
                    # Connect nearby components
                    print(f"  🔗 Connecting nearby components...")
                    vnc_mask = connect_nearby_components(vnc_mask, connect_distance)
                    
                    # Dilate final mask
                    if dilation_radius > 0 and struct_elem is not None:
                        print(f"  🔄 Dilating mask (radius={dilation_radius})...")
                        vnc_mask = morphology.dilation(vnc_mask, struct_elem)
                else:
                    vnc_mask = np.zeros_like(raw_labels, dtype=bool)
                    components_kept = 0
            else:
                vnc_mask = np.zeros_like(raw_labels, dtype=bool)
                components_kept = 0
            
            # Save VNC mask
            print(f"  💾 Saving VNC mask...")
            tifffile.imwrite(vnc_mask_path, (vnc_mask * 255).astype(np.uint8))
        
        # Calculate distances for overlapping labels
        overlapping_labels = unique_labels[overlap_fractions >= overlap_thresh]
        num_overlapping = len(overlapping_labels)
        avg_min_distance = 0
        
        print(f"  📐 Overlapping labels: {num_overlapping}")
        
        if num_overlapping > 0 and np.any(vnc_mask):
            print(f"  📏 Calculating distance transform...")
            distance_map = distance_transform_edt(~vnc_mask)
            min_distances = []
            
            chunk_size = 200
            for i in range(0, num_overlapping, chunk_size):
                chunk_labels = overlapping_labels[i:i+chunk_size]
                for lab in chunk_labels:
                    label_mask = (raw_labels == lab)
                    label_distances = distance_map[label_mask]
                    if len(label_distances) > 0:
                        min_distances.append(np.min(label_distances))
            
            if min_distances:
                avg_min_distance = np.mean(min_distances)
                print(f"  ✅ Average min distance: {avg_min_distance:.2f} pixels")
        
        gc.collect()
        
        return {
            'filename': base_name,
            'num_overlapping_labels': num_overlapping,
            'avg_min_distance_to_vnc': avg_min_distance,
            'vnc_mask_exists': np.any(vnc_mask),
            'components_kept': components_kept,
            'success': True
        }
        
    except MemoryError as e:
        print(f"  ❌ MEMORY ERROR: {str(e)}")
        gc.collect()
        return {
            'filename': label_file,
            'error': f'memory_error: {str(e)}',
            'traceback': traceback.format_exc(),
            'success': False
        }
    except Exception as e:
        print(f"  ❌ ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return {
            'filename': label_file,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'success': False
        }

# ========== MAIN PROCESSING LOOP ==========
print(f"\n🚀 Starting sequential processing...")

if label_files_to_process:
    total_to_process = len(label_files_to_process)
    successful_count = 0
    failed_count = 0
    
    for idx, label_file in enumerate(label_files_to_process, 1):
        print(f"\n{'='*60}")
        print(f"Processing file {idx}/{total_to_process}")
        print(f"{'='*60}")
        
        result = process_single_file(label_file, struct_elem, idx, total_to_process)
        
        if result['success']:
            report_data.append(result)
            processed_files.add(label_file)
            successful_count += 1
            print(f"\n  ✅ SUCCESS: {result['filename']}")
            print(f"     - Overlapping labels: {result['num_overlapping_labels']}")
            print(f"     - Avg distance to VNC: {result['avg_min_distance_to_vnc']:.2f} pixels")
            print(f"     - Components kept: {result['components_kept']}")
        else:
            processed_files.add(label_file)
            failed_count += 1
            print(f"\n  ❌ FAILED: {label_file}")
            print(f"     Error: {result.get('error', 'unknown')}")
        
        # Save checkpoint after each file
        print(f"\n  💾 Saving checkpoint...")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'processed_files': processed_files, 'report_data': report_data}, f)
        
        print(f"\n  📊 Progress: {idx}/{total_to_process} files")
        print(f"     Successful: {successful_count}, Failed: {failed_count}")
        
        gc.collect()
        if idx < total_to_process:
            print(f"  ⏸️  Waiting 1 second...")
            time.sleep(1)

# Save final checkpoint
print(f"\n💾 Saving final checkpoint...")
with open(checkpoint_file, 'wb') as f:
    pickle.dump({'processed_files': processed_files, 'report_data': report_data}, f)

# ========== CREATE EXCEL REPORT ==========
print(f"\n{'='*60}")
print(f"📊 Creating Excel report...")

if report_data:
    successful_results = [r for r in report_data if r.get('success', True)]
    
    if successful_results:
        report_df = pd.DataFrame(successful_results)
        report_df = report_df.sort_values('filename')
        
        # Calculate statistics
        total_labels = np.sum(report_df['num_overlapping_labels'].values)
        mean_distance = np.mean(report_df['avg_min_distance_to_vnc'].values)
        std_distance = np.std(report_df['avg_min_distance_to_vnc'].values)
        
        # Create summary
        summary_data = {
            'filename': ['', 'STATISTICS', 'SUMMARY'],
            'num_overlapping_labels': [
                '',
                f"Total: {total_labels}",
                total_labels
            ],
            'avg_min_distance_to_vnc': [
                '',
                f"Mean: {mean_distance:.2f} ± {std_distance:.2f} pixels",
                mean_distance
            ],
            'vnc_mask_exists': [
                '',
                f"Files with VNC: {np.sum(report_df['vnc_mask_exists'].values)}/{len(report_df)}",
                np.sum(report_df['vnc_mask_exists'].values)
            ],
            'components_kept': [
                '',
                f"Avg: {np.mean(report_df['components_kept'].values):.1f}",
                ''
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        final_df = pd.concat([report_df, summary_df], ignore_index=True)
        
        excel_path = os.path.join(vnc_output_folder, "vnc_distance_report.xlsx")
        
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                final_df.to_excel(writer, sheet_name='Average Min Distances', index=False)
                worksheet = writer.sheets['Average Min Distances']
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    worksheet.column_dimensions[column_letter].width = min(max_length + 2, 50)
            print(f"  💾 Excel report saved to: {excel_path}")
        except Exception as e:
            print(f"  ⚠️ Could not create Excel: {e}")
            csv_path = os.path.join(vnc_output_folder, "vnc_distance_report.csv")
            report_df.to_csv(csv_path, index=False)
            print(f"  💾 CSV saved to: {csv_path}")

# Print final summary
print(f"\n{'='*60}")
print(f"🎉 Processing complete!")
print(f"   Total files processed: {len(processed_files)}/{len(all_label_files)}")
print(f"   Successful: {successful_count}")
print(f"   Failed: {failed_count}")

if failed_count > 0:
    print(f"\n⚠️  Failed files:")
    failed_files = [r for r in report_data if not r.get('success', False)] if report_data else []
    for ff in failed_files:
        print(f"   - {ff.get('filename', ff.get('error', 'unknown'))}")

# Clean up checkpoint if all files processed
if len(processed_files) == len(all_label_files):
    print(f"\n🧹 Removing checkpoint (all files processed)...")
    try:
        os.remove(checkpoint_file)
    except:
        pass
else:
    print(f"\n⚠️  Note: Only {len(processed_files)}/{len(all_label_files)} files were processed.")
    print(f"   Checkpoint saved at: {checkpoint_file}")
    print(f"   Run the script again to resume processing.")

print(f"\n📁 VNC masks saved to: {vnc_output_folder}")
print(f"📁 Distance reports saved to same folder:")
print(f"   - vnc_distance_report.xlsx (formatted Excel)")
print(f"   - vnc_distance_report.csv (simple CSV)")
print(f"\n📌 Manual selections saved to: {manual_selection_file}")