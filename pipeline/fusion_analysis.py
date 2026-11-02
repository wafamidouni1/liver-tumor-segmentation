import nibabel as nib
import numpy as np
from scipy import ndimage
import os

# ── Paths ─────────────────────────────────────────────────
TUMOR_MASK  = r"./results\liver_14_tumor_mask.nii.gz"
SEGMENTS_DIR = r"./results\liver_14_segments"
CT_PATH     = r"./data\images\liver_14.nii.gz"

print("=" * 55)
print("LIVER ANALYSIS REPORT")
print("=" * 55)

# ── Load V4 tumor mask ────────────────────────────────────
print("\nLoading V4 segmentation...")
v4_nib   = nib.load(TUMOR_MASK)
v4_data  = v4_nib.get_fdata()
spacing  = v4_nib.header.get_zooms()
voxel_vol = float(spacing[0]) * \
            float(spacing[1]) * \
            float(spacing[2])

print(f"  Voxel size: {spacing[0]:.2f} x "
      f"{spacing[1]:.2f} x {spacing[2]:.2f} mm")
print(f"  Voxel volume: {voxel_vol:.2f} mm3")

# ── Calculate liver and tumor volumes ─────────────────────
liver_voxels = int((v4_data == 1).sum())
tumor_voxels = int((v4_data == 2).sum())
liver_vol_ml = liver_voxels * voxel_vol / 1000
tumor_vol_ml = tumor_voxels * voxel_vol / 1000
ratio        = (tumor_vol_ml/liver_vol_ml*100
                if liver_vol_ml > 0 else 0)

print(f"\n{'='*55}")
print(f"VOLUMES (from V4 model):")
print(f"{'='*55}")
print(f"  Liver volume:    {liver_vol_ml:>10.1f} mL")
print(f"  Tumor volume:    {tumor_vol_ml:>10.2f} mL")
print(f"  Tumor/Liver:     {ratio:>10.2f} %")

# ── Simple FLR ────────────────────────────────────────────
simple_flr    = liver_vol_ml - tumor_vol_ml
simple_flr_pct = (simple_flr/liver_vol_ml*100
                  if liver_vol_ml > 0 else 0)
flr_safe      = simple_flr_pct >= 25

print(f"\n{'='*55}")
print(f"SIMPLE FLR (tumor removed only):")
print(f"{'='*55}")
print(f"  FLR volume:      {simple_flr:>10.1f} mL")
print(f"  FLR%:            {simple_flr_pct:>10.1f} %")
print(f"  Safe threshold:  {'':>10} > 25%")
print(f"  Status:          {'':>10} "
      f"{'SAFE ✓' if flr_safe else 'UNSAFE ✗'}")

# ── Tumor nodules and diameter ────────────────────────────
tumor_mask         = (v4_data == 2).astype(int)
labeled, num_nod   = ndimage.label(tumor_mask)

# Largest tumor diameter
if tumor_voxels > 0:
    coords   = np.where(tumor_mask)
    x_size   = (coords[0].max()-coords[0].min()) * spacing[0]
    y_size   = (coords[1].max()-coords[1].min()) * spacing[1]
    z_size   = (coords[2].max()-coords[2].min()) * spacing[2]
    diameter = max(x_size, y_size, z_size) / 10  # cm
else:
    diameter = 0

print(f"\n{'='*55}")
print(f"TUMOR CHARACTERISTICS:")
print(f"{'='*55}")
print(f"  Number of nodules: {num_nod:>8}")
print(f"  Largest diameter:  {diameter:>8.1f} cm")

# ── Load Couinaud segments ────────────────────────────────
print(f"\n{'='*55}")
print(f"COUINAUD SEGMENT ANALYSIS (TotalSegmentator):")
print(f"{'='*55}")

segment_volumes = {}
tumor_in_segments = {}

for seg_id in range(1, 9):
    seg_path = os.path.join(
        SEGMENTS_DIR,
        f"liver_segment_{seg_id}.nii.gz"
    )
    if not os.path.exists(seg_path):
        print(f"  Segment {seg_id}: file not found")
        continue

    seg_data = nib.load(seg_path).get_fdata()

    # Volume of this segment
    seg_voxels = int((seg_data > 0).sum())
    seg_vol_ml = seg_voxels * voxel_vol / 1000
    segment_volumes[seg_id] = seg_vol_ml

    # Check if tumor overlaps with this segment
    if tumor_voxels > 0:
        # Resize if needed
        if seg_data.shape != v4_data.shape:
            from scipy.ndimage import zoom
            scale = [v4_data.shape[i]/seg_data.shape[i]
                     for i in range(3)]
            seg_data = zoom(seg_data, scale, order=0)

        overlap = int(
            ((seg_data > 0) & (v4_data == 2)).sum()
        )
        if overlap > 0:
            tumor_in_segments[seg_id] = overlap

    print(f"  Segment {seg_id}:  {seg_vol_ml:>8.1f} mL")

# ── Identify main tumor segment ───────────────────────────
if tumor_in_segments:
    main_seg = max(tumor_in_segments,
                   key=tumor_in_segments.get)
    main_seg_vol = segment_volumes.get(main_seg, 0)

    # Segment FLR
    seg_flr     = liver_vol_ml - main_seg_vol
    seg_flr_pct = (seg_flr/liver_vol_ml*100
                   if liver_vol_ml > 0 else 0)
    seg_safe    = seg_flr_pct >= 25

    print(f"\n{'='*55}")
    print(f"TUMOR LOCATION:")
    print(f"{'='*55}")
    print(f"  Tumor in segment:  {main_seg:>8}")
    print(f"  Segment volume:    {main_seg_vol:>8.1f} mL")
    print(f"\nSEGMENT FLR "
          f"(if segment {main_seg} removed):")
    print(f"  FLR volume:        {seg_flr:>8.1f} mL")
    print(f"  FLR%:              {seg_flr_pct:>8.1f} %")
    print(f"  Safe threshold:    {'':>8} > 25%")
    print(f"  Status:            {'':>8} "
          f"{'SAFE ✓' if seg_safe else 'UNSAFE ✗'}")
else:
    print(f"\n  No tumor detected in segments")

# ── Final summary ─────────────────────────────────────────
print(f"\n{'='*55}")
print(f"FINAL SUMMARY")
print(f"{'='*55}")
print(f"  Liver:         {liver_vol_ml:.1f} mL "
      f"(normal: 1200-1500 mL)")
print(f"  Tumor:         {tumor_vol_ml:.2f} mL")
print(f"  Ratio:         {ratio:.2f}%")
print(f"  Simple FLR%:   {simple_flr_pct:.1f}% "
      f"{'✓' if flr_safe else '✗'}")
if tumor_in_segments:
    print(f"  Tumor segment: {main_seg}")
    print(f"  Segment FLR%:  {seg_flr_pct:.1f}% "
          f"{'✓' if seg_safe else '✗'}")
print(f"\n  NOTE: Clinical interpretation required")
print(f"        by treating hepatobiliary surgeon")
print(f"{'='*55}")