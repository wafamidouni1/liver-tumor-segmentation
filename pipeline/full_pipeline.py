import os
import subprocess
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom

# ── Configuration ─────────────────────────────────────────
TOTALSEG_EXE  = (r"./totalseg_env"
                 r"\Scripts\TotalSegmentator.exe")
DATA_DIR      = r"./data\images"
RESULTS_DIR   = r"./results"
RESIZE_FACTOR = 0.5


def resize_ct(input_path, output_path, factor=0.5):
    """
    Resize CT for TotalSegmentator only
    Uses nibabel — no SimpleITK conflict with Slicer
    V4 model always uses ORIGINAL full resolution
    """
    print(f"  Loading CT...")
    img    = nib.load(input_path)
    data   = img.get_fdata()
    affine = img.affine.copy()

    print(f"  Original shape: {data.shape}")
    zoomed = zoom(data, factor, order=1)
    print(f"  Resized shape:  {zoomed.shape}")

    new_affine = affine.copy()
    for i in range(3):
        new_affine[i, i] = affine[i, i] / factor

    new_img = nib.Nifti1Image(
        zoomed.astype(np.float32), new_affine)
    nib.save(new_img, output_path)
    print(f"  Saved: {output_path}")
    return output_path


def run_totalsegmentator(small_ct_path, segments_dir):
    """
    Run TotalSegmentator in isolated environment
    Prevents conflict with Slicer Python
    """
    os.makedirs(segments_dir, exist_ok=True)

    import os as _os
    env = dict(_os.environ)
    env["PYTHONPATH"] = ""
    env["PYTHONHOME"] = ""
    env["PATH"] = (
        r"./totalseg_env\Scripts;" +
        env.get("PATH", "")
    )

    result = subprocess.run([
        TOTALSEG_EXE,
        "-i", small_ct_path,
        "-o", segments_dir,
        "--task", "liver_segments",
        "--nr_thr_resamp", "1",
        "--nr_thr_saving", "1"
    ], capture_output=True, text=True, env=env)

    if result.returncode == 0:
        files = [f for f in os.listdir(segments_dir)
                 if f.endswith('.nii.gz')]
        print(f"  {len(files)}/8 segments saved")
        return True
    else:
        print(f"  Error: {result.stderr[-500:]}")
        return False


def run_fusion(tumor_mask_path, segments_dir):
    """
    Fuse V4 tumor mask with TotalSegmentator segments
    Calculate all volumes and FLR
    """
    # Load V4 mask
    v4_nib    = nib.load(tumor_mask_path)
    v4_data   = v4_nib.get_fdata()
    spacing   = v4_nib.header.get_zooms()
    voxel_vol = (float(spacing[0]) *
                 float(spacing[1]) *
                 float(spacing[2]))

    # Volumes from V4
    liver_voxels   = int((v4_data == 1).sum())
    tumor_voxels   = int((v4_data == 2).sum())
    liver_vol_ml   = liver_voxels * voxel_vol / 1000
    tumor_vol_ml   = tumor_voxels * voxel_vol / 1000
    ratio          = (tumor_vol_ml / liver_vol_ml * 100
                      if liver_vol_ml > 0 else 0)
    simple_flr     = liver_vol_ml - tumor_vol_ml
    simple_flr_pct = (simple_flr / liver_vol_ml * 100
                      if liver_vol_ml > 0 else 0)

    # Tumor characteristics
    tumor_arr        = (v4_data == 2).astype(int)
    labeled, num_nod = ndimage.label(tumor_arr)

    if tumor_voxels > 0:
        coords   = np.where(tumor_arr)
        diameter = max(
            (coords[0].max()-coords[0].min())*spacing[0],
            (coords[1].max()-coords[1].min())*spacing[1],
            (coords[2].max()-coords[2].min())*spacing[2]
        ) / 10
    else:
        diameter = 0

    # Segment analysis
    segment_volumes   = {}
    tumor_in_segments = {}

    for seg_id in range(1, 9):
        seg_path = os.path.join(
            segments_dir,
            f"liver_segment_{seg_id}.nii.gz"
        )
        if not os.path.exists(seg_path):
            continue

        seg_nib    = nib.load(seg_path)
        seg_data   = seg_nib.get_fdata()
        seg_sp     = seg_nib.header.get_zooms()
        seg_vox    = (float(seg_sp[0]) *
                      float(seg_sp[1]) *
                      float(seg_sp[2]))
        seg_voxels = int((seg_data > 0).sum())
        seg_vol_ml = seg_voxels * seg_vox / 1000
        segment_volumes[seg_id] = seg_vol_ml

        if tumor_voxels > 0:
            if seg_data.shape != v4_data.shape:
                scale    = [v4_data.shape[i] /
                            seg_data.shape[i]
                            for i in range(3)]
                seg_data = zoom(seg_data, scale, order=0)
            overlap = int(
                ((seg_data > 0) & (v4_data == 2)).sum()
            )
            if overlap > 0:
                tumor_in_segments[seg_id] = overlap

    # Main tumor segment + FLR
    main_seg    = None
    seg_flr     = 0
    seg_flr_pct = 0

    if tumor_in_segments:
        main_seg     = max(tumor_in_segments,
                           key=tumor_in_segments.get)
        main_seg_vol = segment_volumes.get(main_seg, 0)
        seg_flr      = liver_vol_ml - main_seg_vol
        seg_flr_pct  = (seg_flr / liver_vol_ml * 100
                        if liver_vol_ml > 0 else 0)

    return {
        "liver_vol_ml":   round(liver_vol_ml, 1),
        "tumor_vol_ml":   round(tumor_vol_ml, 2),
        "ratio_pct":      round(ratio, 2),
        "simple_flr_ml":  round(simple_flr, 1),
        "simple_flr_pct": round(simple_flr_pct, 1),
        "num_nodules":    num_nod,
        "diameter_cm":    round(diameter, 1),
        "segments":       segment_volumes,
        "tumor_segment":  main_seg,
        "seg_flr_ml":     round(seg_flr, 1),
        "seg_flr_pct":    round(seg_flr_pct, 1),
        "simple_safe":    simple_flr_pct >= 25,
        "seg_safe":       seg_flr_pct >= 25,
    }


def print_results(r):
    """Print formatted results to console"""
    seg  = r.get('tumor_segment', 'N/A')
    segs = r.get('segments', {})

    print(f"\n{'='*55}")
    print(f"LIVER ANALYSIS REPORT — {r['case_name']}")
    print(f"{'='*55}")
    print(f"VOLUMES (V4 Model):")
    print(f"  Liver:         {r['liver_vol_ml']:>8.1f} mL")
    print(f"  Tumor:         {r['tumor_vol_ml']:>8.2f} mL")
    print(f"  Ratio:         {r['ratio_pct']:>8.2f} %")
    print(f"\nSIMPLE FLR:")
    print(f"  Volume:        {r['simple_flr_ml']:>8.1f} mL")
    print(f"  FLR%:          {r['simple_flr_pct']:>8.1f} %")
    print(f"  Status:        "
          f"{'SAFE' if r['simple_safe'] else 'UNSAFE'}")
    print(f"\nTUMOR:")
    print(f"  Nodules:       {r['num_nodules']:>8}")
    print(f"  Diameter:      {r['diameter_cm']:>8.1f} cm")
    print(f"\nCOUINAUD SEGMENTS:")
    for sid, vol in sorted(segs.items()):
        marker = " <- TUMOR" if sid == seg else ""
        print(f"  Segment {sid}:    "
              f"{vol:>8.1f} mL{marker}")
    if seg:
        print(f"\nSEGMENT FLR "
              f"(if segment {seg} removed):")
        print(f"  Volume:        "
              f"{r['seg_flr_ml']:>8.1f} mL")
        print(f"  FLR%:          "
              f"{r['seg_flr_pct']:>8.1f} %")
        print(f"  Status:        "
              f"{'SAFE' if r['seg_safe'] else 'UNSAFE'}")
    print(f"\n  NOTE: Clinical interpretation required")
    print(f"{'='*55}")


def run_full_pipeline(case_name, tumor_mask_path):
    """
    Complete automated pipeline:
    1. Resize CT (for TotalSegmentator only)
    2. Run TotalSegmentator → 8 segments
    3. Fusion analysis → volumes + FLR
    Returns: results dictionary
    """
    print(f"\n{'='*55}")
    print(f"FULL PIPELINE — {case_name}")
    print(f"{'='*55}")

    # Paths
    ct_path      = os.path.join(
        DATA_DIR, f"{case_name}.nii.gz")
    small_ct     = os.path.join(
        RESULTS_DIR, f"{case_name}_small.nii.gz")
    segments_dir = os.path.join(
        RESULTS_DIR, f"{case_name}_segments")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Validate inputs
    if not os.path.exists(ct_path):
        print(f"  CT not found: {ct_path}")
        return None

    if not os.path.exists(tumor_mask_path):
        print(f"  Tumor mask not found: {tumor_mask_path}")
        return None

    # Step 1: Resize CT
    print(f"\n[1/3] Resizing CT for TotalSegmentator...")
    print(f"  (V4 uses original full resolution)")
    resize_ct(ct_path, small_ct, RESIZE_FACTOR)

    # Step 2: TotalSegmentator
    print(f"\n[2/3] Running TotalSegmentator...")
    seg_files = []
    if os.path.exists(segments_dir):
        seg_files = [f for f in os.listdir(segments_dir)
                     if f.endswith('.nii.gz')]

    if len(seg_files) == 8:
        print(f"  Segments already exist — skipping")
    else:
        success = run_totalsegmentator(
            small_ct, segments_dir)
        if not success:
            return None

    # Step 3: Fusion
    print(f"\n[3/3] Running fusion analysis...")
    results              = run_fusion(
        tumor_mask_path, segments_dir)
    results["case_name"] = case_name

    print_results(results)
    return results


if __name__ == "__main__":
    CASE_NAME  = "liver_14"
    TUMOR_MASK = (r"./results"
                  r"\liver_14_tumor_mask.nii.gz")
    run_full_pipeline(CASE_NAME, TUMOR_MASK)