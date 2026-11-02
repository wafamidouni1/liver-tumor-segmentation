import os
import subprocess
import nibabel as nib
import numpy as np
from scipy import ndimage

def run_full_analysis(ct_path, output_dir, tumor_mask):

    case_name    = os.path.basename(ct_path).replace('.nii.gz','')
    segments_dir = os.path.join(output_dir, f"{case_name}_segments")
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: TotalSegmentator
    print(f"\n[1/3] Running TotalSegmentator...")
    totalseg_exe = r"./totalseg_env\Scripts\TotalSegmentator.exe"

    result = subprocess.run([
        totalseg_exe,
        "-i", ct_path,
        "-o", segments_dir,
        "--task", "liver_segments",
        "--nr_thr_resamp", "1",
        "--nr_thr_saving", "1"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"TotalSegmentator error: {result.stderr}")
        return None
    print(f"  Segments saved to: {segments_dir}")

    # Step 2: Load V4 tumor mask
    print(f"\n[2/3] Loading V4 segmentation...")
    if not os.path.exists(tumor_mask):
        print(f"  Tumor mask not found: {tumor_mask}")
        return None

    v4_nib    = nib.load(tumor_mask)
    v4_data   = v4_nib.get_fdata()
    spacing   = v4_nib.header.get_zooms()
    voxel_vol = float(spacing[0]) * float(spacing[1]) * float(spacing[2])

    liver_voxels  = int((v4_data == 1).sum())
    tumor_voxels  = int((v4_data == 2).sum())
    liver_vol_ml  = liver_voxels * voxel_vol / 1000
    tumor_vol_ml  = tumor_voxels * voxel_vol / 1000
    ratio         = (tumor_vol_ml / liver_vol_ml * 100 if liver_vol_ml > 0 else 0)
    simple_flr    = liver_vol_ml - tumor_vol_ml
    simple_flr_pct = (simple_flr / liver_vol_ml * 100 if liver_vol_ml > 0 else 0)

    tumor_mask_arr = (v4_data == 2).astype(int)
    labeled, num_nod = ndimage.label(tumor_mask_arr)

    if tumor_voxels > 0:
        coords   = np.where(tumor_mask_arr)
        x_size   = (coords[0].max() - coords[0].min()) * spacing[0]
        y_size   = (coords[1].max() - coords[1].min()) * spacing[1]
        z_size   = (coords[2].max() - coords[2].min()) * spacing[2]
        diameter = max(x_size, y_size, z_size) / 10
    else:
        diameter = 0

    # Step 3: Segment analysis
    print(f"\n[3/3] Running fusion analysis...")
    segment_volumes   = {}
    tumor_in_segments = {}

    for seg_id in range(1, 9):
        seg_path = os.path.join(segments_dir, f"liver_segment_{seg_id}.nii.gz")
        if not os.path.exists(seg_path):
            continue
        seg_data   = nib.load(seg_path).get_fdata()
        seg_voxels = int((seg_data > 0).sum())
        seg_vol_ml = seg_voxels * voxel_vol / 1000
        segment_volumes[seg_id] = seg_vol_ml

        if tumor_voxels > 0:
            if seg_data.shape != v4_data.shape:
                from scipy.ndimage import zoom
                scale    = [v4_data.shape[i] / seg_data.shape[i] for i in range(3)]
                seg_data = zoom(seg_data, scale, order=0)
            overlap = int(((seg_data > 0) & (v4_data == 2)).sum())
            if overlap > 0:
                tumor_in_segments[seg_id] = overlap

    main_seg    = None
    seg_flr     = 0
    seg_flr_pct = 0

    if tumor_in_segments:
        main_seg     = max(tumor_in_segments, key=tumor_in_segments.get)
        main_seg_vol = segment_volumes.get(main_seg, 0)
        seg_flr      = liver_vol_ml - main_seg_vol
        seg_flr_pct  = (seg_flr / liver_vol_ml * 100 if liver_vol_ml > 0 else 0)

    results = {
        "case_name":      case_name,
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

    print_results(results)
    return results


def print_results(r):
    print(f"\n{'='*55}")
    print(f"LIVER ANALYSIS REPORT — {r['case_name']}")
    print(f"{'='*55}")
    print(f"VOLUMES:")
    print(f"  Liver:         {r['liver_vol_ml']:>8.1f} mL")
    print(f"  Tumor:         {r['tumor_vol_ml']:>8.2f} mL")
    print(f"  Ratio:         {r['ratio_pct']:>8.2f} %")
    print(f"\nSIMPLE FLR:")
    print(f"  Volume:        {r['simple_flr_ml']:>8.1f} mL")
    print(f"  FLR%:          {r['simple_flr_pct']:>8.1f} %")
    print(f"  Status:        {'SAFE' if r['simple_safe'] else 'UNSAFE'}")
    print(f"\nTUMOR:")
    print(f"  Nodules:       {r['num_nodules']:>8}")
    print(f"  Diameter:      {r['diameter_cm']:>8.1f} cm")
    if r['tumor_segment']:
        print(f"\nSEGMENT FLR:")
        print(f"  Tumor in:      Segment {r['tumor_segment']}")
        print(f"  FLR volume:    {r['seg_flr_ml']:>8.1f} mL")
        print(f"  FLR%:          {r['seg_flr_pct']:>8.1f} %")
        print(f"  Status:        {'SAFE' if r['seg_safe'] else 'UNSAFE'}")
    print(f"\n  NOTE: Clinical interpretation required")
    print(f"{'='*55}")


if __name__ == "__main__":
    CT_PATH      = r"./data\images\liver_14.nii.gz"
    OUTPUT_DIR   = r"./results"
    TUMOR_MASK   = r"./results\liver_14_tumor_mask.nii.gz"
    SEGMENTS_DIR = r"./results\liver_14_segments"

    # Skip TotalSegmentator if segments already exist
    case_name = "liver_14"
    segments_exist = all([
        os.path.exists(os.path.join(
            SEGMENTS_DIR, f"liver_segment_{i}.nii.gz"))
        for i in range(1, 9)
    ])

    if segments_exist:
        print("[1/3] Segments already exist — skipping TotalSegmentator")
        
        # Go straight to analysis
        print(f"\n[2/3] Loading V4 segmentation...")
        v4_nib    = nib.load(TUMOR_MASK)
        v4_data   = v4_nib.get_fdata()
        spacing   = v4_nib.header.get_zooms()
        voxel_vol = float(spacing[0]) * \
                    float(spacing[1]) * \
                    float(spacing[2])

        liver_voxels   = int((v4_data == 1).sum())
        tumor_voxels   = int((v4_data == 2).sum())
        liver_vol_ml   = liver_voxels * voxel_vol / 1000
        tumor_vol_ml   = tumor_voxels * voxel_vol / 1000
        ratio          = (tumor_vol_ml/liver_vol_ml*100
                          if liver_vol_ml > 0 else 0)
        simple_flr     = liver_vol_ml - tumor_vol_ml
        simple_flr_pct = (simple_flr/liver_vol_ml*100
                          if liver_vol_ml > 0 else 0)

        tumor_mask_arr   = (v4_data == 2).astype(int)
        labeled, num_nod = ndimage.label(tumor_mask_arr)

        if tumor_voxels > 0:
            coords   = np.where(tumor_mask_arr)
            x_size   = (coords[0].max()-coords[0].min()) \
                       * spacing[0]
            y_size   = (coords[1].max()-coords[1].min()) \
                       * spacing[1]
            z_size   = (coords[2].max()-coords[2].min()) \
                       * spacing[2]
            diameter = max(x_size, y_size, z_size) / 10
        else:
            diameter = 0

        print(f"\n[3/3] Running fusion analysis...")
        segment_volumes   = {}
        tumor_in_segments = {}

        for seg_id in range(1, 9):
            seg_path = os.path.join(
                SEGMENTS_DIR,
                f"liver_segment_{seg_id}.nii.gz"
            )
            if not os.path.exists(seg_path):
                continue
            seg_data   = nib.load(seg_path).get_fdata()
            seg_voxels = int((seg_data > 0).sum())
            seg_vol_ml = seg_voxels * voxel_vol / 1000
            segment_volumes[seg_id] = seg_vol_ml

            if tumor_voxels > 0:
                if seg_data.shape != v4_data.shape:
                    from scipy.ndimage import zoom
                    scale    = [v4_data.shape[i]/seg_data.shape[i]
                                for i in range(3)]
                    seg_data = zoom(seg_data, scale, order=0)
                overlap = int(
                    ((seg_data > 0) & (v4_data == 2)).sum()
                )
                if overlap > 0:
                    tumor_in_segments[seg_id] = overlap

        main_seg    = None
        seg_flr     = 0
        seg_flr_pct = 0

        if tumor_in_segments:
            main_seg     = max(tumor_in_segments,
                               key=tumor_in_segments.get)
            main_seg_vol = segment_volumes.get(main_seg, 0)
            seg_flr      = liver_vol_ml - main_seg_vol
            seg_flr_pct  = (seg_flr/liver_vol_ml*100
                            if liver_vol_ml > 0 else 0)

        results = {
            "case_name":      case_name,
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
        print_results(results)
    else:
        run_full_analysis(CT_PATH, OUTPUT_DIR, TUMOR_MASK)