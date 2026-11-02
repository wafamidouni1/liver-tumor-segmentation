import nibabel as nib
import os

INPUT_DIR  = r"./media\nas\01_Datasets\CT\LITS\Training Batch 1"
OUTPUT_IMG = r"./lits_b1_ready\images"
OUTPUT_LBL = r"./lits_b1_ready\labels"

os.makedirs(OUTPUT_IMG, exist_ok=True)
os.makedirs(OUTPUT_LBL, exist_ok=True)

volumes = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.startswith('volume')
    and f.endswith('.nii')])

print(f"Volumes trouves: {len(volumes)}")

for vol_file in volumes:
    num     = int(vol_file.replace(
        'volume-','').replace('.nii',''))
    out_name = f"lits_{num:03d}.nii.gz"
    seg_file = f"segmentation-{num}.nii"

    vol_path = os.path.join(INPUT_DIR, vol_file)
    seg_path = os.path.join(INPUT_DIR, seg_file)
    out_img  = os.path.join(OUTPUT_IMG, out_name)
    out_lbl  = os.path.join(OUTPUT_LBL, out_name)

    # Image
    print(f"  {vol_file} → {out_name}", end="")
    img = nib.load(vol_path)
    nib.save(img, out_img)

    # Label
    if os.path.exists(seg_path):
        lbl = nib.load(seg_path)
        nib.save(lbl, out_lbl)
        print(f" ✓")
    else:
        print(f" ✗ label manquant!")

print(f"\nImages: {len(os.listdir(OUTPUT_IMG))}")
print(f"Labels: {len(os.listdir(OUTPUT_LBL))}")
print(f"Prets a uploader!")