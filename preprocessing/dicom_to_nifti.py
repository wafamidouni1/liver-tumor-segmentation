import SimpleITK as sitk
import os
import sys

def convert_dicom_to_nifti(dicom_folder, output_name):
    """
    Convert DICOM folder to NIfTI .nii.gz
    
    dicom_folder: path to folder containing .dcm files
    output_name:  name for output file (without extension)
    """
    OUTPUT_DIR = r"./data\images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR, f"{output_name}.nii.gz")

    print(f"Reading DICOM from: {dicom_folder}")
    print(f"Output: {output_path}")

    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(
        dicom_folder)

    if not dicom_names:
        print("No DICOM files found!")
        print("Make sure folder contains .dcm files")
        return False

    print(f"Found {len(dicom_names)} DICOM slices")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    # Print image info
    size    = image.GetSize()
    spacing = image.GetSpacing()
    print(f"Image size:    {size}")
    print(f"Voxel spacing: {spacing[0]:.2f} x "
          f"{spacing[1]:.2f} x {spacing[2]:.2f} mm")

    # Save as NIfTI
    sitk.WriteImage(image, output_path)
    file_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved: {output_path} ({file_mb:.1f} MB)")
    print("Conversion complete!")
    return True

# ── Run ───────────────────────────────────────────────────
if __name__ == "__main__":

    # CHANGE THESE TWO LINES:
    DICOM_FOLDER = r"./Downloads\PATIENT_DICOM5"
    OUTPUT_NAME  = "cas_clinique_05"

    convert_dicom_to_nifti(DICOM_FOLDER, OUTPUT_NAME)