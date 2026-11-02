import slicer, qt, os, sys, subprocess
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = r"./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Install packages ──────────────────────────────────────
try:
    import nibabel
    from scipy import ndimage
except ImportError:
    subprocess.run([sys.executable, "-m", "pip",
                    "install", "nibabel", "scipy"])

from full_pipeline import run_full_pipeline


# ── Get current case name ─────────────────────────────────
def get_case_name():
    vol_nodes = slicer.util.getNodesByClass(
        'vtkMRMLScalarVolumeNode')
    if not vol_nodes:
        return None
    name = vol_nodes[0].GetName()
    return name.replace('.nii.gz','').replace('.nii','')


# ── Auto save tumor mask ──────────────────────────────────
def auto_save_tumor_mask(case_name):
    seg_nodes = slicer.util.getNodesByClass(
        'vtkMRMLSegmentationNode')
    if not seg_nodes:
        print("  ✗ No segmentation found")
        return None

    seg_node  = seg_nodes[-1]
    nrrd_path = os.path.join(RESULTS_DIR,
                             f"{case_name}_seg.nrrd")
    nii_path  = os.path.join(RESULTS_DIR,
                f"{case_name}_tumor_mask.nii.gz")

    # Save as nrrd
    slicer.util.saveNode(seg_node, nrrd_path)
    print(f"  ✓ Saved: {nrrd_path}")

    # Convert to nii.gz
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(nrrd_path)
        sitk.WriteImage(img, nii_path)
        print(f"  ✓ Converted: {nii_path}")
        return nii_path
    except Exception as e:
        print(f"  ✗ Conversion error: {e}")
        return None


# ── PRIORITY: Setup 3D display immediately ───────────────
def setup_3d_display():
    try:
        seg_nodes = slicer.util.getNodesByClass(
            'vtkMRMLSegmentationNode')
        if not seg_nodes:
            return

        # Find MONAILabel node (not TS segments)
        monai_node = None
        for n in seg_nodes:
            if 'liver_segment' not in n.GetName().lower():
                monai_node = n
                break
        if not monai_node:
            monai_node = seg_nodes[-1]

        seg          = monai_node.GetSegmentation()
        display_node = monai_node.GetDisplayNode()
        if not display_node:
            return

        # Create 3D surface
        monai_node.CreateClosedSurfaceRepresentation()
        display_node.SetVisibility3D(True)
        display_node.SetVisibility2DFill(True)

        # Liver = orange transparent
        liver_id = seg.GetSegmentIdBySegmentName("liver")
        if liver_id:
            display_node.SetSegmentOpacity3D(liver_id, 0.4)
            seg.GetSegment(liver_id).SetColor(
                0.88, 0.60, 0.35)

        # Tumor = red solid
        tumor_id = seg.GetSegmentIdBySegmentName("tumor")
        if tumor_id:
            display_node.SetSegmentOpacity3D(tumor_id, 1.0)
            seg.GetSegment(tumor_id).SetColor(
                0.9, 0.1, 0.1)

        # Switch to 4-up view + reset camera
        lm = slicer.app.layoutManager()
        lm.setLayout(
            slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        td = lm.threeDWidget(0)
        if td:
            td.threeDView().resetFocalPoint()

        print("  ✓ 3D display ready!")

    except Exception as e:
        print(f"  Display error: {e}")


# ── Load Couinaud segments ────────────────────────────────
def load_couinaud_segments(case_name):
    segments_dir = os.path.join(
        RESULTS_DIR, f"{case_name}_segments")
    if not os.path.exists(segments_dir):
        print(f"  ✗ Segments not found")
        return

    colors = {
        1: (0.2, 0.8, 0.8),
        2: (0.8, 0.4, 0.0),
        3: (0.2, 0.6, 0.2),
        4: (0.6, 0.2, 0.8),
        5: (0.8, 0.8, 0.2),
        6: (0.2, 0.4, 0.8),
        7: (0.8, 0.2, 0.4),
        8: (0.4, 0.8, 0.4),
    }

    for i in range(1, 9):
        seg_path = os.path.join(
            segments_dir,
            f"liver_segment_{i}.nii.gz")
        if not os.path.exists(seg_path):
            continue
        try:
            node         = slicer.util.loadSegmentation(
                seg_path)
            display_node = node.GetDisplayNode()
            seg          = node.GetSegmentation()
            seg_id       = seg.GetNthSegmentID(0)
            color        = colors.get(i, (0.5, 0.5, 0.5))
            seg.GetSegment(seg_id).SetColor(*color)
            display_node.SetSegmentOpacity3D(seg_id, 0.6)
            display_node.SetVisibility3D(1)
            display_node.SetVisibility(0)
            node.CreateClosedSurfaceRepresentation()
        except Exception as e:
            print(f"  Segment {i} error: {e}")

    print(f"  ✓ 8 segments loaded")


# ── Show results popup ────────────────────────────────────
def show_results_popup(r):
    seg       = r.get('tumor_segment', 'N/A')
    segs      = r.get('segments', {})
    seg_table = ""
    for sid, vol in sorted(segs.items()):
        marker    = " <- TUMOR" if sid == seg else ""
        seg_table += (f"  Segment {sid}:  "
                      f"{vol:>8.1f} mL{marker}\n")
    msg = f"""
{'='*50}
  LIVER ANALYSIS REPORT
  Case: {r['case_name']}
{'='*50}

VOLUMES (V4 Model):
  Liver Volume:      {r['liver_vol_ml']:>8.1f} mL
  Tumor Volume:      {r['tumor_vol_ml']:>8.2f} mL
  Tumor/Liver Ratio: {r['ratio_pct']:>8.2f} %

TUMOR CHARACTERISTICS:
  Number of nodules: {r['num_nodules']:>8}
  Largest diameter:  {r['diameter_cm']:>8.1f} cm

SIMPLE FLR (tumor only removed):
  FLR Volume:        {r['simple_flr_ml']:>8.1f} mL
  FLR%:              {r['simple_flr_pct']:>8.1f} %
  Status:            {'SAFE' if r['simple_safe'] else 'UNSAFE'}
  Threshold:         > 25% (normal liver)

COUINAUD SEGMENTS (TotalSegmentator):
{seg_table}
SEGMENT FLR (if segment {seg} removed):
  FLR Volume:        {r['seg_flr_ml']:>8.1f} mL
  FLR%:              {r['seg_flr_pct']:>8.1f} %
  Status:            {'SAFE' if r['seg_safe'] else 'UNSAFE'}
  Threshold:         > 25% (normal liver)

{'='*50}
NOTE: Clinical interpretation required
      by treating hepatobiliary surgeon
References:
  FLR: Makuuchi et al. Surgery 1989
  Couinaud: Couinaud C. 1957
{'='*50}
"""
    slicer.util.infoDisplay(
        msg, windowTitle="Liver Analysis Report")


# ── MAIN PIPELINE ─────────────────────────────────────────
def run_analysis_from_slicer():
    case_name = get_case_name()
    if not case_name:
        slicer.util.errorDisplay(
            "No CT scan loaded!")
        return

    print(f"\n{'='*50}")
    print(f"FULL ANALYSIS: {case_name}")
    print(f"{'='*50}")

    # Progress dialog
    progress = qt.QProgressDialog(
        "Starting analysis...",
        None, 0, 5)
    progress.setWindowTitle("Liver Analysis")
    progress.setMinimumWidth(400)
    progress.setMinimumDuration(0)
    progress.show()
    slicer.app.processEvents()

    try:
        # ── Step 1: Save mask ─────────────────────────────
        progress.setLabelText(
            "Step 1/5: Saving tumor mask...")
        progress.setValue(1)
        slicer.app.processEvents()
        print("\n[1/5] Saving tumor mask...")
        tumor_mask = auto_save_tumor_mask(case_name)
        if not tumor_mask:
            progress.close()
            slicer.util.errorDisplay(
                "Could not save tumor mask!\n"
                "Run MONAILabel inference first.")
            return

        # ── Step 2: 3D display FIRST ──────────────────────
        progress.setLabelText(
            "Step 2/5: Setting up 3D display...")
        progress.setValue(2)
        slicer.app.processEvents()
        print("\n[2/5] Setting up 3D display...")
        setup_3d_display()
        slicer.app.processEvents()

        # ── Step 3: TotalSegmentator ──────────────────────
        progress.setLabelText(
            "Step 3/5: Running TotalSegmentator (~1 min)...")
        progress.setValue(3)
        slicer.app.processEvents()
        print("\n[3/5] Running pipeline...")
        results = run_full_pipeline(case_name, tumor_mask)
        if results is None:
            progress.close()
            slicer.util.errorDisplay("Pipeline failed!")
            return

        # ── Step 4: Load segments ─────────────────────────
        progress.setLabelText(
            "Step 4/5: Loading Couinaud segments...")
        progress.setValue(4)
        slicer.app.processEvents()
        print("\n[4/5] Loading segments...")
        load_couinaud_segments(case_name)

        # ── Step 5: Show popup ────────────────────────────
        progress.setLabelText(
            "Step 5/5: Generating report...")
        progress.setValue(5)
        slicer.app.processEvents()
        print("\n[5/5] Showing report...")
        progress.close()
        show_results_popup(results)

    except Exception as e:
        progress.close()
        import traceback
        print(traceback.format_exc())
        slicer.util.errorDisplay(f"Error: {str(e)}")


# ── View toggles ──────────────────────────────────────────
def show_option_a():
    nodes = slicer.util.getNodesByClass(
        'vtkMRMLSegmentationNode')
    for node in nodes:
        display = node.GetDisplayNode()
        if 'liver_segment' in node.GetName().lower():
            display.SetVisibility(0)
        else:
            display.SetVisibility(1)
            node.CreateClosedSurfaceRepresentation()
    lm = slicer.app.layoutManager()
    td = lm.threeDWidget(0)
    if td:
        td.threeDView().resetFocalPoint()
    print("✓ View: Liver + Tumor")


def show_option_b():
    nodes = slicer.util.getNodesByClass(
        'vtkMRMLSegmentationNode')
    for node in nodes:
        display = node.GetDisplayNode()
        display.SetVisibility(1)
        node.CreateClosedSurfaceRepresentation()
    lm = slicer.app.layoutManager()
    td = lm.threeDWidget(0)
    if td:
        td.threeDView().resetFocalPoint()
    print("✓ View: All Segments")


# ── Add buttons ───────────────────────────────────────────
def add_buttons():
    toolbar = slicer.util.mainWindow().findChild(
        qt.QToolBar, "MainToolBar")

    btn_main = qt.QPushButton("Full Liver Analysis")
    btn_main.setToolTip(
        "Complete automated pipeline")
    btn_main.setStyleSheet("""
        QPushButton {
            background-color: #1F4E79;
            color: white;
            font-size: 13px;
            font-weight: bold;
            padding: 8px 16px;
            border-radius: 5px;
            margin: 2px;
        }
        QPushButton:hover {
            background-color: #2E75B6;
        }
    """)
    btn_main.clicked.connect(run_analysis_from_slicer)

    btn_a = qt.QPushButton("Liver+Tumor")
    btn_a.setToolTip("Show liver and tumor only")
    btn_a.setStyleSheet("""
        QPushButton {
            background-color: #833C00;
            color: white;
            font-weight: bold;
            padding: 6px 12px;
            border-radius: 4px;
            margin: 2px;
        }
        QPushButton:hover {
            background-color: #C55A11;
        }
    """)
    btn_a.clicked.connect(show_option_a)

    btn_b = qt.QPushButton("All Segments")
    btn_b.setToolTip("Show all 8 Couinaud segments")
    btn_b.setStyleSheet("""
        QPushButton {
            background-color: #375623;
            color: white;
            font-weight: bold;
            padding: 6px 12px;
            border-radius: 4px;
            margin: 2px;
        }
        QPushButton:hover {
            background-color: #548235;
        }
    """)
    btn_b.clicked.connect(show_option_b)

    if toolbar:
        toolbar.addSeparator()
        toolbar.addWidget(btn_main)
        toolbar.addWidget(btn_a)
        toolbar.addWidget(btn_b)
        print("✓ 3 buttons added!")
    else:
        new_tb = qt.QToolBar("Liver Analysis")
        new_tb.addWidget(btn_main)
        new_tb.addWidget(btn_a)
        new_tb.addWidget(btn_b)
        slicer.util.mainWindow().addToolBar(new_tb)
        print("✓ Buttons added to new toolbar!")


# ── Run ───────────────────────────────────────────────────
add_buttons()
print("\n✓ System Ready!")
print("="*45)
print("WORKFLOW:")
print("  1. Load CT scan")
print("  2. MONAILabel → Run")
print("  3. Click 'Full Liver Analysis'")
print("     Step 1: Auto save mask")
print("     Step 2: 3D display immediately ← PRIORITY")
print("     Step 3: TotalSegmentator + FLR")
print("     Step 4: Load segments")
print("     Step 5: Show report")
print("VIEWS:")
print("  'Liver+Tumor'  → clean 3D view")
print("  'All Segments' → 8 Couinaud segments")
print("="*45)
