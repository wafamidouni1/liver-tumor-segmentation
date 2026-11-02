import logging
import os
from typing import Callable, Sequence

import numpy as np
import torch
import nibabel as nib

from monai.inferers import Inferer, SlidingWindowInferer
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    EnsureChannelFirstd,
    EnsureTyped,
    KeepLargestConnectedComponentd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    CropForegroundd,
    ResizeWithPadOrCropd,
)
from monailabel.interfaces.tasks.infer_v2 import InferType
from monailabel.tasks.infer.basic_infer import BasicInferTask
from monailabel.transform.post import Restored

logger = logging.getLogger(__name__)


class SegmentationLiver(BasicInferTask):
    """
    Liver and Tumor Segmentation using SwinUNETR V4
    Trained on Task03 Liver + LiTS datasets
    Liver Dice: 0.916 | Tumor Dice: 0.561
    """

    def __init__(
        self,
        path,
        network=None,
        target_spacing=(1.5, 1.5, 2.0),
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=3,
        description="Liver and Tumor Segmentation using SwinUNETR V4",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            load_strict=False,
            **kwargs,
        )
        self.target_spacing = target_spacing

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            # Step 1: Load CT scan
            LoadImaged(keys="image"),
            EnsureTyped(
                keys="image",
                device=data.get("device") if data else None
            ),
            EnsureChannelFirstd(keys="image"),

            # Step 2: Reorient to standard
            Orientationd(keys="image", axcodes="RAS"),

            # Step 3: Resample to training spacing
            Spacingd(
                keys="image",
                pixdim=self.target_spacing,
                mode="bilinear",
                allow_missing_keys=True
            ),

            # Step 4: Liver CT window
            # -200 to 250 HU = optimal for liver/tumor
            ScaleIntensityRanged(
                keys="image",
                a_min=-200,
                a_max=300,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),

            # Step 5: Crop to foreground
            CropForegroundd(
                keys="image",
                source_key="image"
            ),

        ]

    def inferer(self, data=None) -> Inferer:
        return SlidingWindowInferer(
            roi_size=(96, 96, 64),
            sw_batch_size=2,
            overlap=0.25,
        )

    def inverse_transforms(self, data=None):
        return []

    def post_transforms(self, data=None) -> Sequence[Callable]:
        return [
            EnsureTyped(
                keys="pred",
                device=data.get("device") if data else None
            ),
            # Get probabilities
            Activationsd(keys="pred", softmax=True),
            # Get class labels
            AsDiscreted(keys="pred", argmax=True),
            # Remove small false positives
            KeepLargestConnectedComponentd(
                keys="pred",
                applied_labels=[1]
            ),
            # Restore to original image space
            Restored(keys="pred", ref_image="image"),
        ]

    def writer(self, data, extension=None, dtype=None):
        """
        Override writer to also calculate and log volumes
        """
        result = super().writer(data, extension, dtype)

        # Calculate volumes
        try:
            pred = data.get("pred")
            if pred is not None:
                if hasattr(pred, 'numpy'):
                    pred_np = pred.numpy()
                else:
                    pred_np = np.array(pred)

                # Get voxel size from meta
                meta = data.get("image_meta_dict", {})
                spacing = meta.get(
                    "pixdim",
                    [1, 1.5, 1.5, 2.0]
                )[1:4]
                voxel_vol = float(spacing[0]) * \
                            float(spacing[1]) * \
                            float(spacing[2])

                liver_voxels = int((pred_np == 1).sum())
                tumor_voxels = int((pred_np == 2).sum())
                liver_ml = liver_voxels * voxel_vol / 1000
                tumor_ml = tumor_voxels * voxel_vol / 1000

                logger.info("=" * 40)
                logger.info("SEGMENTATION RESULTS:")
                logger.info(f"  Liver volume: {liver_ml:.1f} mL")
                logger.info(f"  Tumor volume: {tumor_ml:.1f} mL")
                if liver_ml > 0:
                    ratio = tumor_ml / liver_ml * 100
                    logger.info(f"  Tumor/Liver:  {ratio:.1f}%")
                logger.info("=" * 40)

                # Store in result for Slicer to read
                if isinstance(result, tuple):
                    result[1].update({
                        "liver_volume_ml": round(liver_ml, 1),
                        "tumor_volume_ml": round(tumor_ml, 1),
                        "tumor_liver_ratio": round(
                            tumor_ml/liver_ml*100
                            if liver_ml > 0 else 0, 1
                        ),
                    })
        except Exception as e:
            logger.warning(f"Volume calc error: {e}")

        return result