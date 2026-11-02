import os
import torch
import numpy as np
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Spacingd, ScaleIntensityRanged,
    CropForegroundd, ResizeWithPadOrCropd,
    ToTensord, Activationsd, AsDiscreted
)
from monai.inferers import sliding_window_inference
from monailabel.interfaces.tasks.infer_v2 import InferTask, InferType
from monailabel.transform.post import FindContoursd

class MyLiverSegmentation(InferTask):
    def __init__(self, path, network=None, type=InferType.SEGMENTATION,
                 labels=None, dimension=3, description="Liver & Tumor Segmentation"):
        
        if labels is None:
            labels = {
                "liver": 1,
                "tumor": 2,
            }
        
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
        )

    def pre_transforms(self, data=None):
        return Compose([
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Spacingd(
                keys="image",
                pixdim=(1.5, 1.5, 2.0),
                mode="bilinear"
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=-200, a_max=250,
                b_min=0.0, b_max=1.0,
                clip=True
            ),
            CropForegroundd(
                keys="image",
                source_key="image"
            ),
            ResizeWithPadOrCropd(
                keys="image",
                spatial_size=(-1, -1, 128)
            ),
            ToTensord(keys="image"),
        ])

    def inferer(self, data=None):
        return sliding_window_inference(
            inputs=data["image"].unsqueeze(0),
            roi_size=(96, 96, 64),
            sw_batch_size=2,
            predictor=self.network,
            overlap=0.25,
        )

    def post_transforms(self, data=None):
        return Compose([
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True),
        ])

    def network(self):
        try:
            model = SwinUNETR(
                spatial_dims=3,
                in_channels=1,
                out_channels=3,
                feature_size=48,
                use_checkpoint=False,
            )
        except TypeError:
            model = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=3,
                feature_size=48,
                use_checkpoint=False,
            )

        # Load your V4 weights
        weights_path = r"./models\swin_best_model_v4.pt"
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu")
        )
        model.eval()
        return model