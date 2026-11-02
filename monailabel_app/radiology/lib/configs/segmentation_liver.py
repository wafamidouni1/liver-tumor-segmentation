# Liver + Tumor Segmentation Config
# SwinUNETR V4 — Liver:0.916 Tumor:0.561

import logging
import os
import torch
from typing import Any, Dict, Optional, Union

import lib.infers
from monai.networks.nets import SwinUNETR

from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask

logger = logging.getLogger(__name__)


class SegmentationLiver(TaskConfig):
    def __init__(self):
        super().__init__()

    def init(self, name: str, model_dir: str,
             conf: Dict[str, str], planner: Any, **kwargs):
        super().init(name, model_dir, conf, planner, **kwargs)

        # ── Labels ────────────────────────────────────────
        self.labels = {
            "liver": 1,
            "tumor": 2,
        }

        # ── Model path ────────────────────────────────────
        self.path = [
            os.path.join(model_dir,
                         "swin_best_model_v4.pt"),
        ]

        # ── Spacing + ROI ─────────────────────────────────
        self.target_spacing = (1.5, 1.5, 2.0)
        self.roi_size       = (96, 96, 64)

        # ── Build SwinUNETR ───────────────────────────────
        try:
            self.network = SwinUNETR(
                spatial_dims=3,
                in_channels=1,
                out_channels=3,
                feature_size=48,
                use_checkpoint=False,
            )
        except TypeError:
            self.network = SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=1,
                out_channels=3,
                feature_size=48,
                use_checkpoint=False,
            )

        # ── Load V4 weights ───────────────────────────────
        model_path = self.path[0]
        if os.path.exists(model_path):
            state = torch.load(
                model_path, map_location="cpu")
            self.network.load_state_dict(state)
            self.network.eval()
            logger.info(f"✓ V4 model loaded: {model_path}")
        else:
            logger.error(f"✗ Model not found: {model_path}")

    def infer(self) -> Union[InferTask,
                             Dict[str, InferTask]]:
        task: InferTask = lib.infers.SegmentationLiver(
            path=self.path,
            network=self.network,
            roi_size=self.roi_size,
            target_spacing=self.target_spacing,
            labels=self.labels,
            preload=False,
        )
        return task

    def trainer(self) -> Optional[TrainTask]:
        return None

    def strategy(self) -> Union[None, Strategy,
                                Dict[str, Strategy]]:
        return {}

    def scoring_method(self) -> Union[None, ScoringMethod,
                                      Dict[str, ScoringMethod]]:
        return {}