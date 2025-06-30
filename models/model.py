import torch
import torch.nn as nn
from models.mobilenetv3 import LiteMobileNetBackbone
from models.pointnet import PointNet2Backbone
from models.fusion import MaskedFusion
from models.head import BBoxPredictor


class Simple3DDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()


        self.rgb_backbone = LiteMobileNetBackbone()
        self.pc_backbone = PointNet2Backbone()
        self.fusion = MaskedFusion()
        self.bbox_predictor = BBoxPredictor()

    def forward(self, batch):
        #Extract features
        rgb_feats = self.rgb_backbone(batch['images'])
        pc_feats = self.pc_backbone(batch['pointclouds'])

        # Fuse features
        fused_feats = self.fusion(rgb_feats, pc_feats, batch['masks'])

        # Predict boxes
        prediction = self.bbox_predictor(fused_feats)

        return prediction