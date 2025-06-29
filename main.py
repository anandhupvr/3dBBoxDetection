import numpy as np

from torch.utils.data import DataLoader
from data_loader import Challenge3DDataset, collate_fn

from models.mobilenetv3 import LiteMobileNetBackbone
from models.pointnet import PointNet2Backbone
from models.fusion import MaskedFusion
from models.head import BBoxPredictor

def main():
    print("test")

    dataset = Challenge3DDataset("./dl_challenge")

    dataLoader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn) 

    rgb_backbone = LiteMobileNetBackbone()
    pc_backbone = PointNet2Backbone()
    fuse = MaskedFusion()
    predict = BBoxPredictor()

    for batch in dataLoader:

        
        feat1 = rgb_backbone(batch['images'])
        feat2 = pc_backbone(batch['pointclouds'])
        fused = fuse(feat1, feat2, batch['masks'])

        pred = predict(fused)

        print(" rgb feature : ", feat1.shape)
        print(" pc feat : ", feat2.shape)
        print("fused : ", fused.shape)
        print("prediction : ", len(pred))
        for pr in pred:
            print("size : ", pr.shape)


        # For bboxes - depends on which collate option you chose
        if isinstance(batch['bboxes'], list):
            for i, box in enumerate(batch['bboxes']):
                print(f"  Sample {i}: {box.shape}")
        else:
            print("Padded bboxes shape:", batch['bboxes'].shape)
        
        print("Num boxes:", batch['num_boxes'])
        break



if __name__ == "__main__":
    main()