import numpy as np
import torch
from torch.utils.data import DataLoader
from data_loader import Challenge3DDataset, collate_fn

from models.model import Simple3DDetectionModel

from sunrgb_loader import SUNRGBD_OfficialSplit
from datasets.sunrgbd_dataset import SunRGBDDataset

from utils import BoxLoss

def main():

    # dataset = Challenge3DDataset("./dl_challenge")

    # dataLoader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn) 


    # model = Simple3DDetectionModel()
    # loss = BoxLoss()

    # for batch in dataLoader:
    
    #     pred = model(batch)
    #     import pdb; pdb.set_trace()
    #     for pr in pred:
    #         print("prediction shapes  : ", len(pr))
    #         # lsdict = loss(pred, batch)
    #     break

    # dataset = SUNRGBD_OfficialSplit("./", split="train")

    # dataLoader = DataLoader(dataset, batch_size=2);

    # for batch in dataLoader:
    #     print(batch.shape)

    train_dataset = SunRGBDDataset('./sunrgbd_trainval', split='train')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        # collate_fn=SunRGBDDataset.collate_fn
    )

    for batch in train_loader:
        print(batch)




if __name__ == "__main__":
    main()