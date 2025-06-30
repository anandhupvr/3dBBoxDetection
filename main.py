import numpy as np
import torch
from torch.utils.data import DataLoader
from data_loader import Challenge3DDataset, collate_fn

from models.model import Simple3DDetectionModel

from utils import BoxLoss

def main():

    dataset = Challenge3DDataset("./dl_challenge")

    dataLoader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn) 


    model = Simple3DDetectionModel()
    loss = BoxLoss()

    for batch in dataLoader:
    
        pred = model(batch)
        for pr in pred:
            print("prediction shapes  : ", pr.shape)

        lsdict = loss(pred, batch)
        break

if __name__ == "__main__":
    main()