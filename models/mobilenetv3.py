import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class LiteMobileNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet = mobilenet_v3_small(weights="DEFAULT").features
        self.out_channels = 576
    
    def forward(self, x):
        return self.mobilenet(x)

def test():
    model = LiteMobileNetBackbone()
    input = torch.randn(1, 3, 480, 640);
    features = model(input)
    print("features : ", features.shape)

if __name__ == "__main__":
    test()