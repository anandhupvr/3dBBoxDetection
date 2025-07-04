import torch
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, radius, global_max_pool

class SAModule(nn.Module):
    def __init__(self, ratio, radius, mlp_channels):
        super().__init__()
        self.ratio = ratio
        self.radius = radius
        self.conv = PointNetConv(MLP(mlp_channels), add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.radius, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSAModule(nn.Module):
    def __init__(self, mlp_channels):
        super().__init__()
        self.nn = MLP(mlp_channels)

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        return x

class PointNet2Backbone(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()
        self.sa1 = SAModule(0.25, 0.2, [input_dim, 64, 64, 128])         # ~1024 points
        self.sa2 = SAModule(0.25, 0.4, [128 + 3, 128, 128, 256])         # ~256 points
        self.sa3 = SAModule(0.25, 0.8, [256 + 3, 256, 256, 512])         # ~64 points
        self.sa4 = GlobalSAModule([512 + 3, 512, 1024])                  # global features

    def forward(self, data):
        # Assumes data.x is None or can be added later (e.g., fused RGB or mask features)
        x, pos, batch = data.x, data.pos, data.batch
        x1, pos1, batch1 = self.sa1(x, pos, batch)
        x2, pos2, batch2 = self.sa2(x1, pos1, batch1)
        x3, pos3, batch3 = self.sa3(x2, pos2, batch2)
        x4 = self.sa4(x3, pos3, batch3)  # B x 1024
        return x4  # Global feature vector per object

