import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class OptoGNN(nn.Module):
    def __init__(self, esm_dim=320, hidden_dim=64, heads=4):
        super().__init__()
        self.gat1 = GATv2Conv(esm_dim, hidden_dim, heads=heads, dropout=0.2, concat=True)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.2, concat=False)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.regressor(x)
