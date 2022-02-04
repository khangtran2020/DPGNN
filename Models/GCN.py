from dgl.nn.pytorch.conv import GraphConv
from torch import nn
from torch.nn import functional as F
import torch

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.in_dims = in_feats
        self.out_dims = num_classes
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, g, feat):
        h = self.conv1(g, feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        if (self.out_dims == 1):
            h = F.sigmoid(h)
        return h

class GCNMultiLabel(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCNMultiLabel, self).__init__()
        self.in_dims = in_feats
        self.out_dims = num_classes
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, g, feat):
        h = self.conv1(g, feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.sigmoid(h)
        return h