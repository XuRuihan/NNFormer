import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = 2 * in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(2 * in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # support = torch.matmul(input, self.weight)
        # output = torch.matmul(adj, support)
        # if self.bias is not None:
        #     return output + self.bias
        # else:
        #     return output

        fuse = torch.cat([torch.matmul(adj, input), input], dim=-1)
        output = torch.matmul(fuse, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


def normalize(adj):
    """Row-normalize sparse matrix"""
    rowsum = adj.sum(-1)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.0
    r_mat_inv = torch.diag_embed(r_inv)
    # print(r_inv.shape, r_mat_inv.shape, adj.shape)
    adj = r_mat_inv @ adj
    return adj
