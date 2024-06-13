import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import dgl.function as fn

from ogb.graphproppred.mol_encoder import BondEncoder
from dgl.nn.functional import edge_softmax
from modules import MLP, MessageNorm

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        print(f"self.weight intial {self.weight}")
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        print(f"self.weights = {self.weight}, input is {input.size()}")
        
        support = torch.mm(input, self.weight)
        # print(f"support = {support}")
        output = torch.spmm(adj, support)
        # print(f"output in gc1 : {output}")
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



# def MLP(channels, batch_norm=True):
#     """
#     Create a Multi-Layer Perceptron (MLP) neural network.

#     Parameters:
#     - channels (list): List of integers representing the number of input and output channels for each layer.
#     - batch_norm (bool): Flag indicating whether to include batch normalization.

#     Returns:
#     - torch.nn.Sequential: MLP model.
#     """
#     # Define a list comprehension to create a sequence of layers for the MLP
#     layers = [
#         Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
#         for i in range(1, len(channels))
#     ]

#     # Create a Sequential model using the defined layers
#     mlp_model = Seq(*layers)

#     return mlp_model





class GENConv(nn.Module):
    r"""
    
    Description
    -----------
    Generalized Message Aggregator was introduced in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    dataset: str
        Name of ogb dataset.
    in_dim: int
        Size of input dimension.
    out_dim: int
        Size of output dimension.
    aggregator: str
        Type of aggregator scheme ('softmax', 'power'), default is 'softmax'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    learn_beta: bool
        Whether beta is a learnable variable or not. Default is False.
    p: float
        Initial power for power mean aggregation. Default is 1.0.
    learn_p: bool
        Whether p is a learnable variable or not. Default is False.
    msg_norm: bool
        Whether message normalization is used. Default is False.
    learn_msg_scale: bool
        Whether s is a learnable scaling factor or not in message normalization. Default is False.
    norm: str
        Type of ('batch', 'layer', 'instance') norm layer in MLP layers. Default is 'batch'.
    mlp_layers: int
        The number of MLP layers. Default is 1.
    eps: float
        A small positive constant in message construction function. Default is 1e-7.
    """
    def __init__(self,
                 dataset,
                 in_dim,
                 out_dim,
                 aggregator='softmax',
                 beta=1.0,
                 learn_beta=False,
                 p=1.0,
                 learn_p=False,
                 msg_norm=False,
                 learn_msg_scale=False,
                 norm='batch',
                 mlp_layers=1,
                 eps=1e-7):
        super(GENConv, self).__init__()
        
        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for i in range(mlp_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = MLP(channels, norm=norm)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=True) if learn_beta and self.aggr == 'softmax' else beta
        self.p = nn.Parameter(torch.Tensor([p]), requires_grad=True) if learn_p else p

        # if dataset == 'ogbg-molhiv':
        #     self.edge_encoder = BondEncoder(in_dim)
        # elif dataset == 'ogbg-ppa':
        #     self.edge_encoder = nn.Linear(in_dim, in_dim)
        # else:
        #     raise ValueError(f'Dataset {dataset} is not supported.')
        self.edge_encoder = nn.Linear(in_dim, in_dim)

    def forward(self, x, adj):
        # Assume edge_feats are incorporated in some way into adj or through another method not shown here

        # Basic implementation for 'softmax' aggregation (for simplicity, ignoring edge features)
        if self.aggr == 'softmax':
            # print(f"{torch.cuda.is_available()}")
            adj = adj.to_dense()
            input_att = adj * self.beta
            attention = F.softmax(input_att, dim=-1)  # Softmax over each row of the adjacency matrix
            x = torch.matmul(attention, x)  # Matrix multiplication for message passing
        elif self.aggr == 'power':
            # Power aggregation
            adj = torch.clamp(adj, 1e-7, 1e1)  # Clamping values for stability
            adj_power = adj.pow(self.p)
            x = torch.matmul(adj_power, x)
        else:
            raise NotImplementedError(f'Aggregator {self.aggr} is not supported.')

        # Apply MLP to the node features after aggregation
        x = self.mlp(x)

        return x
