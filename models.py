import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch_geometric.nn import global_max_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU as LRU
from torch.nn import Sequential as Seq, Dropout, Linear as Lin


from ogb.graphproppred.mol_encoder import AtomEncoder
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from modules import norm_layer
from layers import GENConv
    
def MLP(channels, batch_norm=True):
    """
    Create a Multi-Layer Perceptron (MLP) neural network.

    Parameters:
    - channels (list): List of integers representing the number of input and output channels for each layer.
    - batch_norm (bool): Flag indicating whether to include batch normalization.

    Returns:
    - torch.nn.Sequential: MLP model.
    """
    # Define a list comprehension to create a sequence of layers for the MLP
    layers = [
        Seq(Lin(channels[i - 1], channels[i]), BN(channels[i]), ReLU())
        for i in range(1, len(channels))
    ]

    # Create a Sequential model using the defined layers
    mlp_model = Seq(*layers)

    return mlp_model




# Function for total variation (TV) normalization
def tv_norm(X, eps=1e-3):
    """
    Perform total variation (TV) normalization on input tensor X.

    Parameters:
    - X (torch.Tensor): Input tensor.
    - eps (float): Small constant to avoid division by zero.

    Returns:
    - torch.Tensor: Result of TV normalization.
    """
    X = X - torch.mean(X, dim=1, keepdim=True)
    X = X / torch.sqrt(torch.sum(X ** 2, dim=1, keepdim=True) + eps)
    return X


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nNodes):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        
        # Define the learnable weight for nodes
        # self.learnable_weight = nn.Parameter(torch.ones(nNodes, nhid) * 1e-2)

        self.dropout = dropout
        stdv = 1e-2
        stdvp = 1e-2
        # self.KNclose = nn.Parameter(torch.randn(nhid, nNodes) * stdv) 
        # self.convs1x1 = nn.Parameter(torch.randn(nhid, nhid, nhid) * stdv)
        self.mlp = Seq(
                MLP([2, 128]), Dropout(0.5), MLP([128, 64]), Dropout(0.5),
                Lin(64, nclass))
     
       

    def forward(self, x, adj):
       
        print(x.size())
        x = F.relu(self.gc1(x, adj))
        print(x.size())
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.gc4(x, adj)
        

        return F.log_softmax(x, dim=1)



class lwGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nNodes):
        super(lwGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nhid)
        self.gc4 = GraphConvolution(nhid, nclass)
        self.nNodes = nNodes
        
        # Define the learnable weight for nodes
        self.learnable_weight1 = nn.Parameter(torch.zeros(nNodes, nhid))
        self.learnable_weight2 = nn.Parameter(torch.zeros(nNodes, nclass))

        self.dropout = dropout
        stdv = 1e-2
        stdvp = 1e-2
        # self.KNclose = nn.Parameter(torch.randn(nhid, nNodes) * stdv) 
        # self.convs1x1 = nn.Parameter(torch.randn(nhid, nhid, nhid) * stdv)
        self.mlp = Seq(
                MLP([2, 128]), Dropout(0.5), MLP([128, 64]), Dropout(0.5),
                Lin(64, nclass))
        
    def edgeConv(self, xe, K, groups=1):
        """
        Perform edge convolution on the input tensor xe using the specified kernel K.

        Parameters:
        - xe (torch.Tensor): Input tensor.
        - K (torch.Tensor): Convolution kernel.
        - groups (int): Number of groups for grouped convolution.

        Returns:
        - torch.Tensor: Result of the edge convolution.
        """
        print(f"xe.dim = {xe.dim()}")
        print(f"K.dim = {K.dim()}")
        if xe.dim() == 4:
            if K.dim() == 2:
                xe = F.conv2d(xe, K.unsqueeze(-1).unsqueeze(-1), groups=groups)
            else:
                xe = F.conv2d(xe, K, padding=int((K.shape[-1] - 1) / 2))
                
        elif xe.dim() == 3:
            if K.dim() == 2:
                xe = F.conv1d(xe, K.unsqueeze(-1), groups=groups)
            else:
                xe = F.conv1d(xe, K, padding=int((K.shape[-1] - 1) / 2))
        return xe
    
    def singleLayer(self, x, K, relu=True, norm=False, groups=1, openclose=False):
        """
        Perform a single-layer operation on the input tensor x using the specified kernel K.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - K (torch.Tensor): Convolution kernel.
        - relu (bool): Apply ReLU activation if True, apply tanh if False.
        - norm (bool): Apply instance normalization if True.
        - groups (int): Number of groups for grouped convolution.
        - openclose (bool): Use open-close operation if True, close-open operation if False.

        Returns:
        - torch.Tensor: Result of the single-layer operation.
        """
        if openclose:  # if K.shape[0] != K.shape[1]:
            x = self.edgeConv(x, K, groups=groups)
            if norm:
                x = F.instance_norm(x)
            if relu:
                x = F.relu(x)
            else:
                x = F.tanh(x)
        if not openclose:  # if K.shape[0] == K.shape[1]:
            x = self.edgeConv(x, K, groups=groups)
            if not relu:
                x = F.tanh(x)
            else:
                x = F.relu(x)
            if norm:
                beta = torch.norm(x)
                x = beta * tv_norm(x)
            x = self.edgeConv(x, K.t(), groups=groups)
        return x

    def finalDoubleLayer(self, x, K1, K2):
        """
        Perform a final double-layer operation on the input tensor x using two specified kernels.

        Parameters:
        - x (torch.Tensor): Input tensor.
        - K1 (torch.Tensor): First convolution kernel.
        - K2 (torch.Tensor): Second convolution kernel.

        Returns:
        - torch.Tensor: Result of the final double-layer operation.
        """
        x = F.tanh(x)
        x = self.edgeConv(x, K1)
        x = F.tanh(x)
        x = self.edgeConv(x, K2)
        x = F.tanh(x)
        x = self.edgeConv(x, K2.t())
        x = F.tanh(x)
        x = self.edgeConv(x, K1.t())
        x = F.tanh(x)
        return x

    def forward(self, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.conv1d(x, self.KNclose.unsqueeze(-1))
        # print(f"x size = {x.size()}")
        # print(f"self.KNclose size = {self.KNclose.size()}")
        # x = self.singleLayer(x, self.KNclose, relu=True, openclose=False, norm=False)
        # print(f"x.size = {x.size()}")
        
        # x = F.relu(self.gc1(x, adj))
        
        
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # x = F.conv1d(x, self.KNclose.unsqueeze(-1))
        # print(f"x.size = {x.size()}")
        
        # x = x.squeeze().t()
        # print(f"x.size = {x.size()}")
        # x = global_max_pool(x, batch=None)
        # print(f"x.size = {x.size()}")
        # x = self.mlp(x)
        # print(f"x.size = {x.size()}")
        
        
        
        
        
        
        # print(f"start forward: {x}")
        
        print(x.size())
        x = F.relu(self.gc1(x, adj))
        print(x.size())
        x = F.dropout(x, self.dropout, training=self.training)
        # Apply the learnable weight for nodes
        print(f"size of x :{x.squeeze(1).size()}, size of weight {self.learnable_weight1.size()}")
        if x.squeeze(1).size(0) == self.nNodes:
            x = torch.mul(x.squeeze(1), self.learnable_weight1)
        
        # if x.squeeze(1).size(0) != 2131:
        #     # Slice the matrix to keep only the first 2000 rows
        #     print(f"size not match: {x.squeeze(1).size(0)}")
        #     lw = self.learnable_weight
        #     lw.cuda()
        #     if x.squeeze(1).size(0) < 2131:
        #         lw = self.learnable_weight[:x.squeeze(1).size(0), :]
        #     else:
        #         ones = torch.ones((x.squeeze(1).size(0) - 2131, lw.size(1)), dtype=lw.dtype, device=lw.device)
        #         lw = torch.cat((lw, ones), dim=0)

        #     print(f"lw size = {lw.size()}")
        #     x = torch.mul(x.squeeze(1), lw)
        # else:
        #     x = torch.mul(x.squeeze(1), self.learnable_weight)
        
        print(f"size of x :{x.size()}")
        x = x.squeeze(1)
        print(f"size of x :{x.size()}, size of weight {self.learnable_weight1.size()}")

        # print(x)
        # x = self.gc2(x, adj)
        # x = self.gc3(x, adj)
        x = self.gc4(x, adj)
        
        if x.squeeze(1).size(0) == self.nNodes:
            x = torch.mul(x.squeeze(1), self.learnable_weight2)
        

        # print(x)
        # x = torch.mean(x, dim=0)
        # x = self.mlp(x)
        # print(f"x.size = {x.size()}")
        return F.log_softmax(x, dim=1)






class DeeperGCN(nn.Module):
    r"""

    Description
    -----------
    Introduced in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    dataset: str
        Name of ogb dataset.
    node_feat_dim: int
        Size of node feature dimension.
    edge_feat_dim: int
        Size of edge feature dimension.
    hid_dim: int
        Size of hidden dimension.
    out_dim: int
        Size of output dimension.
    num_layers: int
        Number of graph convolutional layers.
    dropout: float
        Dropout rate. Default is 0.
    norm: str
        Type of ('batch', 'layer', 'instance') norm layer in MLP layers. Default is 'batch'.
    pooling: str
        Type of ('sum', 'mean', 'max') pooling layer. Default is 'mean'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    lean_beta: bool
        Whether beta is a learnable weight. Default is False.
    aggr: str
        Type of aggregator scheme ('softmax', 'power'). Default is 'softmax'.
    mlp_layers: int
        Number of MLP layers in message normalization. Default is 1.
    """
    def __init__(self,
                 dataset,
                 node_feat_dim,
                 edge_feat_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout=0.,
                 norm='batch',
                 pooling='mean',
                 beta=1.0,
                 learn_beta=False,
                 aggr='softmax',
                 mlp_layers=1):
        super(DeeperGCN, self).__init__()
        
        self.dataset = dataset
        self.num_layers = num_layers
        self.dropout = dropout
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(self.num_layers):
            conv = GENConv(dataset=dataset,
                           in_dim=hid_dim,
                           out_dim=hid_dim,
                           aggregator=aggr,
                           beta=beta,
                           learn_beta=learn_beta,
                           mlp_layers=mlp_layers,
                           norm=norm)
            
            # conv = GraphConvolution(hid_dim, hid_dim)
            
            self.gcns.append(conv)
            self.norms.append(norm_layer(norm, hid_dim))

        # if self.dataset == 'ogbg-molhiv':
        #     self.node_encoder = AtomEncoder(hid_dim)
        # elif self.dataset == 'ogbg-ppa':
        #     self.node_encoder = nn.Linear(node_feat_dim, hid_dim)
        #     self.edge_encoder = nn.Linear(edge_feat_dim, hid_dim)
        # else:
        #     raise ValueError(f'Dataset {dataset} is not supported.')
        
        # transforms sparse or limited atom features into a more expressive representation
        self.node_encoder = nn.Linear(node_feat_dim, hid_dim) # not sure
        
        if pooling == 'sum':
            # self.pooling = SumPooling()
            self.pooling = nn.SumPool1d(kernel_size=2, stride=2, padding=0)
        elif pooling == 'mean':
            # self.pooling = AvgPooling()
            self.pooling = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        elif pooling == 'max':
            # self.pooling = MaxPooling()
            self.pooling = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        else:
            raise NotImplementedError(f'{pooling} is not supported.')
        
        self.output = nn.Linear(int(hid_dim/2), out_dim)

    def forward(self, x, adj):
        edge_feats = None  # Define how you obtain edge features based on `adj`, if needed

        # Encode node features
        
        # print(f"before encoder {x}") # (N x nFeat)
        # indices = torch.arange(x.size(0)).long()  # Convert indices to LongTensor
        hv = self.node_encoder(x)  # Apply AtomEncoder with LongTensor indices
        # hv = x
        
        # print(f"after encoder: {hv}") #(N x nhid)
        for layer in range(self.num_layers):
            # Apply normalization and activation
            hv1 = self.norms[layer](hv)
            hv1 = F.relu(hv1)
            hv1 = F.dropout(hv1, p=self.dropout, training=self.training)
            
            # Perform the graph convolution using the adjacency matrix
            # This requires modifying or replacing GENConv to work with `adj` and `x`
            # For example, a simple graph convolution operation could be:
            # hv = adj @ hv1  # This is a very simplified version; real convolutions are more complex
            
            # Assuming GENConv or its equivalent is adapted to use adj and node features:
            hv = self.gcns[layer](hv1, adj) + hv

        # Pooling: since you don't have a graph object, you'll manually apply pooling over node embeddings
        # This depends on how you've implemented or adapted pooling functions to work without a graph object
        # print(f"dimension: {hv.size()}")
        # hv = hv.unsqueeze()
        print(f"before pooling: {hv.size()}")
        print(f"feature before pooling : {hv}")
        
        h_g = self.pooling(hv)  # Ensure this operation is adapted for direct tensor inputs
        print(f"after pooling: {h_g.size()}")
        print(f"feature after pool = {h_g}")
        
        retval = self.output(h_g)
        print(f"return size: {retval.size()}")
        # print(f"output is {retval}")
        return retval
