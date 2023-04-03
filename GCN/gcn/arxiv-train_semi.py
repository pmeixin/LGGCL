import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
import random
from torch.nn.parameter import Parameter

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, JumpingKnowledge
from torch_geometric.data import NeighborSampler
import torch_geometric.transforms as T
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from utils import load_data, accuracy, sparse_mx_to_torch_sparse_tensor, normalize, label_propagation, drop_feature, aug_random_mask, adj_nor

from torch_geometric.nn.inits import glorot, zeros

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from arxiv_logger import Logger



"""
批处理：full-batch
图数据表示方法：SpMM
模型：GCN_res
数据集：ogbn-arxiv
"""

# 加载数据集
dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/')
# dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
# dataset = PygNodePropPredDataset(name='ogbn-products', root='./products/', transform=T.ToSparseTensor())
print(dataset)
data = dataset[0]
print(data)

#import ipdb;ipdb.set_trace()
# 划分数据集
split_idx = dataset.get_idx_split()
idx_train = split_idx['train']
idx_valid = split_idx['valid']
idx_test = split_idx['test']
features = data.x
features = normalize(features)
features = torch.from_numpy(features)
labels = data.y
adj = torch.eye(data.x.shape[0])
for i in range(data.edge_index.shape[1]):
    adj[data.edge_index[0][i]][edge_index[1][i]] = 1
#adj_t_dense = data.adj_t.to_dense()
#for i in range(adj_t_dense.shape[1]):
#    adj[adj_t_dense[0][i]][adj_t_dense[1][i]] = 1
adj = adj.float()
adj = adj_nor(adj)




# 定义评估器
evaluator = Evaluator(name='ogbn-arxiv')
# evaluator = Evaluator(name='ogbn-products')



class ONINorm(torch.nn.Module):
    def __init__(self, T=5, norm_groups=1, *args, **kwargs):
        super(ONINorm, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-5

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps*eye
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)

        S = S.div(norm_S)
        B = [torch.Tensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = B[self.T].matmul(Zc).div_(norm_S.sqrt())
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)

        # W = W.view_as(weight)
        # import ipdb;ipdb.set_trace()
        return W.view_as(weight)
# 定义网络
# GCN
@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
class GCNConv1(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Its node-wise formulation is given by:

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \sum_{j \in \mathcal{N}(v) \cup
        \{ i \}} \frac{e_{j,i}}{\sqrt{\hat{d}_j \hat{d}_i}} \mathbf{x}_j

    with :math:`\hat{d}_i = 1 + \sum_{j \in \mathcal{N}(i)} e_{j,i}`, where
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1.0`)

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = False, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv1, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_normalization1 = ONINorm(T=4, norm_groups=1)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        self.W = self.weight_normalization1(self.weight)
        x = x @ self.W

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class GraphConvolution1(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_normalization1 = ONINorm(T=4, norm_groups=1)
        # nn.init.xavier_uniform_(self.weight.data, gain=1.414)  # 初始化
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

    def forward(self, x, adj):
        # x = F.dropout(x, 0.5, training=self.training)
        self.W = self.weight_normalization1(self.weight)
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class GCNNet(nn.Module):
    def __init__(self, dataset, hidden=256, num_layers=3):
        """
        :param dataset: 数据集
        :param hidden: 隐藏层维度，默认256
        :param num_layers: 模型层数，默认为3
        """
        super(GCNNet, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(dataset.num_node_features, hidden))
        self.bns.append(nn.BatchNorm1d(hidden))

        for i in range(self.num_layers - 2):
            self.convs.append(GCNConv1(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.convs.append(GCNConv(hidden, dataset.num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, adj_t = data.x, data.adj_t

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)  # 小数据集不norm反而效果更好
            if i==2:
                out = x
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.convs[-1](x, adj_t)
        x = F.log_softmax(x, dim=1)
        #import ipdb;ipdb.set_trace()

        return x


# GCN_res
class GCN_res(nn.Module):
    def __init__(self, dataset, hidden=256, num_layers=6):
        super(GCN_res, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.input_fc = nn.Linear(dataset.num_node_features, hidden)

        for i in range(self.num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.out_fc = nn.Linear(hidden, dataset.num_classes)
        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.input_fc.reset_parameters()
        self.out_fc.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, data):
        x, adj_t = data.x, data.adj_t

        x = self.input_fc(x)
        x_input = x  # .copy()

        layer_out = []  # 保存每一层的结果
        for i in range(self.num_layers):
            x = self.convs[i](x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x, inplace=True)
            x = F.dropout(x, p=0.5, training=self.training)

            if i == 0:
                x = x + 0.2 * x_input
            else:
                x = x + 0.2 * x_input + 0.5 * layer_out[i - 1]
            layer_out.append(x)

        weight = F.softmax(self.weights, dim=0)
        for i in range(len(layer_out)):
            layer_out[i] = layer_out[i] * weight[i]

        x = sum(layer_out)
        x = self.out_fc(x)
        x = F.log_softmax(x, dim=1)

        return x


# 实例化模型
model = GCNNet(dataset=dataset, hidden=256, num_layers=4)#3
# model = GCN_res(dataset=dataset, hidden=128, num_layers=8)
print(model)

# 转换为cpu或cuda格式
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
data = data.to(device)
data.adj_t = data.adj_t.to_symmetric()  # 对称归一化
#train_idx = train_idx.to(device)

# 定义损失函数和优化器
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

# 定义训练函数
def train():
    model.train()

    out = model(data)
    loss = criterion(out[train_idx], data.y.squeeze(1)[train_idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# 定义测试函数
@torch.no_grad()
def test():
    model.eval()

    out = model(data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


# 程序入口
if __name__ == '__main__':
    runs = 5
    logger = Logger(runs)
    
    #####获取最佳1
    acc_final = torch.zeros(runs)
    ####获取最佳

    for run in range(runs):
        ### 数据划分
        index_train = []
        index_val = []
        
        for i_label in range(data.y.max()+1):
            index_sub = [i for i,x in enumerate(data.y) if x==i_label]#train/val index
            siz = len(index_sub)
            if siz >= 60:
                index_sub = random.sample(index_sub, 60)
                index_train += index_sub[:30]
                index_val += index_sub[30:]
        
        #import ipdb;ipdb.set_trace()
        index_train.sort()
        index_val.sort()
        index_train_val = index_val + index_train
        
        index_test = [i for i in range(data.y.shape[0]) if i not in index_train_val]
        #import ipdb;ipdb.set_trace()
        
        train_mask = sample_mask(index_train, data.y.shape)#array([ True,  True,  True, ..., False, False, False])
        val_mask = sample_mask(index_val, data.y.shape)
        test_mask = sample_mask(index_test, data.y.shape)
        idx_train = torch.Tensor(train_mask).bool()
        idx_val = torch.Tensor(val_mask).bool()
        idx_test = torch.Tensor(test_mask).bool()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()  
        
        ####获取最佳2
        max_val = 0
        max_test = 0
        max_idx = 0
        ####获取最佳
        print(sum(p.numel() for p in model.parameters()))
        model.reset_parameters()

        for epoch in range(500):
            loss = train()
            # print('Epoch {:03d} train_loss: {:.4f}'.format(epoch, loss))

            result = test()
            train_acc, valid_acc, test_acc = result
            ####获取最佳3
            if valid_acc >= max_val:
                max_val = valid_acc
                max_test = test_acc
                max_idx = epoch
            ####获取最佳
            
            # print(f'Train: {train_acc:.4f}, Val: {valid_acc:.4f}, 'f'Test: {test_acc:.4f}')
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')

            logger.add_result(run, result)
        
        #####获取最佳4    
        acc_final[run] = max_test
        print(max_test)
        print(max_idx)
        ####获取最佳4

    logger.print_statistics()
    ####获取最佳5
    print("=== Final ===")
    print(torch.max(acc_final))
    print(torch.min(acc_final))
    #print("30次平均",torch.mean(test_acc))
    #print("30次标准差",test_acc.std())
    #print("20次平均",torch.mean(test_acc[:20]))
    #print("20次标准差",test_acc[:20].std())
    print("10次平均",torch.mean(acc_final))
    print("10次标准差",acc_final.std())
    #import ipdb;ipdb.set_trace()
    print(acc_final)
    ####获取最佳