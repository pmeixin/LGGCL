from __future__ import division
from __future__ import print_function
import optuna
import random

import time
import argparse
import numpy as np
from copy import deepcopy as dcp
import scipy.sparse as sp
import math

import os.path as osp
import os
from torch_geometric.utils import dropout_adj, to_dense_adj, to_scipy_sparse_matrix, add_self_loops, dense_to_sparse
import clustering
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, sparse_mx_to_torch_sparse_tensor, normalize, label_propagation, drop_feature, aug_random_mask, adj_nor
from models import GCN,MLP

from torch_geometric.datasets import Planetoid, CitationFull,WikiCS, Coauthor, Amazon
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--data_aug', type=int, default=1,
                    help='do data augmentation.')
parser.add_argument('--neg_type', type=int, default=0,
                    help='0无neg,1neg选择')
parser.add_argument('--pos_type', type=float, default=0,
                    help='0,selection;1 not selection')
parser.add_argument('--kk', type=int, default=1,
                    help='y_pre select k')
parser.add_argument('--cl_num', type=int, default=2,
                    help='0 train set;1 all dataset.')
parser.add_argument('--sample_size', type=float, default=1.0,
                    help='sample size')
parser.add_argument('--encoder_type', type=int, default=3,
                    help='do data augmentation.')
parser.add_argument('--debias', type=float, default=0.1,
                    help='debias rate.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight', type=float, default=0.,
                    help='Initial loss rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--tau', type=float, default=0.4,
                    help='tau rate .')
parser.add_argument('--dataset', type=str, default='Cora',
                    help='Cora/CiteSeer/PubMed/')
parser.add_argument('--encoder', type=str, default='GCN',
                    help='GCN/SGC/GAT/')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
                    default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--nmb_cluster','--k', type=int, default=7,
                    help='number of cluster for k-means (default:10000)')
parser.add_argument('--verbose', action='store_true', help='chatty')

args = parser.parse_args()
times = 10

#load dataset
#def get_dataset(path, name):
#    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP','WikiCS','Amazon-Photo']
#    name = 'dblp' if name == 'DBLP' else name
#    print(name)
#    return (CitationFull if name == 'dblp' else Planetoid)(path,name,transform=T.NormalizeFeatures())
    
def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'Karate', 'WikiCS', 'Coauthor-CS', 'Coauthor-Phy',
                    'Amazon-Computers', 'Amazon-Photo', 'ogbn-arxiv', 'ogbg-code']
    name = 'dblp' if name == 'DBLP' else name
    root_path = osp.expanduser('~/datasets')

    if name == 'Coauthor-CS':
        return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())

    if name == 'Coauthor-Phy':
        return Coauthor(root=path, name='physics', transform=T.NormalizeFeatures())

    if name == 'WikiCS':
        return WikiCS(root=path, transform=T.NormalizeFeatures())

    if name == 'Amazon-Computers':
        return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())

    if name == 'Amazon-Photo':
        return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())

    if name.startswith('ogbn'):
        return PygNodePropPredDataset(root=osp.join(root_path, 'OGB'), name=name, transform=T.NormalizeFeatures())

    return (CitationFull if name == 'dblp' else Planetoid)(osp.join(root_path, 'Citation'), name, transform=T.NormalizeFeatures())

if args.dataset=='Cora' or args.dataset=='CiteSeer' or args.dataset=='PubMed':
    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    print(path)
    print("hhhh")
else:
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    print(path)
    print("hhhh")
dataset = get_dataset(path, args.dataset)    
data = dataset[0]
#data.edge_index, _  = pyg.utils.add_self_loops(data.edge_index)

#import ipdb;ipdb.set_trace()


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
    

#数据处理
features = data.x
features = normalize(features)
features = torch.from_numpy(features)
labels = data.y
adj = torch.eye(data.x.shape[0])
for i in range(data.edge_index.shape[1]):
    adj[data.edge_index[0][i]][data.edge_index[1][i]] = 1
adj = adj.float()
adj = adj_nor(adj)


best_model = None
best_val_acc = 0.0
def train(model, optimizer, epoch, features, adj, idx_train, idx_val, labels, data_aug, encoder_type, debias, kk, sample_size, neg_type, pos_type):
    global best_model
    global best_val_acc
    t = time.time()
    model.train()
    optimizer.zero_grad()

    #supervised CE loss
    y_pre, _ = model(features, adj, encoder_type)
    loss_train = F.nll_loss(y_pre[idx_train], labels[idx_train])
    acc_train = accuracy(y_pre[idx_train], labels[idx_train])
    
    #sample
    node_mask = torch.empty(features.shape[0],dtype=torch.float32).uniform_(0,1).cuda()
    node_mask = node_mask < sample_size

    #sampled_cl_pos_mask
    train_label = labels[node_mask].contiguous().view(-1,1)
    train_cl_pos_mask = torch.eq(train_label, train_label.T).float().cuda()

    #train_cl_neg_mask
    str1 = 1
    str0 = 0
    str0 = np.array(str0)
    str1 = np.array(str1)
    str_tensor = torch.tensor(str1).cuda()
    str0_tensor = torch.tensor(str0).cuda()
    train_cl_neg_mask = torch.where(train_cl_pos_mask>0, str0_tensor, str_tensor)

    #clustering

    node_mask_2 = torch.empty(features.shape[0], dtype=torch.float32).uniform_(0, 1).cuda()
    node_mask_2 = node_mask_2 <= 1
    deepcluster = clustering.__dict__[args.clustering](y_pre.shape[1])
    clustering_index = deepcluster.cluster(features, verbose=args.verbose)
    clustering_index = torch.LongTensor(clustering_index)
    clustering_index_np = torch.zeros(data.x.shape[0], y_pre.shape[1])
    clustering_index_np = clustering_index_np.scatter_(1, clustering_index, 1)
    clustering_index_np = clustering_index_np[node_mask]
    clustering_neg_mask = torch.mm(clustering_index_np, clustering_index_np.T) <=0
    clustering_true_mask = torch.mm(clustering_index_np, clustering_index_np.T) == 1
    clustering_true_mask = clustering_true_mask.cuda()
    clustering_neg_mask = clustering_neg_mask.cuda()

    #positive selection, pos_mask
    if pos_type == 0:
        y_pre = y_pre.detach()
        y_pre_p = y_pre[node_mask].cuda()

        _, y_topind = torch.topk(y_pre_p, kk)

        y_ol = torch.zeros(y_pre_p.shape).cuda()
        y_ol = y_ol.scatter_(1, y_topind, 1)

        out_pos_mask = torch.mm(y_ol, y_ol.T) > 0
        out_pos_mask = out_pos_mask.cuda()
        #out_pos_mask 1 represents true pos node pairs that nodes belong to the same class

        del y_ol, y_topind
        torch.cuda.empty_cache()

    else :
        out_pos_mask = torch.eye(node_mask.sum()).cuda()

    pos_mask = train_cl_pos_mask
    #filter pos mask
    #pos_mask = torch.mul(out_pos_mask, clustering_true_mask).cuda()
    #pos_mask = out_pos_mask
    #pos_mask = torch.eye(node_mask.sum()).cuda()

    #negative selection, out_neg_mask
    if neg_type == 0:
        y_pre = y_pre.detach()
        y_pre_n = y_pre[node_mask].cuda()

        _, y_poslabel = torch.topk(y_pre_n, kk)
        y_pl = torch.zeros(y_pre_n.shape).cuda()
        #import ipdb;ipdb.set_trace()
        y_pl = y_pl.scatter_(1, y_poslabel, 1)
        out_neg_mask = torch.mm(y_pl, y_pl.T) <= 0
        out_neg_mask = out_neg_mask.cuda()
        #out_neg_mask 1 represents true neg node pairs that nodes belong to different class


        del y_pl, y_poslabel
        torch.cuda.empty_cache()
    else :
        out_neg_mask = (1 - torch.eye(node_mask.sum())).cuda()
        #out_neg_mask 1 represents false neg node pairs that all different nodes

    #neg_mask = all_neg

    y_soft_pre = F.softmax(y_pre[node_mask], dim=1).cuda()
    y_pre_mean = torch.mean(y_soft_pre, dim=1, keepdim=True).cuda()
    y_pre_std = torch.std(y_soft_pre, dim=1, keepdim=True).cuda()

    neg_weight = torch.zeros(y_soft_pre.shape).cuda()

    #for i in range(neg_weight.shape[0]):
    #for j in range(neg_weight.shape[1]):
    #neg_weight[i][j] = torch.add(y_soft_pre[i][j], torch.neg(y_pre_mean[i])) * torch.add(y_soft_pre[i][j], torch.neg(y_pre_mean[i]))
    #neg_weight[i][j] = torch.neg(neg_weight[i][j] / 2*(y_pre_std[i]))
    #neg_weight[i][j] = torch.exp(neg_weight[i][j])

    for i in range(neg_weight.shape[0]):
        neg_weight[i] = torch.add(y_soft_pre[i], torch.neg(y_pre_mean[i])) * torch.add(y_soft_pre[i], torch.neg(y_pre_mean[i]))
        neg_weight[i] = torch.neg(neg_weight[i] / 2*(y_pre_std[i]))
        neg_weight[i] = torch.exp(neg_weight[i])

    #filter neg mask
    #neg_mask = torch.mul(out_neg_mask, clustering_neg_mask).cuda()
    #neg_mask = torch.logical_xor(out_neg_mask, clustering_neg_mask).cuda()
    #neg_mask = out_neg_mask
    neg_mask = train_cl_neg_mask

    neg_mask = neg_mask.to(torch.float64)
    #nl_neg_mask = (1 - torch.eye(node_mask.sum())).cuda()

    for i in range(neg_mask.shape[0]):
        neg_label_ind = torch.max(y_pre[node_mask], dim=1, keepdim=True)[1].T.cuda()
    out = torch.take(neg_weight[i], neg_label_ind).cuda()
    neg_mask[i] = neg_mask[i] * out
    
    if data_aug == 1:
        features1 = drop_feature(features, 0.3)
        features2 = drop_feature(features, 0.4)
        
        _, output1 = model(features1, adj, encoder_type)
        _, output2 = model(features2, adj, encoder_type)
        
        del features1, features2
        torch.cuda.empty_cache()

        loss_cl = model.cl_lossaug(output1, output2, node_mask, neg_mask, pos_mask, debias)
    else:
        pass

    if neg_type == 0 or pos_type == 0:
        if epoch<=60 :
            loss = loss_train + 0.0001 * loss_cl
        else:
            loss = loss_train + 0.8 * loss_cl
    else:
        loss = loss_train + args.weight * loss_cl
        
    loss.backward()
    optimizer.step()
            
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        y_pre, _ = model(features, adj, encoder_type)

    loss_val = F.nll_loss(y_pre[idx_val], labels[idx_val])
    acc_val = accuracy(y_pre[idx_val], labels[idx_val])
    if acc_val >= best_val_acc:
        best_val_acc = acc_val
        best_model = dcp(model)
            
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_cl: {:.4f}'.format(loss_cl.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(model, features, adj, labels, idx_test, encoder_type):
    model.eval()
    y_pre, _ = model(features, adj, encoder_type)
    loss_test = F.nll_loss(y_pre[idx_test], labels[idx_test])
    acc_test = accuracy(y_pre[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test


##SGC    
def propagate(feature, A, order, alpha):
    y = feature
    for i in range(order):
        y = (1 - alpha) * torch.spmm(A, y).detach_() + alpha * y
        
    return y.detach_()
    
if args.encoder == 'SGC':
    features = propagate(features, adj, 2, 0.)
    
#main 
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
#data.edge_index = data.edge_index.cuda()



test_acc = torch.zeros(times)
test_acc = test_acc.cuda()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

for i in range(times):


    index_train = []
    index_val = []
    
    for i_label in range(data.y.max()+1):
        index_sub = [i for i,x in enumerate(data.y) if x==i_label]#train/val index
        index_sub = random.sample(index_sub,60)
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


    best_model = None
    best_val_acc = 0.0
    # Model and optimizer
    if args.encoder == 'GCN':
        model = GCN(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout,
                    tau = args.tau).cuda()
    else:
        model = MLP(nfeat=features.shape[1],
                    nhid=args.hidden,
                    nclass=labels.max().item() + 1,
                    dropout=args.dropout,
                    tau = args.tau).cuda()
                    
    optimizer = optim.Adam(model.parameters(),
                lr=args.lr, weight_decay=args.weight_decay)
    
    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(model, optimizer, epoch, features, adj, idx_train, idx_val, labels, args.data_aug, args.encoder_type, args.debias, args.kk, args.sample_size, args.neg_type, args.pos_type)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    # Testing
    test_acc[i] = test(best_model, features, adj, labels, idx_test, args.encoder_type)


print("=== Final ===")
print(torch.max(test_acc))
print(torch.min(test_acc))
#print("30次平均",torch.mean(test_acc))
#print("30次标准差",test_acc.std())
#print("20次平均",torch.mean(test_acc[:20]))
#print("20次标准差",test_acc[:20].std())
print("10次平均",torch.mean(test_acc))
print("10次标准差",test_acc.std())
#import ipdb;ipdb.set_trace()

print(test_acc)