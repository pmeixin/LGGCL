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
from models import GCN, MLP

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
parser.add_argument('--kk', type=int, default=1,
                    help='y_pre select k')
parser.add_argument('--sample_size', type=float, default=0.,
                    help='sample size')
parser.add_argument('--neg_type', type=float, default=0,
                    help='0,selection;1 not selection')
parser.add_argument('--pos_type', type=float, default=0,
                    help='0,selection;1 not selection')
parser.add_argument('--encoder_type', type=int, default=3,
                    help='do data augmentation.')
parser.add_argument('--debias', type=float, default=0.12,
                    help='debias rate.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
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
def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP','WikiCS','Amazon-Photo']
    name = 'dblp' if name == 'DBLP' else name
    print(name)
    return (CitationFull if name == 'dblp' else Planetoid)(path,name,transform=T.NormalizeFeatures())

if args.dataset=='Cora' or args.dataset=='CiteSeer' or args.dataset=='PubMed':
    path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
    print(path)
else:
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
dataset = get_dataset(path, args.dataset)    
data = dataset[0]

#data processing
idx_train = data.train_mask
idx_val = data.val_mask
idx_test = data.test_mask
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

#training process
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

    #sample nodes
    node_mask = torch.empty(features.shape[0],dtype=torch.float32).uniform_(0,1).cuda()
    node_mask = node_mask < sample_size#'True' means that this node is selected to participate in training
    
    #positive selection
    if pos_type == 0:
        y_pre = y_pre.detach()
        y_pre_p = y_pre[node_mask].cuda()
        _, y_topind = torch.topk(y_pre_p, kk)
        y_ol = torch.zeros(y_pre_p.shape).cuda()
        y_ol = y_ol.scatter_(1, y_topind, 1)
        out_pos_mask = torch.mm(y_ol, y_ol.T) > 0#'True' means that the node pair has the same pseudo-label
        out_pos_mask = out_pos_mask.cuda()
        del y_ol, y_topind
        torch.cuda.empty_cache()
    else :
        out_pos_mask = torch.eye(node_mask.sum()).cuda()
    
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
    clustering_true_mask = torch.mm(clustering_index_np, clustering_index_np.T) == 1#'True' indicates that the node pair belongs to the same cluster
    clustering_true_mask = clustering_true_mask.cuda()
    clustering_neg_mask = clustering_neg_mask.cuda()

    #self-checking mechanism
    pos_mask = torch.mul(out_pos_mask, clustering_true_mask).cuda()#'True' indicates that the node pair has the same pseudo-label and belongs to the same cluster
    gl = torch.mul(out_pos_mask, clustering_true_mask).cuda()    
    flb = torch.sum(out_pos_mask==1)
    fla = torch.sum(gl==1)
    sccs = flb-fla
    sccs.detach().cpu().numpy()
    sccs = sccs.tolist()
    fp = open('/home/juanxin/'+args.dataset+'filter.txt','r+')
    n = fp.read()
    fp.write(str(sccs)+",")

    #negative selection
    if neg_type == 0:
        y_pre = y_pre.detach()
        y_pre_n = y_pre[node_mask].cuda()       
        _, y_poslabel = torch.topk(y_pre_n, kk)
        y_pl = torch.zeros(y_pre_n.shape).cuda()
        y_pl = y_pl.scatter_(1, y_poslabel, 1)
        out_neg_mask = torch.mm(y_pl, y_pl.T) <= 0#'True' means that the node pair has different pseudo-label
        out_neg_mask = out_neg_mask.cuda()      
        del y_pl, y_poslabel
        torch.cuda.empty_cache()    
    else :
        out_neg_mask = (1 - torch.eye(node_mask.sum())).cuda()

    #reweighting negative ndoes
    y_soft_pre = F.softmax(y_pre[node_mask], dim=1).cuda()
    y_pre_mean = torch.mean(y_soft_pre, dim=1, keepdim=True).cuda()
    y_pre_std = torch.std(y_soft_pre, dim=1, keepdim=True).cuda()
    neg_weight = torch.zeros(y_soft_pre.shape).cuda()
    for i in range(neg_weight.shape[0]):
        neg_weight[i] = torch.add(y_soft_pre[i], torch.neg(y_pre_mean[i])) * torch.add(y_soft_pre[i], torch.neg(y_pre_mean[i]))
        neg_weight[i] = torch.neg(neg_weight[i] / 2*(y_pre_std[i]))
        neg_weight[i] = torch.exp(neg_weight[i])
    neg_mask = out_neg_mask
    neg_mask = neg_mask.to(torch.float64)
    for i in range(neg_mask.shape[0]):
        neg_label_ind = torch.max(y_pre[node_mask], dim=1, keepdim=True)[1].T.cuda()
        out = torch.take(neg_weight[i], neg_label_ind).cuda()
        neg_mask[i] = neg_mask[i] * out

    #contrastive loss
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
    
    #trainning loss        
    if neg_type == 0 or pos_type == 0:
        if epoch<=50:
            loss = loss_train + 0.0001 * loss_cl
        else:
            loss = loss_train + 0.8 * loss_cl
    else:
        loss = loss_train
    
    loss.backward()
    optimizer.step()
            
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        y_pre, _ = model(features, adj, encoder_type)

    loss_val = F.nll_loss(y_pre[idx_val], labels[idx_val])
    acc_val = accuracy(y_pre[idx_val], labels[idx_val])
    if acc_val > best_val_acc:
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

#encoder is SGC, features update
def propagate1(feature, A, order, alpha):
    y = feature
    out = feature
    for i in range(order):
        y = torch.spmm(A, y).detach_()
        out = out + y
        
    return out.detach_()/(order + 1)
    
def propagate(feature, A, order, alpha):
    y = feature
    for i in range(order):
        y = (1 - alpha) * torch.spmm(A, y).detach_() + alpha * y
        
    return y.detach_()
    
def propagate2(features, adj, degree, alpha):
    ori_features = features
    emb = alpha * features
    for i in range(degree):
        features = torch.spmm(adj, features)
        emb = emb + (1-alpha)*features/degree
    return emb
    
if args.encoder == 'SGC':
    features = propagate(features, adj, 2, 0.)

#main 
features = features.cuda()
adj = adj.cuda()
labels = labels.cuda()
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
data.edge_index = data.edge_index.cuda()

test_acc = torch.zeros(times)
test_acc = test_acc.cuda()

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

for i in range(times):
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
print("10次平均",torch.mean(test_acc))
print("10次标准差",test_acc.std())
fp.close()
print(test_acc)








