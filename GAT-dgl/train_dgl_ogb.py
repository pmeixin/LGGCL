from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from models_dgl_cl import GAT
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, CitationFull,WikiCS, Coauthor, Amazon
import os.path as osp
import scipy.sparse as sp
import clustering

import argparse
import networkx as nx
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import DGLGraph

from ogb.nodeproppred import DglNodePropPredDataset

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='False pygGAT; True spGAT')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--weight', type=float, default=0.8, help='loss_cl weight')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--out_heads', type=int, default=1, help='Number of head attentions.')
parser.add_argument('--dropout1', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout2', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dataset', type=str, default='Cora', help='Cora/CiteSeer/PubMed/')
parser.add_argument("--attn-drop", type=float, default=.6, help="attention dropout")
parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
parser.add_argument("--gpu", type=int, default=-1, help="which GPU to use. Set -1 to use CPU.")
parser.add_argument('--sample_size', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--debias', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'], default='Kmeans', help='clustering algorithm (default: Kmeans)')
parser.add_argument('--nmb_cluster','--k', type=int, default=7, help='number of cluster for k-means (default:10000)')
parser.add_argument('--verbose', action='store_true', help='chatty')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#dimension feature
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

#load dataset
if args.dataset == 'cora':
    data = CoraGraphDataset()
elif args.dataset == 'citeseer':
    data = CiteseerGraphDataset()
elif args.dataset == 'pubmed':
    data = PubmedGraphDataset()
elif args.dataset == 'ogbn-arxiv':
    data = DglNodePropPredDataset(name = 'ogbn-arxiv')
else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))

#data processing
split_idx = data.get_idx_split()
g, labels = data[0]
labels = labels.squeeze()
features = g.ndata['feat']
g = dgl.to_bidirected(g)
idx_train = split_idx['train']
idx_val = split_idx['valid']
idx_test = split_idx['test']
num_feats = features.shape[1]
n_classes = (labels.max() + 1).item()
n_edges = g.number_of_edges()
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
n_edges = g.number_of_edges()
heads = ([args.nb_heads] * 1) + [args.out_heads]

#training process
def train(epoch, model, optimizer, features, labels, idx_train, idx_val, idx_test, sample_size, debias, kk: int = 1):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    #supervised CE loss
    output, _ = model(features) 
    test_mask_logits = output
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    
    #sample
    node_mask = torch.empty(features.shape[0],dtype=torch.float32).uniform_(0,1).cuda()
    node_mask = node_mask < sample_size#'True' means that this node is selected to participate in training
    
    #positive selection
    y_pre = output.detach()
    y_pre_p = y_pre[node_mask].cuda()
    _, y_topind = torch.topk(y_pre_p, kk)
    y_ol = torch.zeros(y_pre_p.shape).cuda()
    y_ol = y_ol.scatter_(1, y_topind, 1)
    out_pos_mask = torch.mm(y_ol, y_ol.T) > 0#'True' means that the node pair has the same pseudo-label
    out_pos_mask = out_pos_mask.cuda()
    del y_ol, y_topind
    torch.cuda.empty_cache()

    #clustering
    node_mask_2 = torch.empty(features.shape[0], dtype=torch.float32).uniform_(0, 1).cuda()
    node_mask_2 = node_mask_2 <= 1
    deepcluster = clustering.__dict__[args.clustering](y_pre.shape[1])
    clustering_index = deepcluster.cluster(features, verbose=args.verbose)
    clustering_index = torch.LongTensor(clustering_index)
    clustering_index_np = torch.zeros(features.shape[0], y_pre.shape[1])
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
    y_pre_n = y_pre[node_mask].cuda()       
    _, y_poslabel = torch.topk(y_pre_n, kk)
    y_pl = torch.zeros(y_pre_n.shape).cuda()
    y_pl = y_pl.scatter_(1, y_poslabel, 1)
    out_neg_mask = torch.mm(y_pl, y_pl.T) <= 0#'True' means that the node pair has different pseudo-label
    out_neg_mask = out_neg_mask.cuda()      
    del y_pl, y_poslabel
    torch.cuda.empty_cache()   

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
    features1 = drop_feature(features, 0.3)
    features2 = drop_feature(features, 0.4)
    _, out1 = model(features1)
    _, out2 = model(features2)
    del features1, features2
    torch.cuda.empty_cache()
    loss_cl = model.cl_lossaug(out1, out2, None, node_mask, labels, neg_mask, pos_mask, 0, 1, debias)
    
    #trainning loss 
    if epoch<=90:
        loss = loss_train + 0.0001 * loss_cl
    else:
        loss = loss_train + args.weight * loss_cl
    
    loss.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output, _ = model(features)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'loss_cl: {:.4f}'.format(loss_cl.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'acc_test: {:.4f}'.format(acc_test.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item(), acc_test.data.item(), test_mask_logits.detach().cpu().numpy()

def compute_test(model, features, idx_test):
    model.eval()
    output = model(features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
          
    return acc_test.item()

times = 10
test_acc = torch.zeros(times)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

for i in range(times):
    # Model and optimizer
    if args.gpu < 0:
        args.cuda = False
    else:
        args.cuda = True
        g = g.int().to(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    if args.sparse:
        model = SpGAT(nfeat=features.shape[1], 
                    nhid=args.hidden, 
                    nclass=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    nheads=args.nb_heads, 
                    alpha=args.alpha)
    else:
        model = GAT(g,
                    1,
                    num_feats, 
                    args.hidden,
                    n_classes,
                    heads,
                    F.elu,
                    args.dropout1, 
                    args.dropout2, 
                    args.attn_drop,
                    args.alpha,
                    args.residual)
    optimizer = optim.Adam(model.parameters(), 
                           lr=args.lr, 
                           weight_decay=args.weight_decay)
    
    if args.cuda:
        model.cuda()

    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    best_acc = 0
    best_test_logits = 0
    for epoch in range(args.epochs):
        loss_e, acc_e, test_logits = train(epoch, model, optimizer, features, labels, idx_train, idx_val, idx_test, args.sample_size, args.debias)
        loss_values.append(loss_e)
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
            best_acc = acc_e
            best_test_logits = test_logits
        else:
            bad_counter += 1
    
        if bad_counter == args.patience:
            break
    
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    
    # Testing
    print(best_acc)
    test_acc[i] = best_acc

tsne = TSNE()
test_label = labels
    
out = tsne.fit_transform(test_logits)
fig = plt.figure()
for i in range(7):
    indices = test_label == i
    x, y = out[indices.cpu().numpy()].T
    plt.scatter(x, y, label=str(i))
plt.savefig('cora_LGGCL_gat.png')
plt.show()

print("=== Final ===")
print(torch.max(test_acc))
print(torch.min(test_acc))
print("10次平均",torch.mean(test_acc))
print("10次标准差",test_acc.std())

print(test_acc)