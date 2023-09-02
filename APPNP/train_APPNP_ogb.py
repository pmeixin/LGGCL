import arguments
import numpy as np

import torch
import torch.nn.functional as F

from utils import accuracy, load_data, sparse_mx_to_torch_sparse_tensor
from models import APPNP
import random
from early_stop import EarlyStopping, Stop_args

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import os.path as osp
from torch_geometric.datasets import Planetoid, CitationFull, WikiCS, Coauthor, Amazon
import torch_geometric as pyg
import torch_geometric.transforms as T
import scipy.sparse as sp
import clustering

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

def aug_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   row_sum = np.array(adj.sum(1))
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

#adj normalization
def adj_nor(edge):
    degree = torch.sum(edge, dim=1)
    degree = 1 / torch.sqrt(degree)
    degree = torch.diag(degree)
    adj = torch.mm(torch.mm(degree, edge), degree)
    return adj

#Row-normalize sparse matrix
def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

args = arguments.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load dataset
def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP','WikiCS','Amazon-Photo', 'ogbn-arxiv']
    name = 'dblp' if name == 'DBLP' else name
    print(name)
    if name == 'ogbn-arxiv': 
        return PygNodePropPredDataset(root=path, name=name, transform=T.NormalizeFeatures())
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
split_idx = dataset.get_idx_split()
idx_train = split_idx['train']
idx_val = split_idx['valid']
idx_test = split_idx['test']
features = data.x
features = normalize(features)
features = torch.from_numpy(features)
labels = data.y
labels = torch.squeeze(labels, 1)
adj = torch.eye(data.x.shape[0])
for i in range(data.edge_index.shape[1]):
    adj[data.edge_index[0][i]][data.edge_index[1][i]] = 1
adj = adj.float()
adj = aug_normalized_adjacency(adj)
adj = sparse_mx_to_torch_sparse_tensor(adj)

#training process
def train(epoch, sample_size, debias, kk: int = 1):
    model.train()
    optimizer.zero_grad()

    #supervised CE loss  
    output,_ = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + args.weight_decay * torch.sum(model.Linear1.weight ** 2) / 2
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
    _, out1 = model(features1, adj)
    _, out2 = model(features2, adj)
    del features1, features2
    torch.cuda.empty_cache()
    loss_cl = model.cl_lossaug(out1, out2, None, node_mask, labels, neg_mask, pos_mask, 0, 1, debias)

    #trainning loss 
    if epoch<=200:
        loss = loss_train + 0.0001 * loss_cl
    else:
        loss = loss_train + 0.8 * loss_cl
    
    loss.backward()
    optimizer.step()

    # Evaluate validation set performance separately,
    model.eval()
    output,_ = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'loss_cl: {:.4f}'.format(loss_cl.data.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'loss_val: {:.4f}'.format(loss_val.item()),
        'acc_val: {:.4f}'.format(acc_val.item()))

    return loss_val.item(), acc_val.item()

def test():
    model.eval()
    output,_ = model(features, adj)
    test_mask_logits = output
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item(), test_mask_logits.detach().cpu().numpy()

times = 3
test_acc = torch.zeros(times)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.ini_seed)
if torch.cuda.is_available(): 
    torch.cuda.manual_seed(args.ini_seed)

for i in range(times):
    # Model and optimizer
    model = APPNP(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout, 
                K=args.K, 
                alpha=args.alpha,
                tau = 0.4).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),
                           lr=args.lr)
    
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    
    stopping_args = Stop_args(patience=args.patience, max_epochs=args.epochs)
    early_stopping = EarlyStopping(model, **stopping_args)
    for epoch in range(args.epochs):
        loss_val, acc_val = train(epoch, args.sample_size, args.debias)
        if early_stopping.check([acc_val, loss_val], epoch):
            break
    
    print("Optimization Finished!")
    
    # Restore best model
    print('Loading {}th epoch'.format(early_stopping.best_epoch))
    model.load_state_dict(early_stopping.best_state)
    test_acc[i], test_logits = test()

tsne = TSNE()
test_label = labels
    
out = tsne.fit_transform(test_logits)
fig = plt.figure()
for i in range(7):
    indices = test_label == i
    x, y = out[indices.cpu().numpy()].T
    plt.scatter(x, y, label=str(i))
plt.savefig('cora_LGGCL_appnp.png')
plt.show()

print("=== Final ===")
print(torch.max(test_acc))
print(torch.min(test_acc))
print("10次平均",torch.mean(test_acc))
print("10次标准差",test_acc.std())

print(test_acc)
