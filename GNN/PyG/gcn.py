#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter & Optimizer & GDC

# In[ ]:


# import packages

import os.path as osp
import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv


# ## Parameter Setting

# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
          help='Use GDC preprocessing.')
parser.add_argument('--seed',type=int,default='100',help='Random seed')
parser.add_argument('--no-cuda', action='store_true', default=False,
          help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
          help='Validate during training pass.')
# hidden = [16, 32, 64 ,128]
parser.add_argument('--hidden', type=int, default=32,
          help='Number of hidden units.')
# dropput = [0.3, 0.5, 0.8]
parser.add_argument('--dropout', type=float, default=0.5,
          help='Dropout rate (1 - keep probability).')                        
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


# In[ ]:


# Set seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

# Load data
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]


# ## GDC

# In[ ]:


# Whether use GDC
if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                dim=0), exact=True)
    data = gdc(data)


# ## GCN Structure

# In[ ]:


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden, 
        cached=True, normalize=not args.use_gdc)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes, 
        cached=True, normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

#         self.reg_params = self.conv1.parameters()
#         self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)


# ## Optimizers

# In[ ]:


# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.02,weight_decay = 5e-4) 
#lr=[0.005, 0.01, 0.02,0.03]; weight_decay=[0, 5e-4]

# optimizer = torch.optim.Adadelta(params=model.parameters(), lr=3.2, weight_decay=5e-4)
#lr=[1, 1.5, 2, 2.3, 2.5, 2.8, 3, 3.2, 3.5]; weight_decay=[0, 5e-4]

# optimizer = torch.optim.Adagrad(params=model.parameters(), lr=0.04,weight_decay=5e-4)
#lr=[0.005, 0.01, 0.02,0.03,0.04,0.05]; weight_decay=[0, 5e-4]

# optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.01, weight_decay=5e-4, amsgrad=False)
#lr=[0.005, 0.01, 0.02,0.03,0.04]; weight_decay=[0, 5e-4]

optimizer = torch.optim.RMSprop(params=model.parameters(), lr=0.01,weight_decay=5e-4, centered=False)
#lr=[0.005, 0.01, 0.015, 0.02,0.03,0.04]; weight_decay=[0, 5e-4]; centered=[True, Flase]

# optimizer = torch.optim.Rprop(params=model.parameters(), lr=0.02)
#lr=[0.005, 0.01, 0.02,0.03,0.04,0.05]; weight_decay=[0, 5e-4]

# optimizer = torch.optim.SGD(params=model.parameters(), lr=0.15, momentum=0.9, weight_decay=5e-4)
#lr=[0.005, 0.01, 0.02,0.03,0.04,0.05]; weight_decay=[0, 5e-4]; momentum=[0,0.1,0.9]


# In[ ]:


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


# In[ ]:


best_val_acc = test_acc = 0
t = time.time()

for epoch in range(1, 401):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    if epoch%5 == 0:
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t))

