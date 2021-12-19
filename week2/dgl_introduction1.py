"""
Node Classification with DGL
============================

GNNs are powerful tools for many machine learning tasks on graphs. In
this introductory tutorial, you will learn the basic workflow of using
GNNs for node classification, i.e. predicting the category of a node in
a graph.

By completing this tutorial, you will be able to

-  Load a DGL-provided dataset.
-  Build a GNN model with DGL-provided neural network modules.
-  Train and evaluate a GNN model for node classification on either CPU
   or GPU.

This tutorial assumes that you have experience in building neural
networks with PyTorch.

(Time estimate: 13 minutes)

"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


# Loading Cora Dataset
# --------------------
# 

import dgl.data

dataset = dgl.data.CoraGraphDataset()
print('Number of categories:', dataset.num_classes)


######################################################################
# 一个DGL数据集对象可能包含一个或多个图形。本教程中使用的 Cora 数据集仅包含一个图形

g = dataset[0]

######################################################################
# A DGL graph can store node features and edge features in two
# dictionary-like attributes called ``ndata`` and ``edata``.
# In the DGL Cora dataset, the graph contains the following node features:
# 
# - ``train_mask``: A boolean tensor indicating whether the node is in the
#   training set.
#
# - ``val_mask``: A boolean tensor indicating whether the node is in the
#   validation set.
#
# - ``test_mask``: A boolean tensor indicating whether the node is in the
#   test set.
#
# - ``label``: The ground truth node category.
#
# -  ``feat``: The node features.
# 

print('Node features')
print(g.ndata)
print('Edge features')
print(g.edata)


######################################################################
# 定义图卷积网络GCN，两层GCN
# 

from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h) 

        return h
    
######################################################################
#训练GCN
# 

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(101):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
train(g, model)


######################################################################
# Training on GPU
# ---------------
# 
# Training on GPU requires to put both the model and the graph onto GPU
# with the ``to`` method, similar to what you will do in PyTorch.
# 
# .. code:: python
#
#    g = g.to('cuda')
#    model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes).to('cuda')
#    train(g, model)
#


