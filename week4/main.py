import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
from model import GCN
from load_graph import inductive_split, load_ogb
from sampler import MultiLayerRandomSampler

def setup_features(block,feat,device):
    feat=feat.to(device)
    block=block.to(device)
    nodeID=block.srcdata[dgl.NID]
    block.srcdata['feat']=feat[nodeID]
    block.dstdata['feat']=feat[nodeID]


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    提取节点子集的特征和标签 
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))


def run(args, device, data):
    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data
    in_feats = train_nfeat.shape[1]

    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    dataloader_device = th.device('cpu')
    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        dataloader_device = device

    train_res=[]
    test_res=[]
    val_res=[]
    for _ in range(args.runs):
        
        # Create PyTorch DataLoader for constructing blocks
        # sampler = dgl.dataloading.neighbor.MultiLayerNeighborSampler(
        #     [int(fanout) for fanout in args.fan_out.split(',')])
        sampler = MultiLayerRandomSampler(
            [int(fanout) for fanout in args.fan_out.split(',')])
        dataloader = dgl.dataloading.NodeDataLoader(
            train_g,
            train_nid,
            sampler,
            device=dataloader_device,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)

        # Define model and optimizer 
        model=GCN(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
        model = model.to(device)
        loss_fcn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Training loop
        for epoch in range(args.num_epochs):
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                            seeds, input_nodes, device)
                # blocks=setup_features(blocks,g.ndata)
                blocks = [block.int().to(device) for block in blocks]
                model.train()
                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} '.format(
                    epoch, step, loss.item(), acc.item()))
            if epoch % args.eval_every == 0 and epoch != 0:
                eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device)
                print('Eval Acc {:.4f}'.format(eval_acc))
                test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device)
                print('Test Acc: {:.4f}'.format(test_acc))

        train_acc=evaluate(model, train_g, train_nfeat, train_labels, train_nid, device)
        eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device)
        test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device)
        train_res.append(train_acc)
        test_res.append(test_acc)
        val_res.append(eval_acc)
    print(f'train_Accuarcy {np.mean(train_res):.4f} ± {np.std(train_res):.4f}')
    print(f'test_Accuarcy {np.mean(test_res):.4f} ± {np.std(test_res):.4f}')
    print(f'val_Accuarcy {np.mean(val_res):.4f} ± {np.std(val_res):.4f}')



    

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=500)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='5,10')
    argparser.add_argument('--batch-size', type=int, default=1024)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.0005)
    argparser.add_argument('--runs', type=int, default=1)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    g, n_classes = load_ogb('ogbn-arxiv')
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        test_nfeat = test_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels')
        val_labels = val_g.ndata.pop('labels')
        test_labels = test_g.ndata.pop('labels')
    else:
        train_g = val_g = test_g = g
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
        train_labels = val_labels = test_labels = g.ndata.pop('labels')

    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)
    
    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels

    run(args, device, data)