import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch_geometric.datasets import Planetoid
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch.utils import tensorboard

class GCN(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.num_layers=num_layers
        if num_layers==1:
            self.convs.append(GCNConv(in_channels, out_channels,add_self_loops=True,normalize=True))
        else:    
            self.convs.append(GCNConv(in_channels, hidden_channels,add_self_loops=True,normalize=True))
            for i in range(num_layers-2):
                self.convs.append(GCNConv(hidden_channels,hidden_channels,add_self_loops=True,normalize=True))
            self.convs.append(GCNConv(hidden_channels,out_channels,add_self_loops=True,normalize=True))
        self.dropout=torch.nn.Dropout(dropout)

    def forward(self, x,edge_index):
        h=x
        for i,conv in enumerate(self.convs):
            h = conv(h,edge_index)
            if i!=self.num_layers-1:
                h = F.relu(h)
                h=self.dropout(h)
        return h

class GCNGAT(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels,dropout):
        super().__init__()
        self.layer1 = GCNConv(in_channels, hidden_channels,add_self_loops=True,normalize=True)
        self.layer2 = GATConv(hidden_channels,out_channels,dropout=0.5)  
        self.dropout=torch.nn.Dropout(dropout)

    def forward(self, x,edge_index):
        x=self.layer1(x,edge_index)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.layer2(x,edge_index)
        return x

class GATGCN(torch.nn.Module):
    def __init__(self,in_channels, hidden_channels, out_channels,dropout):
        super().__init__()
        self.layer1 = GATConv(in_channels, hidden_channels,dropout=0.5) 
        self.layer2 = GCNConv(hidden_channels,out_channels,add_self_loops=True,normalize=True)
        self.dropout=torch.nn.Dropout(dropout)

    def forward(self, x,edge_index):
        x=self.layer1(x,edge_index)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.layer2(x,edge_index)
        return x

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)
    
    f = "lr{}-dp{}-decay{}-layers{}-hidden{}-model{}".format(args.lr, args.dropout, args.weight_decay,args.layers,args.hidden,args.model)
    os.mkdir(os.path.join(args.logs, f))
    log_dir = os.path.join(args.logs, f)

    data = Planetoid(name=args.data, root=args.path)
    dataset = data[0]
    
    res=[]
    for i in range(args.run):
        if args.model == 'GCN':
            model = GCN(dataset.num_features, args.hidden, data.num_classes, args.layers, args.dropout)
        elif args.model == 'GCNGAT':
            model=GCNGAT(dataset.num_features, args.hidden, data.num_classes,args.dropout)
        elif args.model == 'GATGCN':
            model=GCNGAT(dataset.num_features, args.hidden, data.num_classes,args.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        if args.run > 1:
            os.mkdir(os.path.join(args.logs, f, 'run_'+str(i)))
            log_dir = os.path.join(args.logs, f, 'run_'+str(i))
        writer = tensorboard.SummaryWriter(log_dir=log_dir)
        
        for epoch in range(1,args.epoch+1):
            model.train()
            optimizer.zero_grad()
            out = model(dataset.x,dataset.edge_index)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_loss', loss.data.cpu().numpy(), global_step=epoch)

            model.eval()
            with torch.no_grad():
                pred = model(dataset.x,dataset.edge_index).argmax(dim=1)           
                correct = (pred[dataset.val_mask] == dataset.y[dataset.val_mask]).sum()
                valacc = int(correct) / int(dataset.val_mask.sum())                            
                writer.add_scalar('val_acc', valacc, global_step=epoch)
                
        model.eval()
        pred = model(dataset.x,dataset.edge_index).argmax(dim=1)
        correct = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
        testacc = int(correct) / int(dataset.test_mask.sum())
        res.append(testacc)
        print("# run {}: testacc {:.4f}".format(i, testacc))

    print(f'average Accuarcy: {np.mean(res):.4f} ± {np.std(res):.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='the path for loading dataset',
                        type=str, default='./data')
    parser.add_argument('--logs', type=str, default='./logs', help='the path to save model and results')                    
    parser.add_argument('--data', help='the name of dataset',
                        type=str,default="Cora")
    parser.add_argument('--lr',type=float,default=1e-2)    
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)       
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--run', type=int, default=10)
    parser.add_argument("--model", type=str, default="GATGCN")
    parser.add_argument("--hidden", type=int, default=16)         
    parser.add_argument("--layers", type=int, default=2, help="number of hidden gcn layers")
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    
    main(args)
    