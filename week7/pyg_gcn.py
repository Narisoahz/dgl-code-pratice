import os
from sklearn.metrics import accuracy_score
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch_geometric.datasets import Planetoid
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from torch.utils import tensorboard
import optuna
from optuna.trial import TrialState
import torch.optim as optim
from optuna.visualization import plot_optimization_history

class GCN(torch.nn.Module):
    def __init__(self,in_channels,out_channels,trial):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.num_layers = trial.suggest_int("num_layers", 1, 3)
        self.trial=trial
        
        if self.num_layers==1:
            self.convs.append(GCNConv(in_channels, out_channels,add_self_loops=True,normalize=True))
        else:
            for i in range(self.num_layers-1):
                hidden_channels = trial.suggest_int("n_units_l{}".format(i), 16, 128)  
                self.convs.append(GCNConv(in_channels, hidden_channels,add_self_loops=True,normalize=True))
                in_channels=hidden_channels
            self.convs.append(GCNConv(in_channels,out_channels,add_self_loops=True,normalize=True))
        
    def forward(self, x,edge_index):
        h=x
        for i,conv in enumerate(self.convs):
            h = conv(h,edge_index)
            if i!=self.num_layers-1:
                h = F.relu(h)
                p = self.trial.suggest_float("dropout_l{}".format(i), 0, 1)
                h = F.dropout(h,p, training=self.training)
        return h


def objective(trial):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = Planetoid(name="Cora", root="./data")
    dataset = data[0]
    dataset.to(device)
    model = GCN(dataset.num_features,data.num_classes,trial).to(device)
    
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay=trial.suggest_float("weight_decay",5e-4,5e-1,log=True)
    
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr,weight_decay=weight_decay)
        
    for epoch in range(1,201):
        model.train()
        optimizer.zero_grad()
        out = model(dataset.x,dataset.edge_index)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()
        

        model.eval()
        with torch.no_grad():
            pred = model(dataset.x,dataset.edge_index).argmax(dim=1)           
            correct = (pred[dataset.val_mask] == dataset.y[dataset.val_mask]).sum()
            accuracy = int(correct) / int(dataset.val_mask.sum())                            
        # trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()        
    
    model.eval()
    pred = model(dataset.x,dataset.edge_index).argmax(dim=1)
    correct = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
    accuracy = int(correct) / int(dataset.test_mask.sum())
    print("testacc {:.4f}".format(accuracy))
    trial.report(accuracy, epoch)
    if trial.should_prune():
            raise optuna.exceptions.TrialPruned()  

    return accuracy
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    try:
        args = parser.parse_args()
    except IOError as msg:
        parser.error(str(msg))
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()

    fig=optuna.visualization.plot_parallel_coordinate(study, params=["dropout_l0", "weight_decay"])
    fig.show()

    fig=optuna.visualization.plot_slice(study)
    fig.show()

    fig=optuna.visualization.plot_param_importances(study)
    fig.show()
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    