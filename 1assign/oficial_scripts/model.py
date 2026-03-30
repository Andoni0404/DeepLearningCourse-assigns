import torch
import torch.nn as nn
import torch.optim as optim
import optuna

def build_model(trial: optuna.Trial, input_size: int, device: torch.device) -> nn.Module:
    layers = []
    
    n_layers = trial.suggest_int("n_layers", 1, 3) 
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.1)

    in_features = input_size
    for i in range(n_layers):
        # Y máximo 64 neuronas por capa
        out_features = trial.suggest_int(f"n_units_l{i}", 16, 64)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        in_features = out_features

    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers).to(device)
def build_optimizer(trial: optuna.Trial, model: nn.Module):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

def build_loss(trial: optuna.Trial):
    loss_name = trial.suggest_categorical("loss_fn", ["L1", "MSE", "Huber"])
    if loss_name == "L1":
        return nn.L1Loss()
    elif loss_name == "MSE":
        return nn.MSELoss()
    else:
        return nn.HuberLoss(delta=1.0)