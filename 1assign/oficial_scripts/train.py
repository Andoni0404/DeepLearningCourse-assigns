import torch
import torch.nn as nn
import optuna
from torch.utils.tensorboard import SummaryWriter

def train_and_evaluate(model, optimizer, criterion, train_loader, data, trial, max_epochs, device):
    best_val_mae = float("inf")
    X_val_device = data["X_val"].to(device)
    y_val_device = data["y_val"].to(device)

    writer = SummaryWriter(log_dir=f"runs/optuna_study/trial_{trial.number}")

    for epoch in range(max_epochs):
        model.train()
        running_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_device)
            val_mae = nn.L1Loss()(val_preds, y_val_device).item()
            val_mse = nn.MSELoss()(val_preds, y_val_device).item()
            
            # Cálculo de R² Score
            y_bar = torch.mean(y_val_device)
            ss_tot = torch.sum((y_val_device - y_bar) ** 2)
            ss_res = torch.sum((y_val_device - val_preds) ** 2)
            val_r2 = (1 - ss_res / ss_tot).item()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            # Guardamos estas métricas para el reporte final
            trial.set_user_attr("best_mse", val_mse)
            trial.set_user_attr("best_r2", val_r2)

        writer.add_scalar("MAE/Validation", val_mae, epoch)
        
        trial.report(val_mae, epoch)
        if trial.should_prune():
            writer.close()
            raise optuna.TrialPruned()

    writer.close()
    return best_val_mae