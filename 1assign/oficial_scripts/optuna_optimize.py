import random
import warnings
import numpy as np
import torch
import optuna

from preprocess import get_data
from model import build_model, build_optimizer, build_loss
from train import train_and_evaluate

warnings.filterwarnings("ignore", category=UserWarning)

# CONFIGURATION
SEED = 42
MAX_EPOCHS = 300 
N_TRIALS = 200
STUDY_NAME = "insurance_power_features" 
STORAGE = "sqlite:///optuna_insurance.db"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
data = get_data(seed=SEED)

def objective(trial: optuna.Trial) -> float:
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size, shuffle=True
    )

    model = build_model(trial, data["input_size"], DEVICE)
    optimizer = build_optimizer(trial, model)
    criterion = build_loss(trial)

    return train_and_evaluate(model, optimizer, criterion, train_loader, data, trial, MAX_EPOCHS, DEVICE)

def run_optimization():
    # Sampler TPE Multivariante: Robusto y eficiente
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    pruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource=MAX_EPOCHS, reduction_factor=3)

    study = optuna.create_study(
        direction="minimize", study_name=STUDY_NAME,
        storage=STORAGE, load_if_exists=False,
        sampler=sampler, pruner=pruner
    )

    # Tu mejor trial conocido para guiar al TPE
    study.enqueue_trial({
        "lr": 0.0074, "batch_size": 64, "n_layers": 5, "dropout_rate": 0.0,
        "n_units_l0": 11, "n_units_l1": 30, "n_units_l2": 38, "n_units_l3": 60,
        "n_units_l4": 61, "optimizer": "Adam", "weight_decay": 1e-4, "loss_fn": "Huber"
    }, skip_if_exists=True)

    print(f"\nIniciando búsqueda estable con TPE...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # REPORTE FINAL
    print("\n" + "="*50)
    print("RESULTADOS FINALES")
    print("="*50)
    best = study.best_trial
    print(f"Mejor MAE: {study.best_value:.2f}")
    print(f"Mejor MSE: {best.user_attrs.get('best_mse', 0):.2f}")
    print(f"Mejor R²:  {best.user_attrs.get('best_r2', 0):.4f}")
    print("-" * 50)
    print("Configuración:", study.best_params)
    print("="*50)

    return study

