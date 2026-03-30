import random
import warnings
import numpy as np
import torch
import optuna

# Importamos de nuestros otros archivos
from preprocess import get_data
from model import build_model, build_optimizer, build_loss
from train import train_and_evaluate

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
SEED = 42
MAX_EPOCHS = 300 
N_TRIALS = 150 # 150 intentos son suficientes para una red pequeña
STUDY_NAME = "insurance_lightweight_v2_changeNumeroCapas" # Nombre nuevo para empezar de cero
STORAGE = "sqlite:///optuna_insurance.db"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Inicializamos semilla y cargamos datos (con el nuevo Feature Engineering)
set_seed(SEED)
data = get_data(seed=SEED)

def objective(trial: optuna.Trial) -> float:
    # 1. Tamaño del batch
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # 2. Dataloader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["X_train"], data["y_train"]),
        batch_size=batch_size, shuffle=True
    )

    # 3. Construimos los componentes usando nuestro model.py actualizado
    model = build_model(trial, data["input_size"], DEVICE)
    optimizer = build_optimizer(trial, model)
    criterion = build_loss(trial)

    # 4. Entrenamos y devolvemos el MAE de validación
    return train_and_evaluate(model, optimizer, criterion, train_loader, data, trial, MAX_EPOCHS, DEVICE)

def run_optimization():
    # Sampler TPE Multivariante: Robusto y eficiente
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=False)
    # Pruner para cancelar trials malos rápido
    pruner = optuna.pruners.HyperbandPruner(min_resource=10, max_resource=MAX_EPOCHS, reduction_factor=3)

    study = optuna.create_study(
        direction="minimize", 
        study_name=STUDY_NAME,
        storage=STORAGE, 
        load_if_exists=True, # Si se corta, retoma donde lo dejó
        sampler=sampler, 
        pruner=pruner
    )

    # Le damos una pista inicial muy ligera (2 capas, pocas neuronas)
    study.enqueue_trial({
        "lr": 0.00206714653863352, 
        "batch_size": 32, 
        "n_layers": 3, 
        "dropout_rate": 1.8973456760486675e-05,
        "n_units_l0": 26, 
        "n_units_l1": 40, 
        "n_units_l2": 52,
        "optimizer": "Adam", 
        "weight_decay": 8.887346491178652e-05, 
        "loss_fn": "Huber"
    }, skip_if_exists=True)

    print(f"\nIniciando búsqueda rápida con redes ligeras en dispositivo: {DEVICE}...")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    # REPORTE FINAL
    print("\n" + "="*50)
    print("🏆 RESULTADOS FINALES 🏆")
    print("="*50)
    best = study.best_trial
    print(f"Mejor MAE: {study.best_value:.2f}")
    print(f"Mejor MSE: {best.user_attrs.get('best_mse', 0):.2f}")
    print(f"Mejor R²:  {best.user_attrs.get('best_r2', 0):.4f}")
    print("-" * 50)
    print("Mejor Configuración:", study.best_params)
    print("="*50)

    return study