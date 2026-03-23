import os
import io
import copy
import random
import warnings

import numpy as np
import pandas as pd
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# 0. CONFIGURACIÓN GENERAL
# =========================================================
SEED = 42
MAX_EPOCHS = 500
PATIENCE = 50
STUDY_NAME = "dl_insurance_extreme_search"
STORAGE = "sqlite:///optuna_insurance.db"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducibilidad
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

# =========================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =========================================================
directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.join(directorio_actual, "../data/insurance.csv")

df = pd.read_csv(ruta_csv)
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("charges", axis=1).values
y = df_encoded["charges"].values

# Estratificación por smoker si existe
stratify_col = df_encoded["smoker_yes"] if "smoker_yes" in df_encoded.columns else None

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=stratify_col
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

input_size = X_train.shape[1]

print(f"Tamaño train: {X_train_tensor.shape}")
print(f"Tamaño val:   {X_val_tensor.shape}")

# =========================================================
# 2. DEFINICIÓN DEL MODELO DINÁMICO
# =========================================================
class InsuranceNet(nn.Module):
    def __init__(self, trial, input_size: int):
        super().__init__()
        self.layers = nn.ModuleList()

        # Se mantiene tu espacio de búsqueda
        n_layers = trial.suggest_int("n_layers", 1, 10)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)

        in_features = input_size
        for i in range(n_layers):
            out_features = trial.suggest_int(f"n_units_l{i}", 8, 64)

            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.BatchNorm1d(out_features))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

            in_features = out_features

        self.layers.append(nn.Linear(in_features, 1))
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# =========================================================
# 3. UTILIDADES DE ENTRENAMIENTO
# =========================================================
def get_optimizer(trial, model):
    lr = trial.suggest_float("lr", 1e-5, 0.1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSprop"])

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    return optimizer


def create_train_loader(trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )
    return train_loader


def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(DEVICE, non_blocking=True)
        batch_y = batch_y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Más estabilidad para arquitecturas agresivas
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        bs = batch_X.size(0)
        running_loss += loss.item() * bs
        total_samples += bs

    return running_loss / total_samples


@torch.no_grad()
def evaluate_loss(model, X_tensor, y_tensor, criterion):
    model.eval()
    X_tensor = X_tensor.to(DEVICE)
    y_tensor = y_tensor.to(DEVICE)

    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor).item()
    return loss


@torch.no_grad()
def predict(model, X_tensor):
    model.eval()
    X_tensor = X_tensor.to(DEVICE)
    preds = model(X_tensor)
    return preds.cpu().numpy().reshape(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================================================
# 4. FUNCIÓN OBJETIVO DE OPTUNA
# =========================================================
def objective(trial):
    train_loader = create_train_loader(trial)
    model = InsuranceNet(trial, input_size=input_size).to(DEVICE)

    criterion = nn.L1Loss()  # MAE
    optimizer = get_optimizer(trial, model)

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        train_mae = train_one_epoch(model, train_loader, optimizer, criterion)
        val_mae = evaluate_loss(model, X_val_tensor, y_val_tensor, criterion)

        # Reporte intermedio para pruning
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 25 == 0 or epoch == MAX_EPOCHS - 1:
            print(
                f"[Trial {trial.number:03d}] "
                f"Epoch {epoch:03d} | "
                f"Train MAE: {train_mae:.4f} | "
                f"Val MAE: {val_mae:.4f} | "
                f"Best: {best_val_mae:.4f}"
            )

        if epochs_no_improve >= PATIENCE:
            break

    # Solo atributos serializables por JSON
    trial.set_user_attr("best_val_mae", float(best_val_mae))
    trial.set_user_attr("best_epoch", int(best_epoch))
    trial.set_user_attr("model_params", int(count_parameters(model)))

    return best_val_mae


# =========================================================
# 5. REENTRENAMIENTO DEL MEJOR MODELO
# =========================================================
def build_model_from_trial(trial):
    return InsuranceNet(trial, input_size=input_size).to(DEVICE)


def retrain_best_model(best_trial):
    print("\nReentrenando el mejor modelo encontrado...")

    model = build_model_from_trial(best_trial)
    optimizer = get_optimizer(best_trial, model)
    criterion = nn.L1Loss()

    train_loader = create_train_loader(best_trial)

    best_val_mae = float("inf")
    best_state_dict = None
    best_epoch = 0
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        train_mae = train_one_epoch(model, train_loader, optimizer, criterion)
        val_mae = evaluate_loss(model, X_val_tensor, y_val_tensor, criterion)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 25 == 0 or epoch == MAX_EPOCHS - 1:
            print(
                f"[FINAL] Epoch {epoch:03d} | "
                f"Train MAE: {train_mae:.4f} | "
                f"Val MAE: {val_mae:.4f} | "
                f"Best: {best_val_mae:.4f}"
            )

        if epochs_no_improve >= PATIENCE:
            break

    model.load_state_dict(best_state_dict)
    return model, best_val_mae, best_epoch


# =========================================================
# 6. MAIN
# =========================================================
if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=20,
        n_warmup_steps=30,
        interval_steps=10
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )

    print("\nIniciando experimento EXTREMO con 500 trials...")
    study.optimize(objective, n_trials=500, show_progress_bar=True)

    print("\n¡Búsqueda EXTREMA completada!")
    print("Mejores parámetros encontrados:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print(f"\nMejor MAE en validación (Optuna): ${study.best_value:.4f}")

    # Reentrenamiento real del mejor modelo
    best_model, final_val_mae_internal, final_best_epoch = retrain_best_model(study.best_trial)

    # Predicciones finales
    val_preds = predict(best_model, X_val_tensor)
    final_mae = mean_absolute_error(y_val, val_preds)
    final_r2 = r2_score(y_val, val_preds)

    print("\n--- RESULTADOS FINALES DEL MODELO GANADOR ---")
    print(f"MAE final en validación: ${final_mae:.4f}")
    print(f"R2 final en validación: {final_r2:.4f}")
    print(f"Mejor época del reentrenamiento: {final_best_epoch}")

    # Guardado de artefactos
    os.makedirs("artifacts", exist_ok=True)

    model_path = os.path.join("artifacts", "best_insurance_model.pth")
    scaler_path = os.path.join("artifacts", "insurance_scaler_state.npz")

    torch.save(best_model.state_dict(), model_path)

    np.savez(
        scaler_path,
        mean_=scaler.mean_,
        scale_=scaler.scale_,
        var_=scaler.var_
    )

    # Tamaño real del modelo entrenado
    buffer = io.BytesIO()
    torch.save(best_model.state_dict(), buffer)
    size_mb = len(buffer.getvalue()) / (1024 * 1024)

    print(f"\nParámetros entrenables del modelo: {count_parameters(best_model):,}")
    print(f"Tamaño del modelo entrenado: {size_mb:.6f} MB")
    print(f"Modelo guardado en: {model_path}")
    print(f"Scaler guardado en: {scaler_path}")