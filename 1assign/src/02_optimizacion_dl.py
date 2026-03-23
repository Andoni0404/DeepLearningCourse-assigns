import os
import time
import io
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

print("=== INICIANDO FASE 2: OPTIMIZACIÓN DE DEEP LEARNING CON OPTUNA ===")

# ==========================================
# 0. CARGA Y PREPROCESAMIENTO DE DATOS
# ==========================================
directorio_actual = os.path.dirname(os.path.abspath(__file__))
ruta_csv = os.path.join(directorio_actual, '../data/insurance.csv')
df = pd.read_csv(ruta_csv)

df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop('charges', axis=1).values
y = df_encoded['charges'].values

# División Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado (Fundamental para Redes Neuronales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir a Tensores de PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ==========================================
# 1. FUNCIÓN DE CREACIÓN DEL MODELO DINÁMICO
# ==========================================
def build_model(trial, input_dim):
    layers = []
    # Optuna decide cuántas capas ocultas (1 a 3)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    # Optuna decide si aplicamos Dropout para evitar overfitting
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
    
    in_features = input_dim
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 16, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        in_features = out_features
        
    layers.append(nn.Linear(in_features, 1)) # Capa final (predicción)
    return nn.Sequential(*layers)

# ==========================================
# 2. FUNCIÓN OBJETIVO PARA OPTUNA
# ==========================================
def objective(trial):
    # Sugerir hiperparámetros
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Crear modelo y Dataloader
    model = build_model(trial, input_dim=X_train_tensor.shape[1])
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.L1Loss() # Usamos MAE como función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 100
    
    # Bucle de entrenamiento
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Evaluación en Test
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        val_mae = mean_absolute_error(y_test_tensor.numpy(), preds.numpy())
        
    return val_mae

# ==========================================
# 3. EJECUCIÓN DEL ESTUDIO
# ==========================================
if __name__ == "__main__":
    nombre_estudio = "dl_insurance_opt"
    base_datos = f"sqlite:///{os.path.join(directorio_actual, 'optuna_dl.db')}"
    
    study = optuna.create_study(
        study_name=nombre_estudio,
        storage=base_datos,
        load_if_exists=True,
        direction="minimize"
    )
    
    print("Iniciando búsqueda de hiperparámetros. ¡Ejecuta 50 trials!")
    print(f"Para ver el dashboard, abre otra terminal y lanza: optuna-dashboard {base_datos}")
    
    start_time = time.time()
    study.optimize(objective, n_trials=50) # Prueba 50 arquitecturas diferentes
    total_time = time.time() - start_time

    # ==========================================
    # 4. RECONSTRUIR Y EVALUAR EL MEJOR MODELO
    # ==========================================
    print("\nReconstruyendo el mejor modelo para extraer métricas finales...")
    best_trial = study.best_trial
    
    # Entrenamos el mejor modelo rápido para sacar sus métricas
    best_model = build_model(best_trial, input_dim=X_train_tensor.shape[1])
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=best_trial.params['batch_size'], shuffle=True)
    optimizer = optim.Adam(best_model.parameters(), lr=best_trial.params['lr'])
    criterion = nn.L1Loss()
    
    for epoch in range(150): # Le damos 150 épocas al modelo campeón
        best_model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(best_model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    # Medir rendimiento en Test
    best_model.eval()
    with torch.no_grad():
        preds = best_model(X_test_tensor).numpy()
        y_true = y_test_tensor.numpy()
        
        final_mae = mean_absolute_error(y_true, preds)
        final_rmse = np.sqrt(mean_squared_error(y_true, preds))
        final_r2 = r2_score(y_true, preds)
        
    # Calcular tamaño en memoria (Guardándolo en un buffer de bytes)
    buffer = io.BytesIO()
    torch.save(best_model.state_dict(), buffer)
    size_mb = len(buffer.getvalue()) / (1024 * 1024)

    print(f"\n--- RESULTADOS FASE 2: DEEP LEARNING OPTIMIZADO ---")
    print(f"Mejores Hiperparámetros: {best_trial.params}")
    print(f"MAE Medio:  ${final_mae:.2f}")
    print(f"RMSE Medio: ${final_rmse:.2f}")
    print(f"R² Medio:   {final_r2:.4f}")
    print(f"Tiempo total de búsqueda (50 redes): {total_time:.2f} segundos")
    print(f"Tamaño del modelo (pesos): {size_mb:.6f} MB")