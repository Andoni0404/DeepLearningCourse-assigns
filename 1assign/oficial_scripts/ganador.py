import torch
import torch.nn as nn
import optuna

# Importamos tus módulos
from preprocess import get_data
from model import build_model, build_optimizer

# 1. Cargamos el ganador de la base de datos
estudio = optuna.load_study(
    study_name="insurance_lightweight_v2_changeNumeroCapas", 
    storage="sqlite:///optuna_insurance.db"
)
mejor_trial = estudio.best_trial

print("🏆 HIPERPARÁMETROS DEL MODELO GANADOR 🏆")
print(f"Mejor Error en Validación (MAE): ${mejor_trial.value:.4f}")
for clave, valor in mejor_trial.params.items():
    print(f"  - {clave}: {valor}")

# ========================================================
# 2. LA PRUEBA FINAL: EVALUACIÓN EN EL CONJUNTO DE TEST
# ========================================================
print("\n Entrenando el modelo final para la prueba de Test...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = get_data(seed=42)

# Extraemos los datos (PyTorch Dataloader para el train)
batch_size = mejor_trial.params.get("batch_size", 32)
train_dataset = torch.utils.data.TensorDataset(data["X_train"], data["y_train"])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Construimos el modelo y el optimizador con la "receta" ganadora
input_size = data["X_train"].shape[1]
modelo_final = build_model(mejor_trial, input_size, device)
optimizador = build_optimizer(mejor_trial, modelo_final)

# Usaremos la función de pérdida EXACTA que eligió Optuna
nombre_loss = mejor_trial.params.get("loss_fn", "Huber")
if nombre_loss == "MSE":
    criterion = nn.MSELoss()
elif nombre_loss == "L1" or nombre_loss == "MAE":
    criterion = nn.L1Loss()
else:
    criterion = nn.HuberLoss()

# Entrenamos rápido el modelo final (300 épocas)
max_epochs = 300
modelo_final.train()
for epoch in range(max_epochs):
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizador.zero_grad()
        preds = modelo_final(xb)
        loss = criterion(preds, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo_final.parameters(), max_norm=5.0)
        optimizador.step()

# --- EL EXAMEN FINAL (TEST) ---
modelo_final.eval()
with torch.no_grad():
    X_test_device = data["X_test"].to(device)
    y_test_device = data["y_test"].to(device)
    
    test_preds = modelo_final(X_test_device)
    test_mae = nn.L1Loss()(test_preds, y_test_device).item()
    test_mse = nn.MSELoss()(test_preds, y_test_device).item()
    
    # R2 Score en Test
    y_bar = torch.mean(y_test_device)
    ss_tot = torch.sum((y_test_device - y_bar) ** 2)
    ss_res = torch.sum((y_test_device - test_preds) ** 2)
    test_r2 = (1 - ss_res / ss_tot).item()

print("\n📊 RESULTADOS FINALES EN EL CONJUNTO DE TEST (Mundo Real) 📊")
print(f"  - MAE (Error Absoluto Medio): ${test_mae:.2f}")
print(f"  - R² Score: {test_r2:.4f}")