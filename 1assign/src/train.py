import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# 1. IMPORTAMOS EL MODELO DESDE EL OTRO ARCHIVO
# Importamos el modelo ganador (UltimateDeepModel). 
# Nota: ShallowInsuranceModel también está disponible en model.py 
# para pruebas de baseline y comparativas de arquitectura.
from model import UltimateInsuranceModel

# 2. REPRODUCIBILIDAD
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 3. DEVICE MANAGEMENT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# 4. CARGA Y PREPARACIÓN DE DATOS
df = pd.read_csv('../data/insurance.csv') 
df = pd.get_dummies(df, drop_first=True)

X = df.drop('charges', axis=1).values
y = df['charges'].values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. CONFIGURACIÓN DEL MODELO
model_ultimate = UltimateInsuranceModel(X_train_t.shape[1]).to(device)
optimizer = torch.optim.Adam(model_ultimate.parameters(), lr=0.0005) 
loss_fn = nn.MSELoss()
calculadora_mae = nn.L1Loss()

# 6. ENTRENAMIENTO
epochs = 800
train_losses = []
val_losses = []

print("Entrenando Ultimate Deep NN...")
for t in range(epochs):
    model_ultimate.train()
    batch_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model_ultimate(X_batch)
        loss = loss_fn(pred, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())
        
    train_losses.append(sum(batch_losses)/len(batch_losses))
    
    model_ultimate.eval()
    with torch.no_grad():
        val_pred = model_ultimate(X_val_t.to(device))
        v_loss = loss_fn(val_pred, y_val_t.to(device)).item()
        val_losses.append(v_loss)

# 7. EVALUACIÓN Y GUARDADO
model_ultimate.eval()
with torch.no_grad():
    preds_finales = model_ultimate(X_val_t.to(device))
    mae_pytorch = calculadora_mae(preds_finales, y_val_t.to(device)).item()

print(f"\n MAE FINAL PyTorch: ${mae_pytorch:.2f}")

# Guardamos el archivo .pth en la carpeta de modelos
torch.save(model_ultimate.state_dict(), "modelo_seguros_final.pth")
print("✅ Modelo guardado con éxito.")