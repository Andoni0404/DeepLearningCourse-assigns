import time
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import cross_validate
import os
import pandas as pd

print("=== INICIANDO FASE 1: BASELINE DE MACHINE LEARNING ===")

# ==========================================
# 0. CARGA Y PREPROCESAMIENTO DE DATOS
# ==========================================
print("Cargando y preparando el dataset 'insurance.csv'...")
# 1. Averiguamos la ruta absoluta de donde está ESTE archivo (.py)
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# 2. Construimos la ruta al CSV relativa a este archivo
ruta_csv = os.path.join(directorio_actual, '../data/insurance.csv')

# 3. Cargamos el DataFrame
df = pd.read_csv(ruta_csv)

# One-Hot Encoding para variables categóricas
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# División en Train y Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalado de datos (Vital para el SVR que está dentro del Stacking)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 1. DEFINICIÓN DEL MODELO AVANZADO (STACKING)
# ==========================================
print("\nConfigurando el modelo Stacking Regressor...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)

base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)),
    ('svr', SVR(kernel='rbf', C=10000, gamma='scale')) 
]

stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge()
)

# ==========================================
# 2. ENTRENAMIENTO Y EVALUACIÓN
# ==========================================
print("Entrenando y evaluando (Validación Cruzada 5-Fold)...")
start_time = time.time()

# Usamos cross_validate para sacar múltiples métricas a la vez
scoring_metrics = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2']
scores = cross_validate(stacking_model, X_train_scaled, y_train, scoring=scoring_metrics, cv=kf)

stacking_model.fit(X_train_scaled, y_train) 
end_time = time.time()
training_time = end_time - start_time

# Extraemos las métricas medias
mae_medio = -scores['test_neg_mean_absolute_error'].mean()
rmse_medio = -scores['test_neg_root_mean_squared_error'].mean()
r2_medio = scores['test_r2'].mean()

# ==========================================
# 3. MÉTRICAS CLAVE PARA COMPARAR CON DEEP LEARNING
# ==========================================
# Simulamos cuánto ocuparía el modelo al guardarlo en producción (MLOps)
model_size_bytes = sys.getsizeof(pickle.dumps(stacking_model))
model_size_mb = model_size_bytes / (1024 * 1024)

print(f"\n--- RESULTADOS FASE 1: BASELINE (STACKING) ---")
print(f"MAE Medio:  ${mae_medio:.2f} (Error absoluto típico)")
print(f"RMSE Medio: ${rmse_medio:.2f} (Penaliza errores graves)")
print(f"R² Medio:   {r2_medio:.4f} (Varianza explicada, max 1.0)")
print(f"Tiempo:     {training_time:.2f} segundos")
print(f"Tamaño:     {model_size_mb:.4f} MB")

# ==========================================
# 4. INTERPRETABILIDAD (FEATURE IMPORTANCE)
# ==========================================
print("\nGenerando gráfica de importancia de variables...")
rf_entrenado = stacking_model.named_estimators_['rf']
importances = rf_entrenado.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Dibujamos la gráfica
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Importancia de las Variables en el Coste del Seguro (Random Forest Base)')
plt.xlabel('Importancia Relativa')
plt.ylabel('Variables')
plt.tight_layout()
plt.show()