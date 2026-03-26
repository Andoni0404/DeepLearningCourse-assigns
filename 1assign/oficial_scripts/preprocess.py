import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_data(seed: int = 42):
    # Ruta al CSV
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "../data/insurance.csv") 

    df = pd.read_csv(csv_path)
    
    # 1. Convertimos smoker a dummy antes para poder operar con ella
    # 'smoker_yes' será 1 si fuma, 0 si no.
    df = pd.get_dummies(df, columns=['smoker'], drop_first=True)
    
    # --- INGENIERÍA DE VARIABLES (FEATURE ENGINEERING) ---
    # 2. Interacción Obesidad + Fumador (El factor que más dispara el precio)
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    df['obese_smoker'] = df['is_obese'] * df['smoker_yes']
    
    # 3. Interacción Edad + Fumador (El riesgo aumenta con los años)
    df['age_smoker'] = df['age'] * df['smoker_yes']
    
    # 4. Edad al cuadrado (Captura la subida no lineal de costes médicos)
    df['age_2'] = df['age'] ** 2
    
    # 5. Aplicamos dummies al resto de categóricas (sex, region)
    df = pd.get_dummies(df, drop_first=True)
    # ------------------------------------------------------

    X = df.drop("charges", axis=1)
    y = df["charges"]
    stratify_col = df["smoker_yes"] if "smoker_yes" in df.columns else None

    # Splits (ahora X tiene las columnas nuevas)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify_col
    )
    stratify_temp = X_temp["smoker_yes"] if "smoker_yes" in X_temp.columns else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=seed, stratify=stratify_temp
    )

    # Scaling (El scaler ahora normalizará también 'obese_smoker', 'age_2', etc.)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Tensors
    tensors = {
        "X_train": torch.tensor(X_train_scaled, dtype=torch.float32),
        "y_train": torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1),
        "X_val": torch.tensor(X_val_scaled, dtype=torch.float32),
        "y_val": torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1),
        "X_test": torch.tensor(X_test_scaled, dtype=torch.float32),
        "y_test": torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1),
        "input_size": X_train_scaled.shape[1]
    }
    return tensors