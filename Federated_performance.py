
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os


def mae_rmse_scaled(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print(f"MAE escalada: {mae:.3f}")
    # print(f"RMSE escalada: {rmse:.3f}")
    return mae, rmse

# === Función de métricas con trimmed ===
def all_metrics_show(y_true, y_pred, mae_ESC, rmse_ESC, bandwidth_mhz, trim_percent=0.1, verbose=True):
    r2 = r2_score(y_true, y_pred)
    ape = np.abs((y_pred - y_true) / y_true) * 100
    mape_original = np.mean(ape)

    n_trim = max(1, int(trim_percent * len(ape)))
    trimmed_ape = np.sort(ape.flatten())[:-n_trim]
    trimmed_mape = np.mean(trimmed_ape)

    bw_to_prbs = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100}
    n_prbs = np.array([bw_to_prbs.get(bw, 0) for bw in bandwidth_mhz])
    th_max = np.where(n_prbs == 0, 1e-6, n_prbs * 1000)
    error_norm = np.abs((y_pred - y_true) / th_max) * 100
    trimmed_mane = np.mean(np.sort(error_norm)[:-n_trim])

    mane = np.mean(error_norm)

    if verbose:
        print("Evaluación centralizada del modelo federado:")
        print(f"MAE [kbps]: {mean_absolute_error(y_true, y_pred):.2f}")
        print(f"RMSE [kbps]: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
        print(f"MAE escalada: {mae_ESC:.3f}")
        print(f"RMSE escalada: {rmse_ESC:.3f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE [%]: {mape_original:.2f}")
        print(f"MANE [%]: {mane:.2f}")
        print(f"Trimmed MAPE [%]: {trimmed_mape:.2f}")
        print(f"Trimmed MANE [%]: {trimmed_mane:.2f}")
    
    return r2, mape_original, trimmed_mape, mane, trimmed_mane

#scatter plot
def scatter_pred_vs_real(y_true, y_pred, title="Estimaciones vs Valores Reales modelo Federado"):
    # Crear el scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, color="teal", edgecolors="black")
    
    # Añadir línea ideal (predicciones perfectas)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Línea Ideal (y = x)")
    
    
    # Añadir valor manual de R²
    plt.text(0.05, 0.95, "$R^2 = 0.8619$", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    
    # Añadir etiquetas y título
    plt.title(title)
    plt.xlabel("Valores reales de throughput (kbps)")
    plt.ylabel("Estimaciones de throughput  (kbps)")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)


def scatter_pred_vs_real_log(y_true, y_pred, title="Estimaciones vs Valores Reales (escala logarítmica) Federado"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, color="darkorange", edgecolors="black")
    
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Línea Ideal (y = x)")
    
    # Añadir valor manual de R²
    plt.text(0.05, 0.95, "$R^2 = 0.8619$", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    
    plt.xscale("log")
    plt.yscale("log")
    
    plt.title(title)
    plt.xlabel("Valores reales de throughput (escala log, kbps)")
    plt.ylabel("Estimaciones de throughput (escala log, kbps)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show(block=False)


def convergencia():
    # === Cargar el CSV con las métricas ===
    df = pd.read_csv("convergence_federated_rmse.csv")

    # === Crear la figura ===
    plt.figure(figsize=(8, 5))
    
    # === Graficar RMSE de entrenamiento (si está disponible)
    if "train_rmse" in df.columns:
        plt.plot(df["round"], df["train_rmse"], marker='o', linestyle='--', label="Entrenamiento")
    

    # === Graficar RMSE de validación
    if "val_rmse" in df.columns:
        plt.plot(df["round"], df["val_rmse"], marker='s', linestyle='-', label="Validación")

        
    # === Graficar RMSE de entrenamiento (si está disponible)
    # if "test_rmse" in df.columns:
    #     plt.plot(df["round"], df["test_rmse"], marker='x', linestyle='--', color='green', label="RMSE (test)")    

    # === Etiquetas y diseño
    plt.xlabel("Ronda global")
    plt.ylabel("RMSE")
    #plt.title("Curva de convergencia del modelo federado")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # === Guardar y mostrar la figura
    #plt.savefig("convergencia_rmse_federado.png")
    plt.show(block=False)


# === Cargar dataset ===
df = pd.read_excel("dataset_con_site_id.xlsx")
#df = pd.read_excel("C:/Users/User/xgboost-FL/dataset_filtrado_renumerado_MAPE.xlsx")
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("[", "", regex=False)
df.columns = df.columns.str.replace("]", "", regex=False)


def feature_selection(n):
    """Devuelve una lista con las n primeras features seleccionadas."""
    if not (1 <= n <= 10):
        raise ValueError("n debe estar entre 1 y 10")

    all_features = [
        'cell_BW_MHz', 'PRButil_rat', 'avg_CQI', 'median_CQI',
        'p5_CQI', 'avg_actUE', 'voip_ue_rat', 'video_ue_rat',
        'ftp_ue_rat', 'http_ue_rat'
    ]

    return all_features[:n]

feature_cols = feature_selection(10)



# === Definir features y target ===
output_col = "thruCell_kbps"



X = df[feature_cols]
y = df[output_col]
bw = X["cell_BW_MHz"].astype(float).values


# === Cargar splits desde archivos JSON ===
split_dir = "C:/Users/User/xgboost-FL/xgboostFL/site_splits"
train_idx, test_idx = [], []

for site_id in range(1, 34):
    split_file = os.path.join(split_dir, f"site_{site_id}_split.json")
    with open(split_file, "r") as f:
        split = json.load(f)
    
    train_idx.extend(split["train"] + split["val"])  # Entrenamiento = train + val
    test_idx.extend(split["test"])                   


# === Dividir usando los índices predefinidos ===
X_train = X.loc[train_idx].reset_index(drop=True)
y_train = y.loc[train_idx].reset_index(drop=True)
bw_train = bw[train_idx]

X_test = X.loc[test_idx].reset_index(drop=True)
y_test = y.loc[test_idx].reset_index(drop=True)
bw_test = bw[test_idx]

# === Escalado con stats del conjunto de entrenamiento ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# === DMatrix de test ===
dtest = xgb.DMatrix(X_test_scaled, label=y_test_scaled, feature_names=X.columns.tolist())

# === Cargar y aplicar modelo federado ===
booster = xgb.Booster()
#booster.load_model("final_model.json")

booster.load_model("modelos_federados/modelo_round_27.json")


y_pred_scaled = booster.predict(dtest)

# === Desescalado de predicciones ===
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = y_test.values

# === Evaluación ===
mae_escalada, rmse_escalada=mae_rmse_scaled(y_test_scaled, y_pred_scaled)
all_metrics_show(y_true, y_pred, mae_escalada, rmse_escalada, bw_test, trim_percent=0.1, verbose=True)


# #visualización
scatter_pred_vs_real(y_true, y_pred, title="Estimaciones vs Valores Reales modelo Federado")
scatter_pred_vs_real_log(y_true, y_pred)
convergencia()
input("Pulsar enter para cerrar plots...")