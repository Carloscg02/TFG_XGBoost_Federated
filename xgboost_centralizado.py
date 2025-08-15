import pandas as pd
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import rcParams
import matplotlib.ticker as ticker
from matplotlib.animation import PillowWriter


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict, KFold
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from sklearn.feature_selection import RFE
import json
import os

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


def load_preprocesado(filename, verbose=True):
    """
    Carga los datos y usa los splits predefinidos por site (train, val, test)
    para replicar el entorno federado en el modelo centralizado.
    """

    # 1. Leer dataset y limpiar columnas
    df = pd.read_excel(filename)
    df.columns = df.columns.str.replace(" ", "_").str.replace("[", "").str.replace("]", "")
    

    feature_cols = feature_selection(10)
    X = df[feature_cols]
    
    output_col = "thruCell_kbps"
    y = df[[output_col]]
    
    
    # 2. Cargar splits predefinidos por site
    split_dir = "site_splits"
    train_idx, val_idx, test_idx = [], [], []

    for site_id in range(1, 34):
        split_file = os.path.join(split_dir, f"site_{site_id}_split.json")
        with open(split_file, "r") as f:
            split = json.load(f)
        train_idx.extend(split["train"])
        val_idx.extend(split["val"])
        test_idx.extend(split["test"])

    # 3. Separar los conjuntos
    X_train_orig = X.loc[train_idx].reset_index(drop=True)
    X_val_orig = X.loc[val_idx].reset_index(drop=True)
    X_test_orig = X.loc[test_idx].reset_index(drop=True)

    y_train_orig = y.loc[train_idx].reset_index(drop=True)
    y_val_orig = y.loc[val_idx].reset_index(drop=True)
    y_test_orig = y.loc[test_idx].reset_index(drop=True)

    # 4. Escalado
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train_orig)
    X_val = scaler_X.transform(X_val_orig)
    X_test = scaler_X.transform(X_test_orig)

    y_train = scaler_y.fit_transform(y_train_orig)
    y_val = scaler_y.transform(y_val_orig)
    y_test = scaler_y.transform(y_test_orig)

    # 5. Reconstruir como DataFrames
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_val = pd.DataFrame(X_val, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    y_train = pd.DataFrame(y_train, columns=[output_col])
    y_val = pd.DataFrame(y_val, columns=[output_col])
    y_test = pd.DataFrame(y_test, columns=[output_col])


    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_train_orig, X_val_orig, X_test_orig,
        y_train_orig.values, y_val_orig.values, y_test_orig.values,
        scaler_X, scaler_y
    )

def hyperparameters_grid_search(x_t,y_t,verbose=True):
    #Definimos el modelo base
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.01, 0.1],
        'reg_alpha': [0.01, 0.1, 50, 100],
        'reg_lambda': [0.01, 0.1, 20, 50 , 100],
        'gamma': [0.01, 0.1 ,20, 50,  100]
    }

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1 if verbose else 0
    )
    
    grid_search.fit(x_t, y_t["thruCell_kbps"])

    if(verbose):
        print("\n✅ Mejores hiperparámetros encontrados:")
        print(grid_search.best_params_)
    return grid_search.best_estimator_

def mae_rmse(y_test, y_pred, verbose=True):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    if verbose:
        print("\n🎯 Rendimiento del modelo:")
        print(f"MAE [kbps]: {mae:.3f}")
        print(f"RMSE [kbps]: {rmse:.3f}")
    return mae, rmse


def mape_median(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    ape = np.abs((y_pred - y_true) / y_true) * 100
    return np.mean(ape)


def mane_cell(y_true, y_pred, bandwidth_mhz):
    """
    Calcula MANE [%] ajustado a BW variable por muestra.
    - y_true, y_pred: vectores en kbps
    - bandwidth_mhz: vector con BW por muestra (en MHz)
    
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Mapeo del ancho de banda a PRBs
    bw_to_prbs = {
        1.4: 6,
        3: 15,
        5: 25,
        10: 50,
        15: 75,
        20: 100
    }

    # Convertir BW a PRBs
    n_prbs = np.array([bw_to_prbs.get(bw, 0) for bw in bandwidth_mhz])

    # Calcular TH_max en kbps (1 Mbps = 1000 kbps)
    th_max = n_prbs * 1000  # en kbps

    # Evitar división por cero
    th_max = np.where(th_max == 0, 1e-6, th_max)

    error_normalizado = np.abs((y_pred - y_true) / th_max)
    return np.mean(error_normalizado * 100)

def r2_trimmed_mape_mane(y_true, y_pred, bandwidth_mhz, trim_percent=0.1, verbose=True):
    from sklearn.metrics import r2_score
    import numpy as np

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    bandwidth_mhz = np.array(bandwidth_mhz)

    # === R² ===
    r2 = r2_score(y_true, y_pred)

    # === MAPE y Trimmed MAPE ===
    ape = np.abs((y_pred - y_true) / y_true) * 100
    ape = ape[np.isfinite(ape)]  # quitar nan o inf si y_true == 0

    mape_original = np.mean(ape)

    n_trim = max(1, int(trim_percent * len(ape)))
    idx_trim = np.argsort(ape)[-n_trim:]
    mask = np.ones(len(ape), dtype=bool)
    mask[idx_trim] = False
    trimmed_mape = np.mean(ape[mask])

    # === MANE y Trimmed MANE ===
    bw_to_prbs = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100}
    n_prbs = np.array([bw_to_prbs.get(bw, 0) for bw in bandwidth_mhz])
    th_max = np.where(n_prbs == 0, 1e-6, n_prbs * 1000)

    error_normalizado = np.abs((y_pred - y_true) / th_max) * 100
    error_normalizado = error_normalizado[np.isfinite(error_normalizado)]  # por seguridad

    mane = np.mean(error_normalizado)

    n_trim_mane = max(1, int(trim_percent * len(error_normalizado)))
    idx_trim_mane = np.argsort(error_normalizado)[-n_trim_mane:]
    mask_mane = np.ones(len(error_normalizado), dtype=bool)
    mask_mane[idx_trim_mane] = False
    trimmed_mane = np.mean(error_normalizado[mask_mane])

    return r2, trimmed_mape, mane, trimmed_mane



def rfe_mane_analysis_cv(X_train, y_train, X_val, y_val, xgb_model, bandwidth_val, scaler_y, cv=1, verbose=False):


    """
    Ejecuta RFE (Recursive Feature Elimination) con evaluación por número de features,
    incluyendo MAE, RMSE, MAPE y MANE. Guarda además las features seleccionadas.

    Devuelve:
    ---------
    DataFrame con resultados por número de features, incluyendo las features seleccionadas.
    """
    resultados = []
    n_total_features = X_train.shape[1]
    feature_names = X_train.columns.tolist()

    for n in tqdm(range(1, n_total_features + 1)):
        rfe = RFE(estimator=xgb_model, n_features_to_select=n, step=1)
        rfe.fit(X_train, y_train)

        selected_mask = rfe.support_
        selected_feature_names = [feature for feature, selected in zip(feature_names, selected_mask) if selected]

        X_train_sel = rfe.transform(X_train)
        X_val_sel = rfe.transform(X_val)

        if cv > 1:
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            maes, rmses, mapes, manes = [], [], [], []

            for train_idx, val_idx in kf.split(X_train_sel):
                X_fold_train = X_train_sel[train_idx]
                y_fold_train = y_train.iloc[train_idx].values
                X_fold_val = X_train_sel[val_idx]
                y_fold_val = y_train.iloc[val_idx].values

                model = xgb_model
                model.fit(X_fold_train, y_fold_train)

                y_pred = model.predict(X_fold_val)
                y_true = scaler_y.inverse_transform(y_fold_val.reshape(-1, 1)).ravel()
                y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

                y_safe = np.where(y_true == 0, 1e-6, y_true)
                mape = np.mean(np.abs((y_true - y_pred) / y_safe)) * 100

                bw = bandwidth_val.iloc[val_idx].values
                bw_rounded = np.round(bw, 1)
                bw_to_prbs = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100}
                prbs = np.array([bw_to_prbs.get(b, 0) for b in bw_rounded])
                th_max = np.where(prbs == 0, 1e-6, prbs * 1000.0)
                mane = np.mean(np.abs((y_true - y_pred) / th_max)) * 100

                maes.append(mean_absolute_error(y_true, y_pred))
                rmses.append(np.sqrt(mean_squared_error(y_true, y_pred)))
                mapes.append(mape)
                manes.append(mane)

            resultados.append({
                "n_features": n,
                "MAE": np.mean(maes),
                "RMSE": np.mean(rmses),
                "MAPE": np.mean(mapes),
                "MANE": np.mean(manes),
                "selected_features": selected_feature_names
            })

        else:
            model = xgb_model
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_val_sel)
            y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
            y_true = y_val

            y_safe = np.where(y_true == 0, 1e-6, y_true)
            mape = np.mean(np.abs((y_true - y_pred) / y_safe)) * 100

            bw = bandwidth_val.values
            bw_rounded = np.round(bw, 1)
            bw_to_prbs = {1.4: 6, 3: 15, 5: 25, 10: 50, 15: 75, 20: 100}
            prbs = np.array([bw_to_prbs.get(b, 0) for b in bw_rounded])
            th_max = np.where(prbs == 0, 1e-6, prbs * 1000.0)
            mane = np.mean(np.abs((y_true - y_pred) / th_max)) * 100

            resultados.append({
                "n_features": n,
                "MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAPE": mape,
                "MANE": mane,
                "selected_features": selected_feature_names
            })

        if verbose:
            print(f"{n} features → MAE: {resultados[-1]['MAE']:.2f}, "
                  f"RMSE: {resultados[-1]['RMSE']:.2f}, "
                  f"MAPE: {resultados[-1]['MAPE']:.2f}%, "
                  f"MANE: {resultados[-1]['MANE']:.2f}%, "
                  f"Features: {selected_feature_names}")

    return pd.DataFrame(resultados)

def scatter_pred_vs_real(y_true, y_pred, title="Estimaciones vs Valores Reales modelo centralizado sin RFE"):
    # Crear el scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, color="teal", edgecolors="black")
    
    # Añadir línea ideal (predicciones perfectas)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Línea Ideal (y = x)")
    
    
    # Añadir valor manual de R²
    plt.text(0.05, 0.95, "$R^2 = 0.899$", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    
    # Añadir etiquetas y título
    plt.title(title)
    plt.xlabel("Valores reales de throughput (kbps)")
    plt.ylabel("Estimaciones de throughput  (kbps)")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)

def scatter_pred_vs_real_log(y_true, y_pred, title="Estimaciones vs Valores Reales (logarítmico) modelo centralizado sin RFE"):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7, color="darkorange", edgecolors="black")
    
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=2, label="Línea Ideal (y = x)")
    
    # Añadir valor manual de R²
    plt.text(0.05, 0.95, "$R^2 = 0.899$", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
    
    plt.xscale("log")
    plt.yscale("log")
    
    plt.title(title)
    plt.xlabel("Valores reales de throughput (escala log, kbps)")
    plt.ylabel("Estimaciones de throughput (escala log, kbps)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.show(block=False)

def graficar_mane_vs_features(ruta_xlsx=None, df=None):
    """
    Grafica MANE [%] vs número de features a partir de un DataFrame o Excel.
    Añade anotaciones sobre modelo simplificado y completo, y nombres de las variables seleccionadas.
    """
    if ruta_xlsx:
        df = pd.read_excel(ruta_xlsx)
    elif df is None:
        raise ValueError("Debes proporcionar una ruta a Excel o un DataFrame.")

    # Si los valores están con coma decimal, convertirlos
    df["MANE"] = df["MANE"].astype(str).str.replace(',', '.').astype(float)
    df["n_features"] = df["n_features"].astype(int)

    # Lista ordenada de los nombres de features añadidos por RFE
    nombres_features = ['cell_BW_MHz', 'PRButil_rat', 'avg_CQI', 'median_CQI', 'avg_actUE', 'p5_CQI', 'voip_ue_rat', 'video_ue_rat', 'ftp_ue_rat', 'http_ue_rat']

    # Gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(df["n_features"], df["MANE"], marker='o', linestyle='-')
    #plt.title("Curva de MANE [%] en función del número de features", fontsize=14, fontweight='bold')
    plt.xlabel("Número de features seleccionadas", fontsize=12, fontweight='bold')
    plt.ylabel("MANE [%]", fontsize=12, fontweight='bold')
    plt.grid(True)

    # Añadir etiquetas de nombre de feature en el eje X
    xticks = df["n_features"].tolist()
    feature_labels = nombres_features[:len(xticks)]
    plt.xticks(ticks=xticks, labels=feature_labels, rotation=45, ha='right', fontsize=10)

    # Añadir anotaciones sobre modelo simplificado y completo
    for n, label, offset_x, offset_y in [
        (5, "Modelo simplificado", 0.7, 0.5),
        (10, "Modelo completo", -0.7, 0.5)
    ]:
        y = df.loc[df["n_features"] == n, "MANE"].values
        if len(y) > 0:
            plt.annotate(label,
                         xy=(n, y[0]),
                         xytext=(n + offset_x, y[0] + offset_y),
                         arrowprops=dict(arrowstyle="->", color='black'),
                         fontsize=14,
                         fontweight='bold',
                         ha='center')



    plt.tight_layout()
    plt.show(block=False)


def main():
   
    ruta_archivo = "dataset_con_site_id.xlsx"

    #cargar datos
    X_train, X_val, X_test, \
    y_train, y_val, y_test, \
    X_train_orig, X_val_orig, X_test_orig, \
    y_train_orig, y_val_orig, y_test_orig, \
    scaler_X, scaler_y = load_preprocesado(ruta_archivo)
    
    # Modelo XGBoost final con los hiperparámetros encontrados
    xgb_best = hyperparameters_grid_search(X_train, y_train, verbose=True) #Modelo entrenado, una vez ejecutado, meto los datos manualmente para ahorra computación en las siguientes ejecuciones

    
    #fijar hiperparámetros encontrados por grid search manualmente 
    # xgb_best = XGBRegressor(
    #     objective='reg:squarederror',
    #     n_estimators=100,
    #     max_depth=5,
    #     learning_rate=0.1,
    #     reg_alpha=0.01,
    #     reg_lambda=0.1,
    #     gamma=0.01,
    #     random_state=42,
    #     early_stopping_rounds=10
    # )
    
    
    xgb_best.fit(
        X_train,
        y_train.values.ravel(),
        eval_set=[(X_train, y_train.values.ravel()), (X_val, y_val.values.ravel())],
        eval_metric="rmse",
        verbose=False
    )

    results = xgb_best.evals_result()
    
    
    train_rmse = results['validation_0']['rmse']
    val_rmse = results['validation_1']['rmse']
    rounds = range(1, len(train_rmse) + 1)
    
    plt.plot(rounds, train_rmse, label='Entrenamiento', marker='o')
    plt.plot(rounds, val_rmse, label='Validación', marker='s')
    plt.xlabel('Número de árboles (boosting rounds)')
    plt.ylabel('RMSE')
    plt.title('Curva de convergencia del modelo centralizado')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    
    
    #=== 2. Guarda el modelo entrenado en formato Booster (.json) ===
    #xgb_best.get_booster().save_model("modelo_centralizado.json")
    xgb_best.get_booster().save_model("modelo_centralizado.json")

    
    
    # Predicciones
    y_pred = xgb_best.predict(X_test)


    y_test_des = scaler_y.inverse_transform(y_test)
    y_pred_des = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    

    # Métricas
    mae, rmse = mae_rmse(y_test_des, y_pred_des, verbose=False)
    mae_scaled,rmse_scaled = mae_rmse(y_test, y_pred, verbose=False)
    mape = mape_median(y_test_des, y_pred_des)
    #mane = mane_cell(y_test_des, y_pred_des, X_test_orig['cell_BW_MHz'].values)
    
    
    
    r2, trimmed_mape, mane, trimmed_mane = r2_trimmed_mape_mane(
    y_true=y_test_des, 
    y_pred=y_pred_des, 
    bandwidth_mhz=X_test_orig['cell_BW_MHz'].values, 
    trim_percent=0.10, 
    verbose=False
    )
    
    print("\n----Predicción TH de celda modelo centralizado---------")
    print("-------------------------------------------")
    print(f"{'MAE [kbps]':}: {mae:.3f}")
    print(f"{'RMSE [kbps]':}: {rmse:.3f}")
    print(f"{'MAE escalada':}: {mae_scaled:.3f}")
    print(f"{'RMSE escalada ':}: {rmse_scaled:.3f}")    
    print(f"{'R²':}: {r2:.3f}")
    print(f"{'mAPE [%]':}: {mape:.2f}")
    print(f"{'MANE [%]'}: {mane:.2f}")
    print(f"{'Trimmed MAPE (10%) [%]':}: {trimmed_mape:.2f}%")
    print(f"{'Trimmed MANE (10%) [%]':}: {trimmed_mane:.2f}%")
    print("-------------------------------------------")  


    

    #bandwidth_val = X_val_orig["cell_BW_MHz"].astype(float).values
    y_val_des = scaler_y.inverse_transform(y_val.values.reshape(-1, 1)).ravel()
    #y_pred = model.predict(X_val_sel)
    
    
    
    
    # y_val_des = scaler_y.inverse_transform(y_val.values.reshape(-1, 1)).ravel()

    # # Reindexar para que coincidan con train_index/val_index usados en CV
    # X_train = X_train.reset_index(drop=True)

    # Modelo para RFE (sin early stopping para ser compatible con implementación de rfe)
    xgb_best_rfe = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        reg_alpha=0.01,
        reg_lambda=0.1,
        gamma=0.01,
        random_state=42
    )
    bandwidth_val = X_train_orig["cell_BW_MHz"].reset_index(drop=True)
    
    rfe_resultados = rfe_mane_analysis_cv(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val_des,
        xgb_model=xgb_best_rfe,
        bandwidth_val=bandwidth_val,
        scaler_y=scaler_y,
        cv=5,
        verbose=True
    )
    


    # #print(rfe_resultados)

    rfe_resultados.to_excel("resultados_rfe_mane.xlsx", index=False)
    graficar_mane_vs_features(ruta_xlsx="resultados_rfe_mane.xlsx")

    
    
  
    # Generar scatter plot de predicciones vs reales
    scatter_pred_vs_real(y_test_des, y_pred_des)
    scatter_pred_vs_real_log(y_test_des, y_pred_des)
    input("Pulsar enter para cerrar plots...")
    

    # Número de muestras de throughput > 30 Mbps en train y val
    # n_train_high = np.sum(y_train_orig < 30000)
    # n_val_high = np.sum(y_val_orig < 30000)
    # n_total = len(y_train_orig) + len(y_val_orig)
    
    # # Proporción total
    # proporcion = (n_train_high + n_val_high) / n_total
    
    # print(f"Muestras con throughput >30 Mbps en train+val: {n_train_high + n_val_high}")
    # print(f"Proporción sobre train+val: {proporcion:.2%}")



if __name__ == "__main__":
    main()
    #graficas_RFE()