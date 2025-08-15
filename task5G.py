"""xgboost_comprehensive: A Flower / XGBoost app."""

# from logging import INFO

# import xgboost as xgb
# from flwr_datasets import FederatedDataset
# from flwr_datasets.partitioner import (
#     ExponentialPartitioner,
#     IidPartitioner,
#     LinearPartitioner,
#     SquarePartitioner,
# )

# from datasets import DatasetDict, concatenate_datasets
# from flwr.common import log

# CORRELATION_TO_PARTITIONER = {
#     "uniform": IidPartitioner,
#     "linear": LinearPartitioner,
#     "square": SquarePartitioner,
#     "exponential": ExponentialPartitioner,
# }

# fds = None  # Cache FederatedDataset


# def train_test_split(partition, test_fraction, seed):
#     """Split the data into train and validation set given split rate."""
#     train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
#     partition_train = train_test["train"]
#     partition_test = train_test["test"]

#     num_train = len(partition_train)
#     num_test = len(partition_test)

#     return partition_train, partition_test, num_train, num_test


# def transform_dataset_to_dmatrix(data):
#     """Transform dataset to DMatrix format for xgboost."""
#     x = data["inputs"]
#     y = data["label"]
#     new_data = xgb.DMatrix(x, label=y)
#     return new_data


# def instantiate_fds(partitioner_type, num_partitions):
#     """Initialize FederatedDataset."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = CORRELATION_TO_PARTITIONER[partitioner_type](
#             num_partitions=num_partitions
#         )
#         fds = FederatedDataset(
#             dataset="jxie/higgs",
#             partitioners={"train": partitioner},
#             preprocessor=resplit,
#         )
#     return fds


# def load_data(
#     partitioner_type,
#     partition_id,
#     num_partitions,
#     centralised_eval_client,
#     test_fraction,
#     seed,
# ):
#     """Load partition data."""
#     fds_ = instantiate_fds(partitioner_type, num_partitions)
#     partition = fds_.load_partition(partition_id)
#     partition.set_format("numpy")

#     if centralised_eval_client:
#         train_data = partition
#         num_train = train_data.shape[0]

#         # Use centralised test set for evaluation
#         valid_data = fds_.load_split("test")
#         valid_data.set_format("numpy")
#         num_val = valid_data.shape[0]
#     else:
#         # Train/test splitting
#         train_data, valid_data, num_train, num_val = train_test_split(
#             partition, test_fraction=test_fraction, seed=seed
#         )

#     # Reformat data to DMatrix for xgboost
#     log(INFO, "Reformatting data...")
#     train_dmatrix = transform_dataset_to_dmatrix(train_data)
#     valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

#     return train_dmatrix, valid_dmatrix, num_train, num_val


# def replace_keys(input_dict, match="-", target="_"):
#     """Recursively replace match string with target string in dictionary keys."""
#     new_dict = {}
#     for key, value in input_dict.items():
#         new_key = key.replace(match, target)
#         if isinstance(value, dict):
#             new_dict[new_key] = replace_keys(value, match, target)
#         else:
#             new_dict[new_key] = value
#     return new_dict


# def resplit(dataset: DatasetDict) -> DatasetDict:
#     """Increase the quantity of centralised test samples from 500K to 1M."""
#     return DatasetDict(
#         {
#             "train": dataset["train"].select(
#                 range(0, dataset["train"].num_rows - 500_000)
#             ),
#             "test": concatenate_datasets(
#                 [
#                     dataset["train"].select(
#                         range(
#                             dataset["train"].num_rows - 500_000,
#                             dataset["train"].num_rows,
#                         )
#                     ),
#                     dataset["test"],
#                 ]
#             ),
#         }
#     )
import os
import json
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flwr.common import log



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


# === 1. Cargar Dataset y Particionar por Site ID ===
def load_partition(site_id, seed=42, verbose=True, scale_data=True):
    df = pd.read_excel("dataset_con_site_id.xlsx")
    df.columns = df.columns.str.replace(" ", "_").str.replace("[", "", regex=False).str.replace("]", "", regex=False)

    output_cols = ["thruCell_kbps"]
    #feature_cols = FEATURES_SELECCIONADAS
    feature_cols = feature_selection(10)

    site_data = df[df["site_id"] == site_id]
    X = site_data[feature_cols]
    y = site_data[output_cols]
    

    index_dir = "site_splits"
    os.makedirs(index_dir, exist_ok=True)
    index_file = os.path.join(index_dir, f"site_{site_id}_split.json")
    
    temp_idx, test2_idx = train_test_split(site_data.index.tolist(), test_size=0.2, random_state=seed)

    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            split_indices = json.load(f)
        train_idx = split_indices["train"]
        val_idx = split_indices["val"]
        test_idx = split_indices["test"]
    else:
        temp_idx, test_idx = train_test_split(site_data.index.tolist(), test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(temp_idx, test_size=0.2, random_state=seed)
        with open(index_file, "w") as f:
            json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)


    X_80_orig = X.loc[temp_idx]
    y_80_orig = y.loc[temp_idx]
    X_train_orig = X.loc[train_idx]
    y_train_orig = y.loc[train_idx]
    X_val_orig = X.loc[val_idx]
    y_val_orig = y.loc[val_idx]
    X_test_orig = X.loc[test_idx]
    y_test_orig = y.loc[test_idx]
    
    if scale_data:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
    
        # Ajustar solo sobre el conjunto de entrenamiento
        scaler_X.fit(X_train_orig)
        scaler_y.fit(y_train_orig)
    
        # Transformar todos los subconjuntos con el mismo escalador
        X_train = scaler_X.transform(X_train_orig)
        X_val   = scaler_X.transform(X_val_orig)
        X_test  = scaler_X.transform(X_test_orig)
        X_80    = scaler_X.transform(X_80_orig)
    
        y_train = scaler_y.transform(y_train_orig)
        y_val   = scaler_y.transform(y_val_orig)
        y_test  = scaler_y.transform(y_test_orig)
        y_80    = scaler_y.transform(y_80_orig)
    
        # Reconstrucción como DataFrames
        X_train = pd.DataFrame(X_train, columns=feature_cols)
        X_val   = pd.DataFrame(X_val, columns=feature_cols)
        X_test  = pd.DataFrame(X_test, columns=feature_cols)
        X_80    = pd.DataFrame(X_80, columns=feature_cols)
    
        y_train = pd.DataFrame(y_train, columns=output_cols)
        y_val   = pd.DataFrame(y_val, columns=output_cols)
        y_test  = pd.DataFrame(y_test, columns=output_cols)
        y_80    = pd.DataFrame(y_80, columns=output_cols)
    
        # DMatrix
        ochenta_dmatrix = xgb.DMatrix(X_80, label=y_80.values.flatten())
        train_dmatrix   = xgb.DMatrix(X_train, label=y_train.values.flatten())
        valid_dmatrix   = xgb.DMatrix(X_val,   label=y_val.values.flatten())
        test_dmatrix    = xgb.DMatrix(X_test,  label=y_test.values.flatten())

        if verbose:
            print(f"✅ Datos escalados para Site {site_id} - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        return (
            ochenta_dmatrix, train_dmatrix, valid_dmatrix, X_80, y_80,
            X_train_orig, y_train_orig, X_val_orig, y_val_orig,
            test_dmatrix, X_test_orig, y_test_orig,
            scaler_X, scaler_y,
            X_train.shape[0], X_val.shape[0]
        )

    else:
        ochenta_dmatrix = xgb.DMatrix(X_80, label=y_80_orig)
        train_dmatrix = xgb.DMatrix(X_train_orig, label=y_train_orig)
        valid_dmatrix = xgb.DMatrix(X_val_orig, label=y_val_orig)
        test_dmatrix  = xgb.DMatrix(X_test_orig,  label=y_test_orig.values.flatten())

        if verbose:
            print(f"✅ Datos sin escalar para Site {site_id} - Train: {X_train_orig.shape[0]}, Val: {X_val_orig.shape[0]}, Test: {X_test_orig.shape[0]}")

        return (
            ochenta_dmatrix, train_dmatrix, valid_dmatrix,
            X_train_orig, y_train_orig, X_val_orig, y_val_orig,
            test_dmatrix, X_test_orig, y_test_orig,
            None, None,
            X_train_orig.shape[0], X_val_orig.shape[0]
        )

# === 2. Función para Clientes Federados ===
def load_data(
    partitioner_type,
    partition_id,
    num_partitions,
    centralised_eval_client,
    seed,
    scale_data=True
):
    site_id = partition_id + 1
    return load_partition(site_id, seed=seed, scale_data=scale_data)


def replace_keys(input_dict, match="-", target="_"):
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
