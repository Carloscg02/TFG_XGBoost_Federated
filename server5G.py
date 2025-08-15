"""xgboost_comprehensive: A Flower / XGBoost app."""

# from logging import INFO
# from typing import Dict, List, Optional

# import xgboost as xgb
# from xgboost_comprehensive.task import replace_keys, transform_dataset_to_dmatrix

# from datasets import load_dataset
# from flwr.common import Context, Parameters, Scalar
# from flwr.common.config import unflatten_dict
# from flwr.common.logger import log
# from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from flwr.server.client_manager import SimpleClientManager
# from flwr.server.client_proxy import ClientProxy
# from flwr.server.criterion import Criterion
# from flwr.server.strategy import FedXgbBagging, FedXgbCyclic


# class CyclicClientManager(SimpleClientManager):
#     """Provides a cyclic client selection rule."""

#     def sample(
#         self,
#         num_clients: int,
#         min_num_clients: Optional[int] = None,
#         criterion: Optional[Criterion] = None,
#     ) -> List[ClientProxy]:
#         """Sample a number of Flower ClientProxy instances."""

#         # Block until at least num_clients are connected.
#         if min_num_clients is None:
#             min_num_clients = num_clients
#         self.wait_for(min_num_clients)

#         # Sample clients which meet the criterion
#         available_cids = list(self.clients)
#         if criterion is not None:
#             available_cids = [
#                 cid for cid in available_cids if criterion.select(self.clients[cid])
#             ]

#         if num_clients > len(available_cids):
#             log(
#                 INFO,
#                 "Sampling failed: number of available clients"
#                 " (%s) is less than number of requested clients (%s).",
#                 len(available_cids),
#                 num_clients,
#             )
#             return []

#         # Return all available clients
#         return [self.clients[cid] for cid in available_cids]


# def get_evaluate_fn(test_data, params):
#     """Return a function for centralised evaluation."""

#     def evaluate_fn(
#         server_round: int, parameters: Parameters, config: Dict[str, Scalar]
#     ):
#         # If at the first round, skip the evaluation
#         if server_round == 0:
#             return 0, {}
#         else:
#             bst = xgb.Booster(params=params)
#             for para in parameters.tensors:
#                 para_b = bytearray(para)

#             # Load global model
#             bst.load_model(para_b)
#             # Run evaluation
#             eval_results = bst.eval_set(
#                 evals=[(test_data, "valid")],
#                 iteration=bst.num_boosted_rounds() - 1,
#             )
#             auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

#             # Save results to disk.
#             # Note we add new entry to the same file with each call to this function.
#             with open(f"./centralised_eval.txt", "a", encoding="utf-8") as fp:
#                 fp.write(f"Round:{server_round},AUC:{auc}\n")

#             return 0, {"AUC": auc}

#     return evaluate_fn


# def evaluate_metrics_aggregation(eval_metrics):
#     """Return an aggregated metric (AUC) for evaluation."""
#     total_num = sum([num for num, _ in eval_metrics])
#     auc_aggregated = (
#         sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
#     )
#     metrics_aggregated = {"AUC": auc_aggregated}
#     return metrics_aggregated


# def config_func(rnd: int) -> Dict[str, str]:
#     """Return a configuration with global epochs."""
#     config = {
#         "global_round": str(rnd),
#     }
#     return config


# def server_fn(context: Context):
#     # Read from config
#     cfg = replace_keys(unflatten_dict(context.run_config))
#     num_rounds = cfg["num_server_rounds"]
#     fraction_fit = cfg["fraction_fit"]
#     fraction_evaluate = cfg["fraction_evaluate"]
#     train_method = cfg["train_method"]
#     params = cfg["params"]
#     centralised_eval = cfg["centralised_eval"]

#     if centralised_eval:
#         # This is the exact same dataset as the one downloaded by the clients via
#         # FlowerDatasets. However, we don't use FlowerDatasets for the server since
#         # partitioning is not needed.
#         # We make use of the "test" split only
#         test_set = load_dataset("jxie/higgs")["test"]
#         test_set.set_format("numpy")
#         test_dmatrix = transform_dataset_to_dmatrix(test_set)

#     # Init an empty Parameter
#     parameters = Parameters(tensor_type="", tensors=[])

#     # Define strategy
#     if train_method == "bagging":
#         # Bagging training
#         strategy = FedXgbBagging(
#             evaluate_function=(
#                 get_evaluate_fn(test_dmatrix, params) if centralised_eval else None
#             ),
#             fraction_fit=fraction_fit,
#             fraction_evaluate=fraction_evaluate if not centralised_eval else 0.0,
#             on_evaluate_config_fn=config_func,
#             on_fit_config_fn=config_func,
#             evaluate_metrics_aggregation_fn=(
#                 evaluate_metrics_aggregation if not centralised_eval else None
#             ),
#             initial_parameters=parameters,
#         )
#     else:
#         # Cyclic training
#         strategy = FedXgbCyclic(
#             fraction_fit=1.0,
#             fraction_evaluate=1.0,
#             evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
#             on_evaluate_config_fn=config_func,
#             on_fit_config_fn=config_func,
#             initial_parameters=parameters,
#         )

#     config = ServerConfig(num_rounds=num_rounds)
#     client_manager = CyclicClientManager() if train_method == "cyclic" else None

#     return ServerAppComponents(
#         strategy=strategy, config=config, client_manager=client_manager
#     )


# # Create ServerApp
# app = ServerApp(
#     server_fn=server_fn,
# )


from logging import INFO
from typing import Dict, List, Optional
import os
import xgboost as xgb

from xgboost_comprehensive.task5G import replace_keys
from flwr.common import Context, Parameters, Scalar
from flwr.common.config import unflatten_dict
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
import csv
import os

class CyclicClientManager(SimpleClientManager):
    """Selección cíclica de clientes."""
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        cids = list(self.clients)
        if criterion is not None:
            cids = [cid for cid in cids if criterion.select(self.clients[cid])]

        if num_clients > len(cids):
            log(
                INFO,
                "Sampling failed: available %s < requested %s",
                len(cids),
                num_clients,
            )
            return []
        return [self.clients[cid] for cid in cids]

def get_evaluate_fn():
    """Solo guarda el modelo global cada ronda, sin usar datos de clientes."""
    def evaluate_fn(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        if not parameters.tensors:
            # Ronda 0: no hay modelo
            print("⏩ Skip evaluation: no global model available (first round)")
            return 0, {}

        bst = xgb.Booster()
        bst.load_model(bytearray(parameters.tensors[0]))
        os.makedirs("modelos_federados", exist_ok=True)
        path = f"modelos_federados/modelo_round_{server_round}.json"
        bst.save_model(path)
        print(f"💾 Modelo federado guardado: {path}")
        return 0, {}

    return evaluate_fn


# Flag para limpiar el CSV sólo en la primera llamada
_first_round_written = False

def evaluate_metrics_aggregation(eval_metrics):
    """Agrega las métricas 'val_rmse', 'train_rmse' y 'test_rmse'
    y las va guardando en convergence_federated_rmse.csv, una fila por ronda."""
    global _first_round_written
    log_path = "convergence_federated_rmse.csv"

    # 0) En la primera llamada, borramos cualquier fichero previo
    if not _first_round_written:
        if os.path.exists(log_path):
            os.remove(log_path)
        _first_round_written = True

    # 1) Determinar la ronda actual (número de filas escritas, excluyendo cabecera)
    if os.path.exists(log_path):
        with open(log_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # rows[0] = cabecera, así que la siguiente fila es len(rows)
        current_round = len(rows)
    else:
        current_round = 1

    # 2) Si es la primera ronda, (re)crear archivo y escribir cabecera
    if current_round == 1:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "val_rmse", "train_rmse", "test_rmse"])

    # 3) Agregar métricas (ponderadas por número de ejemplos)
    total_num   = sum(num for num, _ in eval_metrics)
    rmse_val    = sum(m.get("val_rmse",   0.0) * num for num, m in eval_metrics) / total_num
    rmse_train  = sum(m.get("train_rmse", 0.0) * num for num, m in eval_metrics) / total_num
    rmse_test   = sum(m.get("test_rmse",  0.0) * num for num, m in eval_metrics) / total_num

    # 4) Apéndice de la nueva fila
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([current_round, rmse_val, rmse_train, rmse_test])

    # 5) Devolver lo que espera Flower para el History
    return {
        "val_rmse":   rmse_val,
        "train_rmse": rmse_train,
        "test_rmse":  rmse_test,
    }

def config_func(rnd: int) -> Dict[str, str]:
    return {"global_round": str(rnd)}

def server_fn(context: Context):
    cfg                = replace_keys(unflatten_dict(context.run_config))
    num_rounds         = cfg["num_server_rounds"]
    fraction_fit       = cfg["fraction_fit"]
    fraction_evaluate  = cfg["fraction_evaluate"]
    train_method       = cfg["train_method"]
    params             = cfg["params"]

    # Parámetros vacíos iniciales
    parameters = Parameters(tensor_type="", tensors=[])

    if train_method == "bagging":
        strategy = FedXgbBagging(
            evaluate_function               = get_evaluate_fn(),
            fraction_fit                    = fraction_fit,
            fraction_evaluate               = fraction_evaluate,
            on_fit_config_fn                = config_func,
            on_evaluate_config_fn           = config_func,
            evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation,
            initial_parameters              = parameters,
        )
        client_manager = None
    else:
        strategy = FedXgbCyclic(
            fraction_fit                    = 1.0,
            fraction_evaluate               = 1.0,
            on_fit_config_fn                = config_func,
            on_evaluate_config_fn           = config_func,
            evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation,
            initial_parameters              = parameters,
        )
        client_manager = CyclicClientManager()

    return ServerAppComponents(
        strategy       = strategy,
        config         = ServerConfig(num_rounds=num_rounds),
        client_manager = client_manager,
    )

# 6) Crear ServerApp
app = ServerApp(server_fn=server_fn)
