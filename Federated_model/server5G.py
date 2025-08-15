"""xgboost_comprehensive: A Flower / XGBoost app."""

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
