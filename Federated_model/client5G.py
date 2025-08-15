"""xgboost_comprehensive: A Flower / XGBoost app."""

import warnings
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error

from xgboost_comprehensive.task5G import load_data, replace_keys
from flwr.client import Client, ClientApp
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
)
from flwr.common.config import unflatten_dict
from flwr.common.context import Context

warnings.filterwarnings("ignore", category=UserWarning)


class XgbClient(Client):
    def __init__(
        self,
        ochenta_dmatrix: xgb.DMatrix,
        train_dmatrix: xgb.DMatrix,
        valid_dmatrix: xgb.DMatrix,
        test_dmatrix: xgb.DMatrix,
        num_train: int,
        num_val: int,
        num_local_round: int,
        params: dict,
        train_method: str,
        X_80=None,
        y_80=None,
    ):
        self.ochenta_dmatrix = ochenta_dmatrix
        self.train_dmatrix   = train_dmatrix
        self.valid_dmatrix   = valid_dmatrix      
        self.test_dmatrix    = test_dmatrix
        self.num_train       = num_train
        self.num_val         = num_val
        self.num_local_round = num_local_round
        self.params          = params
        self.train_method    = train_method
        self.X_80            = X_80
        self.y_80            = y_80  

    def _local_boost(self, bst: xgb.Booster) -> xgb.Booster:
        for _ in range(self.num_local_round):
            bst.update(self.train_dmatrix, bst.num_boosted_rounds())
        if self.train_method == "bagging":
            start = bst.num_boosted_rounds() - self.num_local_round
            end   = bst.num_boosted_rounds()
            return bst[start:end]
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        rnd = int(ins.config.get("global_round", 1))
        if rnd == 1:
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            bst.load_model(bytearray(ins.parameters.tensors[0]))
            bst = self._local_boost(bst)
        raw = bst.save_raw("json")
        return FitRes(
            status       = Status(code=Code.OK, message="OK"),
            parameters   = Parameters(tensor_type="", tensors=[bytes(raw)]),
            num_examples = self.num_train,
            metrics      = {},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        
        if not ins.parameters.tensors:
            print("⏩ Skip evaluate: no global model received")
            return EvaluateRes(
                status=Status(code=Code.OK, message="No model received"),
                loss=0.0,
                num_examples=0,
                metrics={},
            )
        
        # 1) Carga modelo
        bst = xgb.Booster(params=self.params)
        bst.load_model(bytearray(ins.parameters.tensors[0]))
        
    
        # 2) Predicciones + etiquetas
        y_pred_train = bst.predict(self.train_dmatrix)
        y_true_train = self.train_dmatrix.get_label()
        y_pred_val   = bst.predict(self.valid_dmatrix)
        y_true_val   = self.valid_dmatrix.get_label()
        y_pred_test  = bst.predict(self.test_dmatrix)
        y_true_test  = self.test_dmatrix.get_label()
    
        # 4) Cálculo de RMSE
        rmse_train = float(np.sqrt(mean_squared_error(y_true_train, y_pred_train)))
        rmse_val   = float(np.sqrt(mean_squared_error(y_true_val,   y_pred_val)))
        rmse_test  = float(np.sqrt(mean_squared_error(y_true_test,  y_pred_test)))
        
        # 📊 Logs de comprobación
        print(f" y_test range: {y_true_test.min():.2f} - {y_true_test.max():.2f}")
        print(f" y_pred range: {y_pred_test.min():.2f} - {y_pred_test.max():.2f}")        
    
        # 5) Último print resumen
      #  print(f"[Debug] rmse_train={rmse_train:.4f}, rmse_val={rmse_val:.4f}, rmse_test={rmse_test:.4f}")
    
        # 6) Devuelve las métricas
        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=rmse_test,               # usa el test-RMSE como "loss"
            num_examples=self.num_val,
            metrics={
                "train_rmse": rmse_train,
                "val_rmse":   rmse_val,
                "test_rmse":  rmse_test,
            },
        )


def client_fn(context: Context) -> XgbClient:
    cfg            = replace_keys(unflatten_dict(context.run_config))
    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Ajuste de lr si se usa scaled_lr
    if cfg.get("scaled_lr", False):
        cfg["params"]["eta"] /= num_partitions

    # 1) Cargamos datos (task5G.py), desempaquetando todo:
    (
        ochenta_dmatrix,  # 80% combinado
        train_dmatrix,    # 64%
        valid_dmatrix,    # 16%
        X_80,
        y_80,
        X_train_orig,
        y_train_orig,
        X_val_orig,
        y_val_orig,
        test_dmatrix,
        X_test_orig,
        y_test_orig,
        scaler_X,
        scaler_y,
        num_train,
        num_val,
    ) = load_data(
        partitioner_type        = cfg["partitioner_type"],
        partition_id            = partition_id,
        num_partitions          = num_partitions,
        centralised_eval_client = cfg["centralised_eval_client"],
        seed                    = cfg["seed"],
        scale_data              = True,
    )

    # 2) Y ahora pasamos ochenta_dmatrix al cliente:
    return XgbClient(
        ochenta_dmatrix = ochenta_dmatrix,
        train_dmatrix   = train_dmatrix,
        valid_dmatrix   = valid_dmatrix,
        test_dmatrix    = test_dmatrix,
        num_train       = num_train,
        num_val         = num_val,
        num_local_round = cfg["local_epochs"],
        params          = cfg["params"],
        train_method    = cfg["train_method"],
        X_80            = X_80,
        y_80            = y_80,  
    )


app = ClientApp(client_fn)
