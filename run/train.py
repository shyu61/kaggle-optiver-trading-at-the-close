import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import catboost as cbt
import hydra
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

# import xgboost as xgb
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.processing import feature_engineering, preprocessing

log = logging.getLogger(__name__)


def train_cv_for_ensemble(
    cfg: DictConfig,
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any]]:
    trained_models, best_iters, scores = {}, {}, []
    for model_name in model_dict.keys():
        trained_models[model_name] = []
        best_iters[model_name] = []

    gkf = GroupKFold(n_splits=cfg.n_splits)
    for i, (train_idx, valid_idx) in enumerate(
        tqdm(gkf.split(X, groups=X["stock_id"]), total=cfg.n_splits)
    ):
        X_train, X_valid = (
            X[train_idx].drop("stock_id").to_pandas(),
            X[valid_idx].drop("stock_id").to_pandas(),
        )
        y_train, y_valid = y[train_idx].to_pandas(), y[valid_idx].to_pandas()

        preds_dict = {}
        for model_name, model in model_dict.items():
            if model_name == "lgb":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    callbacks=[lgb.early_stopping(100)],
                )
                best_iters[model_name].append(model.best_iteration_)
            else:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=100,
                    verbose=False,
                )
                best_iters[model_name].append(model.get_best_iteration())

            trained_models[model_name].append(model)
            preds_dict[model_name] = model.predict(X_valid)
            score = mean_absolute_error(
                y_pred=preds_dict[model_name],
                y_true=y_valid,
            )
            log.info(f"{model_name} fold {i} score: {score}")

        scores.append(
            mean_absolute_error(
                y_pred=np.mean(list(preds_dict.values()), axis=0),
                y_true=y_valid,
            )
        )

    best_iters = {k: int(np.mean(v)) for k, v in best_iters.items()}
    log.info(f"ensemble CV score: {np.mean(scores)}")
    log.info(f"best iters: {best_iters}")
    return trained_models, best_iters


def train_whole_dataset(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_dict: Dict[str, Any],
    best_iters: Dict[str, Any],
) -> Dict[str, Any]:
    trained_models = {}
    for model_name in model_dict.keys():
        trained_models[model_name] = []

    for model_name, model in model_dict.items():
        if model_name == "lgb":
            model.n_estimators = best_iters[model_name]
            model.fit(X, y)
        else:
            model.iterations = best_iters[model_name]
            model.fit(X, y, verbose=False)
        trained_models[model_name].append(model)

    return trained_models


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig):
    df = pl.read_csv(
        Path(cfg.dir.input) / "train.csv",
        dtypes={
            "stock_id": pl.UInt16,
            "date_id": pl.UInt16,
            "seconds_in_bucket": pl.UInt16,
            "imbalance_size": pl.Float32,
            "imbalance_buy_sell_flag": pl.Int8,
            "reference_price": pl.Float32,
            "matched_size": pl.Float32,
            "far_price": pl.Float32,
            "near_price": pl.Float32,
            "bid_price": pl.Float32,
            "bid_size": pl.Float32,
            "ask_price": pl.Float32,
            "ask_size": pl.Float32,
            "wap": pl.Float32,
            "target": pl.Float32,
            "time_id": pl.UInt32,
        },
    )
    df = preprocessing(df)
    df = feature_engineering(df, maintain_stock_id=True)

    X, y = df.drop("target"), df["target"]

    model_dict = {}
    if cfg.is_lgb_only:
        model_dict = {
            "lgb": lgb.LGBMRegressor(
                objective="regression_l1",
                n_estimators=500,
                verbosity=-1,
                random_state=42,
            ),
        }
    else:
        model_dict = {
            "lgb": lgb.LGBMRegressor(
                objective="regression_l1", n_estimators=500, verbosity=-1, n_jobs=-1
            ),
            "cbt": cbt.CatBoostRegressor(objective="MAE", iterations=3000),
            # "xgb": xgb.XGBRegressor(
            #     tree_method="hist", objective="reg:absoluteerror", n_estimators=500
            # ),
        }

    log.info("Training models...")
    _, best_iters = train_cv_for_ensemble(cfg, X, y, model_dict)
    if cfg.save_model:
        trained_models = train_whole_dataset(X, y, model_dict, best_iters)
        for model_name, models in trained_models.items():
            joblib.dump(models, f"{model_name}_models.joblib")


if __name__ == "__main__":
    main()
