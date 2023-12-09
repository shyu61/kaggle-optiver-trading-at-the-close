from pathlib import Path
from typing import Any, List

import catboost as cbt
import hydra
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

# import xgboost as xgb
from omegaconf import DictConfig
from sklearn.model_selection import TimeSeriesSplit

from src.processing import feature_engineering


def train_cv(
    cfg: DictConfig, X: pd.DataFrame, y: pd.DataFrame, model: Any
) -> List[Any]:
    models = []
    tsf = TimeSeriesSplit(n_splits=cfg.n_split)
    for i, (train_idx, valid_idx) in enumerate(tsf.split(X)):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=10,
            early_stopping_rounds=100,
        )
        models.append(model)
    return models


@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig):
    df_train = pl.read_csv(Path(cfg.dir.input) / "train.csv")
    df_train = feature_engineering(df_train)

    X, y = df_train.values, df_train["target"].values
    X = X[np.isfinite(y)]
    y = y[np.isfinite(y)]

    model_dict = {
        "lgb": lgb.LGBMRegressor(objective="regression_l1", n_estimators=500),
        "cbt": cbt.CatBoostRegressor(objective="MAE", iterations=3000),
        # "xgb": xgb.XGBRegressor(
        #     tree_method="hist", objective="reg:absoluteerror", n_estimators=500
        # ),
    }

    lgb_models = train_cv(cfg, X, y, model_dict["lgb"])
    cbt_models = train_cv(cfg, X, y, model_dict["cbt"])

    joblib.dump(lgb_models, cfg.dir.model / "lgb_models.joblib")
    joblib.dump(cbt_models, cfg.dir.model / "cbt_models.joblib")
    # xgb_models = train_cv(cfg, X, y, model_dict["xgb"])


if __name__ == "__main__":
    main()
