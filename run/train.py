import logging
from pathlib import Path
from typing import Any, List

import catboost as cbt
import hydra
import joblib
import lightgbm as lgb
import pandas as pd
import polars as pl

# import xgboost as xgb
from omegaconf import DictConfig
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from src.processing import feature_engineering, preprocessing


def train_cv(
    cfg: DictConfig, X: pd.DataFrame, y: pd.DataFrame, model: Any
) -> List[Any]:
    models = []
    tsf = TimeSeriesSplit(n_splits=cfg.n_splits)
    for train_idx, valid_idx in tqdm(tsf.split(X), total=cfg.n_splits):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        if isinstance(model, lgb.LGBMRegressor):
            model.fit(
                X_train.to_pandas(),
                y_train.to_pandas(),
                eval_set=[(X_valid.to_pandas(), y_valid.to_pandas())],
                callbacks=[lgb.early_stopping(100)],
            )
        else:
            model.fit(
                X_train.to_pandas(),
                y_train.to_pandas(),
                eval_set=[(X_valid.to_pandas(), y_valid.to_pandas())],
                early_stopping_rounds=100,
                verbose=False,
            )
        models.append(model)
    return models


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig):
    df = pl.read_csv(
        Path(cfg.dir.input) / "train.csv",
        dtypes={"far_price": pl.Float64, "near_price": pl.Float64},
    )
    df = preprocessing(df)
    df = feature_engineering(df)

    X, y = df.drop("target"), df["target"]

    model_dict = {
        "lgb": lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=500, verbosity=-1
        ),
        "cbt": cbt.CatBoostRegressor(objective="MAE", iterations=3000),
        # "xgb": xgb.XGBRegressor(
        #     tree_method="hist", objective="reg:absoluteerror", n_estimators=500
        # ),
    }

    logging.info("Training lgb models...")
    lgb_models = train_cv(cfg, X, y, model_dict["lgb"])
    joblib.dump(lgb_models, "lgb_models.joblib")
    del lgb_models

    logging.info("Training cbt models...")
    cbt_models = train_cv(cfg, X, y, model_dict["cbt"])
    joblib.dump(cbt_models, "cbt_models.joblib")
    del cbt_models

    # logging.info("Training xgb models...")
    # xgb_models = train_cv(cfg, X, y, model_dict["xgb"])
    # joblib.dump(xgb_models, cfg.dir.model / "xgb_models.joblib")
    # del xgb_models


if __name__ == "__main__":
    main()
