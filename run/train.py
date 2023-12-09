import logging
from pathlib import Path
from typing import Any, Dict

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
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from src.processing import feature_engineering, preprocessing


def train_cv_for_ensemble(
    cfg: DictConfig,
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_dict: Dict[str, Any],
) -> Dict[str, Any]:
    trained_models, scores = {}, []
    for model_name in model_dict.keys():
        trained_models[model_name] = []

    tsf = TimeSeriesSplit(n_splits=cfg.n_splits)
    for i, (train_idx, valid_idx) in enumerate(tqdm(tsf.split(X), total=cfg.n_splits)):
        X_train, X_valid = X[train_idx].to_pandas(), X[valid_idx].to_pandas()
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
            else:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=100,
                    verbose=False,
                )
            trained_models[model_name].append(model)
            preds_dict[model_name] = model.predict(X_valid)
            score = mean_absolute_error(
                y_pred=preds_dict[model_name],
                y_true=y_valid,
            )
            logging.info(f"{model_name} fold {i} score: {score}")

        scores.append(
            mean_absolute_error(
                y_pred=np.mean(list(preds_dict.values()), axis=0),
                y_true=y_valid,
            )
        )

    logging.info(f"ensemble CV score: {np.mean(scores)}")
    return trained_models


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig):
    df = pl.read_csv(
        Path(cfg.dir.input) / "train.csv",
        dtypes={"far_price": pl.Float64, "near_price": pl.Float64},
    )
    df = preprocessing(df)
    df = feature_engineering(df)

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
                objective="regression_l1", n_estimators=500, verbosity=-1
            ),
            "cbt": cbt.CatBoostRegressor(objective="MAE", iterations=3000),
            # "xgb": xgb.XGBRegressor(
            #     tree_method="hist", objective="reg:absoluteerror", n_estimators=500
            # ),
        }

    logging.info("Training models...")
    trained_models = train_cv_for_ensemble(cfg, X, y, model_dict)
    for model_name, models in trained_models.items():
        joblib.dump(models, f"{model_name}_models.joblib")


if __name__ == "__main__":
    main()
