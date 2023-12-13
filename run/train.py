import sys
from logging import DEBUG, StreamHandler, getLogger
from pathlib import Path
from typing import Any, Dict, List, Tuple

import catboost as cbt
import hydra
import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.processing import feature_engineering, preprocessing


def set_logger():
    logger = getLogger(__name__)
    handler = StreamHandler(sys.stdout)
    handler.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.setLevel(DEBUG)
    return logger


logger = set_logger()


def train_cv_for_ensemble(
    cfg: DictConfig,
    X: pl.DataFrame,
    y: pl.DataFrame,
    model_names: List[str],
) -> Tuple[Dict[str, Any]]:
    trained_models, best_iters, scores = {}, {}, []
    for model_name in model_names:
        trained_models[model_name] = []
        best_iters[model_name] = []

    all_preds = np.zeros(len(X))
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
        for model_name in model_names:
            model = init_model(model_name)
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
            all_preds[valid_idx] = preds_dict[model_name]
            score = mean_absolute_error(
                y_pred=preds_dict[model_name],
                y_true=y_valid,
            )
            logger.info(f"{model_name} fold {i} score: {score}")

        scores.append(
            mean_absolute_error(
                y_pred=np.mean(list(preds_dict.values()), axis=0),
                y_true=y_valid,
            )
        )

    best_iters = {k: int(np.mean(v)) for k, v in best_iters.items()}
    logger.info(f"ensemble CV score: {np.mean(scores)}")
    logger.info(f"best iters: {best_iters}")
    return trained_models, best_iters, all_preds


def train_whole_dataset(
    X: pl.DataFrame,
    y: pl.DataFrame,
    model_names: List[str],
    best_iters: Dict[str, Any],
) -> Dict[str, Any]:
    trained_models = {}
    X_train, y_train = X.drop("stock_id").to_pandas(), y.to_pandas()

    for model_name in tqdm(model_names, total=len(model_names)):
        model = init_model(model_name, {"n_estimators": best_iters[model_name]})
        if model_name == "lgb":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, verbose=False)
        trained_models[model_name] = model

    return trained_models


def init_model(model_type: str, params: Dict = {}) -> Any:
    if model_type == "lgb":
        return lgb.LGBMRegressor(
            **{
                "objective": "regression_l1",
                "n_estimators": 500,
                "verbosity": -1,
                "random_state": 42,
                **params,
            }
        )
    elif model_type == "cbt":
        return cbt.CatBoostRegressor(
            **{"objective": "MAE", "n_estimators": 3000, "random_seed": 42, **params}
        )
    elif model_type == "xgb":
        return xgb.XGBRegressor(
            **{
                "tree_method": "hist",
                "objective": "reg:absoluteerror",
                "n_estimators": 500,
                **params,
            }
        )
    else:
        raise ValueError(f"model_type {model_type} is not supported.")


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig):
    df = pl.read_csv(Path(cfg.dir.input) / "train.csv")
    df = preprocessing(df)
    df = feature_engineering(df, maintain_stock_id=True)

    X, y = df.drop("target"), df["target"]

    model_names = ["lgb"]
    if not cfg.is_lgb_only:
        model_names.extend(["cbt"])
        # model_names.extend(["cbt", "xgb"])

    logger.info("Training models...")
    _, best_iters, _ = train_cv_for_ensemble(cfg, X, y, model_names)
    # best_iters = {"lgb": 492, "cbt": 2645}
    if cfg.save_model:
        trained_models = train_whole_dataset(X, y, model_names, best_iters)
        for model_name, models in trained_models.items():
            joblib.dump(models, f"{model_name}_models.joblib")


if __name__ == "__main__":
    main()
