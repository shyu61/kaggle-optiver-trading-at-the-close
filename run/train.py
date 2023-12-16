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
from tqdm import tqdm


def set_logger():
    logger = getLogger(__name__)
    handler = StreamHandler(sys.stdout)
    handler.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.setLevel(DEBUG)
    return logger


logger = set_logger()


def train_purged_cv_for_ensemble(
    cfg: DictConfig,
    df: pl.DataFrame,
    model_names: List[str],
) -> Tuple[Dict[str, Any]]:
    trained_models, best_iters, scores = {}, {}, []
    for model_name in model_names:
        trained_models[model_name] = []
        best_iters[model_name] = []

    fold_size = df["date_id"].n_unique() // cfg.cv.n_splits
    for i in tqdm(range(cfg.cv.n_splits), total=cfg.cv.n_splits):
        start, end = i * fold_size, (i + 1) * fold_size
        # fmt: off
        train_expr = (
            (~pl.col("date_id").is_between(start, end))
            & (~pl.col("date_id").is_between(start - cfg.cv.purge_gap, start + cfg.cv.purge_gap))
            & (~pl.col("date_id").is_between(end - cfg.cv.purge_gap, end + cfg.cv.purge_gap))
        )
        valid_expr = (
            pl.col("date_id").is_between(start, end)
            & (~pl.col("date_id").is_between(start - cfg.cv.purge_gap, start + cfg.cv.purge_gap))
            & (~pl.col("date_id").is_between(end - cfg.cv.purge_gap, end + cfg.cv.purge_gap))
        )
        # fmt: on
        train, valid = df.filter(train_expr), df.filter(valid_expr)
        X_train, X_valid = (
            train.drop("target").to_pandas(),
            valid.drop("target").to_pandas(),
        )
        y_train, y_valid = train["target"].to_pandas(), valid["target"].to_pandas()

        preds_dict = {}
        for model_name in model_names:
            model = init_model(cfg, model_name)
            if model_name == "lgb":
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    callbacks=[lgb.early_stopping(100)],
                )
                logger.info(f"{model_name} fold {i} iteration: {model.best_iteration_}")
                best_iters[model_name].append(model.best_iteration_)
            else:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=100,
                    verbose=False,
                )
                logger.info(
                    f"{model_name} fold {i} iteration: {model.get_best_iteration()}"
                )
                best_iters[model_name].append(model.get_best_iteration())

            trained_models[model_name].append(model)
            preds_dict[model_name] = model.predict(X_valid)
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

    # best_iterの結果を使うのは全データ学習時なので、回数を補正する
    best_iters = {
        k: int(np.mean(v)) * cfg.cv.n_splits // (cfg.cv.n_splits - 1)
        for k, v in best_iters.items()
    }
    logger.info(f"ensemble CV score: {np.mean(scores)}")
    logger.info(f"best iters: {best_iters}")
    return trained_models, best_iters


def train_whole_dataset(
    cfg: DictConfig,
    df: pl.DataFrame,
    model_names: List[str],
    best_iters: Dict[str, Any],
) -> Dict[str, Any]:
    trained_models = {}
    X_train, y_train = df.drop("target").to_pandas(), df["target"].to_pandas()

    for model_name in tqdm(model_names, total=len(model_names)):
        model = init_model(cfg, model_name, {"n_estimators": best_iters[model_name]})
        if model_name == "lgb":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, verbose=False)
        trained_models[model_name] = model

    return trained_models


def init_model(cfg: DictConfig, model_type: str, params: Dict = {}) -> Any:
    if model_type == "lgb":
        if cfg.env == "paperspace":
            return lgb.LGBMRegressor(
                **{
                    "objective": "mae",
                    "n_estimators": 6000,
                    "num_leaves": 256,
                    "subsample": 0.6,
                    "colsample_bytree": 0.8,
                    "learning_rate": 0.01,
                    "max_depth": 11,
                    "n_jobs": -1,
                    "device": "cuda",
                    "importance_type": "gain",
                    "reg_alpha": 0.2,
                    "reg_lambda": 3.25,
                    "verbosity": -1,
                    "random_state": 42,
                    **params,
                }
            )
        else:
            return lgb.LGBMRegressor(
                **{
                    "objective": "mae",
                    "n_estimators": 500,
                    "verbosity": -1,
                    "random_state": 42,
                    **params,
                }
            )
    elif model_type == "cbt":
        return cbt.CatBoostRegressor(
            **{
                "objective": "MAE",
                "n_estimators": 3000,
                **({"task_type": "GPU"} if cfg.env == "paperspace" else {}),
                "random_seed": 42,
                **params,
            }
        )
    elif model_type == "xgb":
        return xgb.XGBRegressor(
            **{
                "tree_method": "hist",
                "objective": "reg:absoluteerror",
                "n_estimators": 500,
                **({"device": "cuda"} if cfg.env == "paperspace" else {}),
                **params,
            }
        )
    else:
        raise ValueError(f"model_type {model_type} is not supported.")


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig):
    if cfg.version == "v1":
        from src.processing_v1 import feature_engineering, preprocessing
    elif cfg.version == "v2":
        from src.processing_v2 import feature_engineering, preprocessing

    df = pl.read_csv(Path(cfg.dir.input) / "train.csv")
    df = preprocessing(df)
    df = feature_engineering(df)

    model_names = []
    for model_name in cfg.model.kinds:
        model_names.append(model_name)

    logger.info("Training models...")
    trained_cv_models, best_iters = train_purged_cv_for_ensemble(cfg, df, model_names)
    for model_name, models in trained_cv_models.items():
        joblib.dump(models, f"{model_name}_cv_models.joblib")
    if cfg.whole_training:
        trained_models = train_whole_dataset(cfg, df, model_names, best_iters)
        for model_name, models in trained_models.items():
            joblib.dump(models, f"{model_name}_models.joblib")


if __name__ == "__main__":
    main()
