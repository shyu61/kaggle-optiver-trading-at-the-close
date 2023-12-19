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
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from tqdm import tqdm


def set_logger():
    logger = getLogger(__name__)
    handler = StreamHandler(sys.stdout)
    handler.setLevel(DEBUG)
    logger.addHandler(handler)
    logger.setLevel(DEBUG)
    return logger


logger = set_logger()


def get_spliter(cfg: DictConfig, df: pl.DataFrame) -> Any:
    if cfg.cv.splitter == "group_k_fold":
        gkf = GroupKFold(n_splits=cfg.cv.n_splits)
        return gkf.split(df, groups=df["stock_id"])
    elif cfg.cv.splitter == "time_series_split":
        tscv = TimeSeriesSplit(n_splits=cfg.cv.n_splits, gap=cfg.cv.purge_gap)
        return tscv.split(df["date_id"].unique().sort())
    elif cfg.cv.splitter == "k_fold":
        kf = KFold(n_splits=cfg.cv.n_splits)
        return kf.split(df["date_id"].unique().sort())
    else:
        raise ValueError(f"{cfg.cv.splitter} is not supported.")


def train_purged_cv_for_ensemble(
    cfg: DictConfig,
    df: pl.DataFrame,
    model_names: List[str],
) -> Tuple[Dict[str, Any]]:
    trained_models, best_iters, scores = {}, {}, []
    for model_name in model_names:
        trained_models[model_name] = []
        best_iters[model_name] = []

    for i, (train_idx, valid_idx) in enumerate(
        tqdm(get_spliter(cfg, df), total=cfg.cv.n_splits)
    ):
        if cfg.cv.splitter == "group_k_fold":
            train, valid = (
                df.drop("stock_id")[train_idx],
                df.drop("stock_id")[valid_idx],
            )
        else:
            train = df.filter(pl.col("date_id").is_in(train_idx))
            valid = df.filter(pl.col("date_id").is_in(valid_idx))
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
                best_iters[model_name].append(
                    int(model.best_iteration_ * df.shape[0] / train.shape[0])
                )
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
                best_iters[model_name].append(
                    int(model.get_best_iteration() * df.shape[0] / train.shape[0])
                )

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
    best_iters = {k: int(np.mean(v)) for k, v in best_iters.items()}
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
    if cfg.cv.splitter == "group_k_fold":
        X_train, y_train = (
            df.drop("stock_id", "target").to_pandas(),
            df["target"].to_pandas(),
        )
    else:
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
    df = feature_engineering(df, keep_stock_id=(cfg.cv.splitter == "group_k_fold"))

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
