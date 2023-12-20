from pathlib import Path
from typing import Dict, List

import hydra
import joblib
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig


def load_models(cfg: DictConfig) -> Dict:
    all_models = {}
    for kind in cfg.model.kinds:
        all_models[kind] = joblib.load(Path(cfg.model.dir) / f"{kind}_cv_models.joblib")
        all_models[kind].append(
            joblib.load(Path(cfg.model.dir) / f"{kind}_models.joblib")
        )
    return all_models


def __add_time_id(df: pl.DataFrame) -> pl.DataFrame:
    tmp = (
        df.group_by("date_id", "seconds_in_bucket")
        .agg(pl.lit(0).alias("dummy"))
        .sort("date_id", "seconds_in_bucket")
    )
    tmp = tmp.with_columns(
        pl.arange(0, len(tmp)).cast(pl.UInt32).alias("time_id")
    ).drop("dummy")

    return df.join(tmp, on=["date_id", "seconds_in_bucket"])


def prepare_data(
    test: pd.DataFrame, cache: pd.DataFrame, feature_engineering, preprocessing
) -> pl.DataFrame:
    cache = pd.concat([cache, test], ignore_index=True, axis=0)
    cache = (
        cache.groupby("stock_id")
        .tail(21)
        .sort_values(by=["date_id", "seconds_in_bucket", "stock_id"])
        .reset_index(drop=True)
    )
    df = pl.DataFrame(cache)
    df = __add_time_id(df)
    if "currently_scored" in df.columns:
        df = df.drop("currently_scored")
    df = preprocessing(df)
    df = feature_engineering(df)
    return df[-len(test):]


def ensemble_prediction(cfg: DictConfig, df: pl.DataFrame, models: List) -> np.ndarray:
    preds = []
    for kind in cfg.model.kinds:
        for i in range(cfg.model.n_splits + 1):
            preds.append(models[kind][i].predict(df.to_pandas()))
    return np.mean(preds, 0)


@hydra.main(config_path="conf", config_name="inference", version_base=None)
def main(cfg: DictConfig):
    if cfg.version == "v1":
        from src.processing_v1 import feature_engineering, preprocessing
    elif cfg.version == "v2":
        from src.processing_v2 import feature_engineering, preprocessing

    if cfg.env == "dev":
        import data.input.public_timeseries_testing_util as optiver2023
    else:
        import optiver2023

    models = load_models(cfg)
    env = optiver2023.make_env()
    iter_test = env.iter_test()

    cache = pd.DataFrame()

    # TODO: add cache for time_series feature engineering. e.g. lag features
    for test, revealed_targets, sample_prediction in iter_test:
        df = prepare_data(test, cache, feature_engineering, preprocessing)
        sample_prediction["target"] = ensemble_prediction(cfg, df, models)
        env.predict(sample_prediction)


if __name__ == "__main__":
    main()
