from pathlib import Path
from typing import Dict

import hydra
import joblib
import numpy as np
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


def add_time_id(df: pl.DataFrame) -> pl.DataFrame:
    tmp = (
        df.group_by("date_id", "seconds_in_bucket")
        .agg(pl.lit(0).alias("dummy"))
        .sort("date_id", "seconds_in_bucket")
    )
    tmp = tmp.with_columns(
        pl.arange(0, len(tmp)).cast(pl.UInt32).alias("time_id")
    ).drop("dummy")

    return df.join(tmp, on=["date_id", "seconds_in_bucket"])


def calibrate(size: int):
    sum = 0
    for i in range(size):
        sum += 1 / (2 ** (size - i))
    return 1 / sum


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

    for test, revealed_targets, sample_prediction in iter_test:
        df = pl.DataFrame(test)
        df = add_time_id(df)
        if "currently_scored" in df.columns:
            df = df.drop("currently_scored")
        df = preprocessing(df)
        df = feature_engineering(df)

        all_models_preds = []
        for kind in cfg.model.kinds:
            preds = np.zeros(len(df))
            for i in cfg.model.n_splits + 1:
                pred = models[kind][i].predict(df.to_pandas())
                preds += pred / (
                    2 ** (cfg.model.n_splits + 1 - i)
                )  # 1/64, 1/32, 1/16, 1/8, 1/4, 1/2
            preds *= calibrate(cfg.model.n_splits + 1)
            all_models_preds.append(preds)
        sample_prediction["target"] = np.mean(all_models_preds, 0)
        env.predict(sample_prediction)


if __name__ == "__main__":
    main()
