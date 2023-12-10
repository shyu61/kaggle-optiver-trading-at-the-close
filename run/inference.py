from pathlib import Path
from typing import Dict

import hydra
import joblib
import numpy as np
import polars as pl
from omegaconf import DictConfig

from src.processing import feature_engineering, preprocessing


def load_models(cfg: DictConfig) -> Dict:
    all_models = {}
    for kind in cfg.model.kinds:
        all_models[kind] = joblib.load(Path(cfg.model.dir) / f"{kind}_models.joblib")
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


@hydra.main(config_path="conf", config_name="inference", version_base=None)
def main(cfg: DictConfig):
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

        preds = []
        for kind in cfg.model.kinds:
            preds.append(models[kind].predict(df.to_pandas()))
        sample_prediction["target"] = np.mean(preds, 0)
        env.predict(sample_prediction)


if __name__ == "__main__":
    main()
