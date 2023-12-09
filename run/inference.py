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
        if "currently_scored" in df.columns:
            df = df.drop("currently_scored")
        df = preprocessing(df)
        df = feature_engineering(df)

        preds = []
        for kind in cfg.model.kinds:
            preds.append(
                np.mean([model.predict(df.to_pandas()) for model in models[kind]], 0)
            )
        sample_prediction["target"] = np.mean(preds, 0)
        env.predict(sample_prediction)


if __name__ == "__main__":
    main()
