from pathlib import Path
from typing import Dict

import hydra
import joblib
import numpy as np
import optiver2023
from omegaconf import DictConfig

from src.processing import feature_engineering


def load_models(cfg: DictConfig) -> Dict:
    lgb_models = joblib.load(Path(cfg.dir.model) / "lgb_models.joblib")
    cbt_models = joblib.load(Path(cfg.dir.model) / "cbt_models.joblib")
    return {
        "lgb": lgb_models,
        "cbt": cbt_models,
    }


@hydra.main(config_path="conf", config_name="inference")
def main(cfg: DictConfig):
    models = load_models(cfg)
    env = optiver2023.make_env()
    iter_test = env.iter_test()

    counter = 0
    for test, revealed_targets, sample_prediction in iter_test:
        feat = feature_engineering(test)

        sample_prediction["target"] = np.mean(
            [model.predict(feat) for model in models], 0  # TODO
        )
        env.predict(sample_prediction)
        counter += 1


if __name__ == "__main__":
    main()
