from pathlib import Path
from typing import Dict

import hydra
import joblib
import numpy as np
import optiver2023
from omegaconf import DictConfig

from src.processing import feature_engineering


def load_models(cfg: DictConfig) -> Dict:
    all_models = {}
    for kind in cfg.model.kinds:
        all_models[kind] = joblib.load(Path(cfg.dir.model) / f"{kind}_models.joblib")
    return all_models


@hydra.main(config_path="conf", config_name="inference")
def main(cfg: DictConfig):
    models = load_models(cfg)
    env = optiver2023.make_env()
    iter_test = env.iter_test()

    counter = 0
    for test, revealed_targets, sample_prediction in iter_test:
        df = feature_engineering(test)

        preds = []
        for kind in cfg.model.kinds:
            preds.append(np.mean([model.predict(df) for model in models[kind]], 0))

        sample_prediction["target"] = np.mean(preds, 0)
        env.predict(sample_prediction)
        counter += 1


if __name__ == "__main__":
    main()
