import json
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from kaggle import KaggleApi

load_dotenv()
MODEL_DIR = Path(__file__).parent.parent / "data" / "model"


def copy_models(tmp_dir: Path):
    for model in MODEL_DIR.rglob("*.joblib"):
        shutil.copy(model, tmp_dir)


def create_metadata(tmp_dir: Path, title: str, user_name: str):
    metadata = {
        "title": title,
        "id": f"{user_name}/{title}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(tmp_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def main():
    api = KaggleApi()
    api.authenticate()

    tmp_dir = Path("./tmp/")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    copy_models(tmp_dir)
    create_metadata(tmp_dir, title="otc-models", user_name=os.getenv("KAGGLE_USERNAME"))

    api.dataset_create_version(
        folder=tmp_dir,
        version_notes="",
        dir_mode="tar",
        convert_to_csv=False,
    )
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
