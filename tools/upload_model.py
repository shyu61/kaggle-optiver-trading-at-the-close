import json
import shutil
from pathlib import Path

import click
from dotenv import load_dotenv
from kaggle import KaggleApi

load_dotenv()
MODEL_DIR = Path(__file__).parent.parent / "data" / "output" / "train"


def copy_models(tmp_dir: Path, exp_name: str):
    for model in (MODEL_DIR / exp_name / "single").rglob("*.joblib"):
        shutil.copy(model, tmp_dir)


def create_metadata(tmp_dir: Path, title: str):
    metadata = {
        "title": title,
        "id": f"shyu61/{title}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(tmp_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


@click.command()
@click.option("--new", "-n", is_flag=True)
@click.option("--exp_name", "-e", default="exp001")
def main(new: bool, exp_name: str):
    api = KaggleApi()
    api.authenticate()

    tmp_dir = Path("./tmp/")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    copy_models(tmp_dir, exp_name)
    create_metadata(tmp_dir, title="OTC-models")

    if new:
        api.dataset_create_new(
            folder=tmp_dir,
            dir_mode="tar",
            convert_to_csv=False,
            public=False,
        )
    else:
        api.dataset_create_version(
            folder=tmp_dir,
            version_notes="",
            dir_mode="tar",
            convert_to_csv=False,
        )
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
