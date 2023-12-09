# optiver-trading-at-the-close

## Preparation
- rye
  - [Installation](https://rye-up.com/guide/installation/)

## Usage
- Local
```bash
rye sync

# train
python -m run.train --dir=local

# inference
python -m run.inference --dir=local

# upload model to kaggle
python -m tools.upload_model
```
- Kaggle
```bash
# inference
python -m run.inference --dir=kaggle
```
