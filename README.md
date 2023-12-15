# kaggle-optiver-trading-at-the-close

## Preparation
- rye
  - [Installation](https://rye-up.com/guide/installation/)

## Usage
- Local
```bash
rye sync

# train
python -m run.train dir=local exp_name=exp001 is_lgb_only=true
# simple train
python -m run.train dir=local exp_name=exp030 is_lgb_only=true save_model=false

# inference
python -m run.inference env=dev dir=local exp_name=exp001

# upload model to kaggle
python -m tools.upload_model --exp_name=exp001
```
- Kaggle
```bash
# inference
python -m run.inference dir=kaggle model.dir=/kaggle/input/otc-models
```
- Paperspace
```bash
GIT_SSH_COMMAND="ssh -F /notebooks/.ssh/config" git pull
python3.10 -m run.train dir=paperspace exp_name=exp001 is_lgb_only=true save_model=false
```
