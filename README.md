# kaggle-optiver-trading-at-the-close

## Preparation
- rye
  - [Installation](https://rye-up.com/guide/installation/)

## Usage
- Local
```bash
rye sync

# train
python -m run.train dir=local exp_name=exp001 version=v1
# simple train
python -m run.train dir=local exp_name=exp030 whole_training=false version=v1

# inference
python -m run.inference env=dev dir=local exp_name=exp001 version=v1

# upload model to kaggle
python -m tools.upload_model --exp_name=exp001
```
- Kaggle
```bash
# inference
python -m run.inference dir=kaggle model.dir=/kaggle/input/otc-models version=v1
```
- Paperspace
```bash
GIT_SSH_COMMAND="ssh -F /notebooks/.ssh/config" git pull
python3.10 -m run.train dir=paperspace env=paperspace exp_name=exp001 version=v1 model.kinds=[lgb,cbt]
```
