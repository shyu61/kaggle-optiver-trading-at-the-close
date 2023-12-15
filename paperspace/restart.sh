#! /usr/bin/bash

# install python
apt update
apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.10 python3.10-dev
echo "alias python='python3.10'" >> /root/.bashrc

# install pip
curl -O https://bootstrap.pypa.io/get-pip.py
python3.10 get-pip.py --user
echo "alias pip='python3.10 -m pip'" >> /root/.bashrc
rm get-pip.py

# install dependencies
awk '/-e file:\./ {flag=1; next} flag' "./kaggle-optiver-trading-at-the-close/requirements.lock" > "requirements.txt"
python3.10 -m pip install -r requirements.txt
source /root/.bashrc

echo "===== completed ====="
