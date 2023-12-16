#! /usr/bin/bash

# install python
apt update
apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install -y python3.10 python3.10-dev

# install pip
curl -O https://bootstrap.pypa.io/get-pip.py
python3.10 get-pip.py --user
rm get-pip.py

# setup for lightgbm
apt install -y \
    --no-install-recommends \
    cmake \
    build-essential \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev

# setup ta-lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure
make
make install
cd ..
rm ta-lib-0.4.0-src.tar.gz

# install dependencies
awk '/-e file:\./ {flag=1; next} flag' "./kaggle-optiver-trading-at-the-close/requirements.lock" > "requirements.txt"
python3.10 -m pip install -r requirements.txt

# install lightgbm for GPU
python3.10 -m pip uninstall lightgbm -y
python3.10 -m pip install lightgbm \
    --no-binary lightgbm \
    --no-cache lightgbm \
    --config-settings=cmake.define.USE_CUDA=ON

echo "===== completed ====="
