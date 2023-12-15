#! /usr/bin/bash

source /notebooks/.env

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

# setup ta-lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure
make
make install
cd ..
rm ta-lib-0.4.0-src.tar.gz

# setup ssh
mkdir .ssh/
cat <<EOS > .ssh/id_ed25519_paper.pub
${PUBLIC_KEY}
EOS

cat <<EOS > .ssh/id_ed25519_paper
${PRIVATE_KEY}
EOS

# add config
cat <<EOS > .ssh/config
Host github.com
    User git
    IdentityFile /notebooks/.ssh/id_ed25519_paper
    StrictHostKeyChecking no
EOS

chmod 600 .ssh/id_ed25519_paper

# clone repository
REPO="kaggle-optiver-trading-at-the-close"
GIT_SSH_COMMAND="ssh -F /notebooks/.ssh/config" git clone git@github.com:shyu61/"$REPO".git

# install dependencies
awk '/-e file:\./ {flag=1; next} flag' "./kaggle-optiver-trading-at-the-close/requirements.lock" > "requirements.txt"
python3.10 -m pip install -r requirements.txt
source /root/.bashrc

# checkout
cd $REPO
git checkout main

# download dataset
mkdir -p ./data/input
cd ./data/input
export KAGGLE_USERNAME=shyu61
export KAGGLE_KEY=${KAGGLE_KEY}

kaggle competitions download -c optiver-trading-at-the-close
unzip optiver-trading-at-the-close.zip
rm optiver-trading-at-the-close.zip
