/apps/data-collection/

# Introduction

This app is a collection of scripts used to collect league pro play data and export them to an azure bucket.

# Installation steps

# Generate SSH key

ssh-keygen -t ed25519 -C "filip.a.niedzielski@gmail.com"

# Add to ssh-agent

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key to add to GitHub

cat ~/.ssh/id_ed25519.pub

# installs nvm (Node Version Manager)

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash

source ~/.bashrc

# download and install Node.js (you may need to restart the terminal)

nvm install 22

# verifies the right Node.js version is in the environment

node -v # should print `v22.11.0`

# verifies the right npm version is in the environment

npm -v # should print `10.9.0`

# Install yarn

npm install -g yarn

git clone git@github.com:Looyyd/draftking-monorepo.git

cd draftking-monorepo
yarn install

sudo apt install python3 python3-pip python3-venv build-essential python-is-python3

# TODO: not sure if this is needed

cd apps/data-collection
mkdir -p logs

python -m venv .venv
source .venv/bin/activate
pip install pandas pyarrow

npm install -g pm2

sudo vim /etc/systemd/system/league-data-collection.service

sudo systemctl daemon-reload
sudo systemctl enable league-data-collection
sudo systemctl start league-data-collection
sudo systemctl status league-data-collection
