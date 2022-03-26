#!/usr/bin/env bash

# set up environment
conda env create -f data/tools/environment.yml
conda activate downloader

pip install --upgrade youtube-dl
pip install mmcv

python data/tools/download.py
