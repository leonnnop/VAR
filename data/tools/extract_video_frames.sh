#!/usr/bin/env bash

python data/tools/extract_frames.py ./data/videos/ ./data/rawframes/ --ext mp4 --new-short 256

conda deactivate
conda remove -n downloader --all