import os
import sys

from pyActionRec.action_classifier import ActionClassifier
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, default='../data/VAR/video_feature')
parser.add_argument("--vid-path", type=str, default='../data/VAR/videos')
parser.add_argument("--video-file", type=str, default='../data/VAR/var_videos.txt')
parser.add_argument("--gpu", type=int, default=0)

args = parser.parse_args()

GPU=args.gpu

all_vid_list = list(open(args.video_file, 'r').readlines())
all_vid_list = [vid.strip() for vid in all_vid_list]

models=[]

models = [('./models/resnet200_anet_2016_deploy.prototxt',
           './models/resnet200_anet_2016.caffemodel',
           1.0, 0, True, 224)]
models.append(('./models/bn_inception_anet_2016_temporal_deploy.prototxt',
                './models/bn_inception_anet_2016_temporal.caffemodel.v5',
                0.2, 1, False, 224))

cls = ActionClassifier(models, dev_id=GPU)

process_list = {}
counter = 0
for vid in tqdm(all_vid_list):
    if os.path.isfile(os.path.join(args.data_path, vid+"_bn.npy")) and os.path.isfile(os.path.join(args.data_path, vid+"_resnet.npy")):
        counter += 1
        process_list[vid] = counter
    elif vid not in process_list:
        vid_path = os.path.join(args.vid_path, vid+'.mp4')
        data_path = os.path.join(args.data_path, vid)
        rst = cls.classify(vid_path, data_path)
        if rst != -1:
            print('Processed video: ', vid)
        counter += 1
        process_list[vid] = counter
