import sys

import numpy as np
import cv2

from . import caffemodel2pytorch as caffe
import torch
torch.set_grad_enabled(False)
from torchvision import transforms

class CaffeNet(object):

    def __init__(self, net_proto, net_weights, device_id, input_size=None):
        caffe.set_mode_gpu()
        caffe.set_device(device_id)
        # self._net = caffe.Net(net_proto, net_weights, caffe.TEST)

        kwargs = {'caffe_proto' : './caffe.proto',
                    'phase'    : caffe.TEST, 
                    'weights'  : net_weights
                }
        prototxt=net_proto

        self._net = caffe.Net(prototxt, **kwargs)
        self._net.eval()

        transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[104, 117, 123], std=[1,1,1]),
            ])

        self._transformer = transformer # transformer for RGB fea extractor


    def predict_single_frame(self, frame, feature_name, over_sample=True, multiscale=None, frame_size=None):

        if frame_size is not None:
            frame = [cv2.resize(x, frame_size) for x in frame]

        # avoid being rescale to 0-1
        os_frame = np.array(frame).astype(np.float32)
        data = torch.stack([self._transformer(x) for x in os_frame], dim=0).numpy()

        out = self._net.forward(data=data, out=[feature_name])
        return out[feature_name].copy()

    def predict_single_flow_stack(self, frame, feature_name, over_sample=True):

        os_frame = np.array([frame,])

        data = os_frame - 128
        out = self._net.forward(data=data, out=[feature_name])
        return out[feature_name].copy()
