
from .action_caffe import CaffeNet
from .action_flow import FlowExtractor
# 
from .video_proc import VideoProc
from .anet_db import Video
import numpy as np
import time
import os
import subprocess
import cv2

# get video duration
def getLength(filename):
    result = subprocess.Popen(["ffprobe", filename],
        stdout = subprocess.PIPE, stderr = subprocess.STDOUT)

    return [x for x in result.stdout.readlines() if "Duration" in str(x, encoding='utf8')]


class ActionClassifier(object):
    """
    This class provides and end-to-end interface to classifying videos into activity classes
    """

    def __init__(self, models=list(), total_norm_weights=None, score_name='', dev_id=0):
        """
        Contruct an action classifier
        Args:
            models: list of tuples in the form of
                    (model_proto, model_params, model_fusion_weight, input_type, conv_support, input_size).
                    input_type is: 0-RGB, 1-Optical flow.
                    conv_support indicates whether the network supports convolution testing, which is faster. If this is
                    not supported, we will use oversampling instead
            total_norm_weights: sum of all model_fusion_weights when normalization is wanted, otherwise use None
        """

        self.__net_vec = [CaffeNet(x[0], x[1], dev_id,
                                   input_size=(340, 256) if x[4] else None
                                   ) for x in models]

        self.__net_weights = [float(x[2]) for x in models]

        if total_norm_weights is not None:
            s = sum(self.__net_weights)
            self.__net_weights = [x/s for x in self.__net_weights]

        self.__input_type = [x[3] for x in models]
        self.__conv_support = [x[4] for x in models]

        self.__num_net = len(models)

        # the input size of the network
        self.__input_size = [x[5] for x in models]

        # whether we should prepare flow stack
        self.__need_flow = max(self.__input_type) > 0

        # the name in the proto for action classes
        self.__score_name_resnet = 'caffe.Flatten_673'
        self.__score_name_bn = 'global_pool'

        if self.__need_flow:
            self.__flow_extractor = FlowExtractor(dev_id)

        self.anet_vid_feas = os.listdir('/mnt/data5/chen/datasets/VAR/anet_video_feature')
        self.extract_needed_cnt = 0


    def classify(self, video=None, data_path=None):
        """

        Args:
            video:

        Returns:
            scores:
            all_features:
        """
        import urllib.parse

        if os.path.isfile(video):
            return self._classify_from_file(video, data_path)

        raise ValueError("Unknown input data type")

    def _classify_from_file(self, filename, data_path):
        """
        Input a file on harddisk
        Args:
            filename:

        Returns:
            cls: classification scores
            all_features: RGB ResNet feature and Optical flow BN Inception feature in a list
        """
 
        duration = getLength(filename)

        duration_in_second = float(duration[0][15:17])*60+float(duration[0][18:23])
        
        info_dict = {
          'annotations': list(),
          'url': '',
          'duration': duration_in_second,
          'subset': 'testing'
         }

        vid_info = Video('0', info_dict)
        # update dummy video info...

        vid_info.path = filename

        vid_name = filename.split('/')[-1].split('.')[0]
        
        video_proc = VideoProc(vid_info)
        video_proc.open_video(True)

        # here we use interval of 30, roughly 1FPS
        frm_it = video_proc.frame_iter(timely=True, ignore_err=True, interval=0.5,
                                       length=6 if self.__need_flow else 1,
                                    #    length=6,
                                       new_size=(340, 256))

        all_features = {'resnet':np.empty(shape=(0,2048)), 'bn':np.empty(shape=(0,1024))}
        all_start = time.time()

        cnt = 0

        model_mask = None
        # process model mask
        mask = [True] * self.__num_net
        n_model = self.__num_net
        if model_mask is not None:
            for i in range(len(model_mask)):
                mask[i] = model_mask[i]
                if not mask[i]:
                    n_model -= 1


        start = time.time()
        for frm_stack in frm_it:

            cnt += 1

            flow_stack = None
            for net, run, in_type, conv_support, net_input_size in \
                    zip(self.__net_vec, mask, self.__input_type, self.__conv_support, self.__input_size):
                if not run:
                    continue

                frame_size = (int(340 * net_input_size / 224), int(256 * net_input_size / 224))
                # frame = frm_stack[0]

                if in_type == 0:
                    # RGB input
                    # for now we only sample one frame w/o applying mean-pooling 
                    all_features['resnet'] = np.concatenate((all_features['resnet'], \
                                net.predict_single_frame(frm_stack[:1], self.__score_name_resnet,
                                       over_sample=False,
                                       frame_size=None if net_input_size == 224 else frame_size
                                       )
                                ), axis=0)

                elif in_type == 1:
                    # Flow input
                    if flow_stack is None:
                        # Extract flow if necessary
                        # we disabled spatial data aug
                        # the size for flow frames are 224 x 224, hard coded
                        flow_frame_size = (224, 224)
                        flow_stack = self.__flow_extractor.extract_flow(frm_stack, flow_frame_size)

                    # store only the optical flow feature for the center crop
                    bn_center = np.squeeze(net.predict_single_flow_stack(flow_stack, self.__score_name_bn,
                                       over_sample = False))
                    bn_center = np.reshape(bn_center, (1, bn_center.shape[0]))
                    all_features['bn'] = np.concatenate((all_features['bn'], bn_center), axis=0)


        end = time.time()
        elapsed = end - start
        print("Frame sample total {}: {} second".format(cnt, elapsed))

        np.save(data_path+"_resnet.npy",all_features['resnet'])
        np.save(data_path+"_bn.npy",all_features['bn'])

        return 1

