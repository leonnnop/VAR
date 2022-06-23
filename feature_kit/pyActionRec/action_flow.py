import sys

import numpy as np
import cv2

def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound
    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

class FlowExtractor(object):

    def __init__(self, dev_id, bound=20, cuda_enabled=True):
        self.bound = bound
        if cuda_enabled:
            self.dtvl1_gpu = cv2.cuda_OpticalFlowDual_TVL1.create()
        else:
            self.dtvl1_cpu = cv2.optflow.createOptFlow_DualTVL1()
        self.cuda_enabled = cuda_enabled


    def func_extract_flow(self, frames):
        output = []
        prev_gray = cv2.cvtColor(frames[0].copy(), cv2.COLOR_BGR2GRAY)
        for frame in frames[1:]:
            gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            frame_0 = prev_gray
            frame_1 = gray

            flows = self.dtvl1_cpu.calc(frame_0, frame_1, None)
            
            flow_x = ToImg(flows[...,0], self.bound)
            flow_y = ToImg(flows[...,1], self.bound)
            output.append((flow_x.copy(), flow_y.copy()))

            prev_gray = gray

        return output

    def gpu_extract_flow(self, frames):
        output = []
        gpu_previous = cv2.cuda_GpuMat()
        gpu_previous.upload(frames[0].copy())
        prev_gray = cv2.cuda.cvtColor(gpu_previous, cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame.copy())

            gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)

            frame_0 = prev_gray
            frame_1 = gray

            flows = self.dtvl1_gpu.calc(frame_0, frame_1, None)

            flows_cpu = flows.download()
            
            flow_x = ToImg(flows_cpu[...,0], self.bound)
            flow_y = ToImg(flows_cpu[...,1], self.bound)
            output.append((flow_x.copy(), flow_y.copy()))

            prev_gray = gray

        return output

    def extract_flow(self, frame_list, new_size=None):
        """
        This function extracts the optical flow and interleave x and y channels
        :param frame_list:
        :return:
        """
        frame_size = frame_list[0].shape[:2]
        if self.cuda_enabled:
            rst = self.gpu_extract_flow(frame_list)
        else:
            rst = self.func_extract_flow(frame_list)

        n_out = len(rst)
        if new_size is None:
            ret = np.zeros((n_out*2, frame_size[0], frame_size[1]))
            for i in range(n_out):
                ret[2*i, :] = rst[i][0].astype('uint8').reshape(frame_size)
                ret[2*i+1, :] = rst[i][1].astype('uint8').reshape(frame_size)
        else:
            ret = np.zeros((n_out*2, new_size[1], new_size[0]))
            for i in range(n_out):
                ret[2*i, :] = cv2.resize(rst[i][0].astype('uint8').reshape(frame_size), new_size)
                ret[2*i+1, :] = cv2.resize(rst[i][1].astype('uint8').reshape(frame_size), new_size)

        return ret.astype(np.float32)

