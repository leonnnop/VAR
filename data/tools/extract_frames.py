# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Lock, Pool

import mmcv
import numpy as np


def extract_frame(vid_item):
    """Generate optical flow using dense flow.

    Args:
        vid_item (list): Video item containing video full path,
            video (short) path, video id.

    Returns:
        bool: Whether generate optical flow successfully.
    """
    full_path, vid_path, vid_id, report_file = vid_item
    if '/' in vid_path:
        act_name = osp.basename(osp.dirname(vid_path))
        out_full_path = osp.join(args.out_dir, act_name)
    else:
        out_full_path = args.out_dir

    run_success = -1

    try:
        video_name = osp.splitext(osp.basename(vid_path))[0]
        out_full_path = osp.join(out_full_path, video_name)

        vr = mmcv.VideoReader(full_path)
        for i, vr_frame in enumerate(vr):
            if vr_frame is not None:
                w, h, _ = np.shape(vr_frame)
                if args.new_short == 0:
                    if args.new_width == 0 or args.new_height == 0:
                        # Keep original shape
                        out_img = vr_frame
                    else:
                        out_img = mmcv.imresize(
                            vr_frame,
                            (args.new_width, args.new_height))
                else:
                    if min(h, w) == h:
                        new_h = args.new_short
                        new_w = int((new_h / h) * w)
                    else:
                        new_w = args.new_short
                        new_h = int((new_w / w) * h)
                    out_img = mmcv.imresize(vr_frame, (new_h, new_w))
                mmcv.imwrite(out_img,
                                f'{out_full_path}/img_{i + 1:05d}.jpg')
            else:
                warnings.warn(
                    'Length inconsistent!'
                    f'Early stop with {i + 1} out of {len(vr)} frames.'
                )
                break
        run_success = 0
    except Exception:
        run_success = -1


    if run_success == 0:
        print(f'{vid_id} {vid_path} done')
        sys.stdout.flush()

        lock.acquire()
        with open(report_file, 'a') as f:
            line = full_path + '\n'
            f.write(line)
        lock.release()
    else:
        print(f'{vid_id} {vid_path} got something wrong')
        sys.stdout.flush()

    return True


def parse_args():
    parser = argparse.ArgumentParser(description='extract rgb frames')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='number of workers to build rawframes')
    parser.add_argument(
        '--out-format',
        type=str,
        default='jpg',
        choices=['jpg', 'h5', 'png'],
        help='output format')
    parser.add_argument(
        '--ext',
        type=str,
        default='avi',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--mixed-ext',
        action='store_true',
        help='process video files with mixed extensions')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument(
        '--new-short',
        type=int,
        default=0,
        help='resize image short side length keeping ratio')
    parser.add_argument(
        '--report-file',
        type=str,
        default='build_report.txt',
        help='report to record files which have been successfully processed')
    args = parser.parse_args()

    return args


def init(lock_):
    global lock
    lock = lock_


if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    print('Reading videos from folder: ', args.src_dir)
    print('Extension of videos: ', args.ext)
    fullpath_list = glob.glob(args.src_dir + '/*' + '.' + args.ext)
    print('Total number of videos found: ', len(fullpath_list))

    vid_list = list(map(osp.basename, fullpath_list))

    lock = Lock()
    pool = Pool(args.num_worker, initializer=init, initargs=(lock, ))
    pool.map(
        extract_frame,
        zip(fullpath_list, vid_list, range(len(vid_list)),
            len(vid_list) * [args.report_file]))
    pool.close()
    pool.join()
