from .utils import *
import os

logger = get_logger()


class Instance(object):
    """
    Representing an instance of activity in the videos
    """

    def __init__(self, idx, anno, vid_id, vid_info, name_num_mapping):
        self._starting, self._ending = anno['segment'][0], anno['segment'][1]
        self._str_label = anno['label']
        self._total_duration = vid_info['duration']
        self._idx = idx
        self._vid_id = vid_id
        self._file_path = None

        if name_num_mapping:
            self._num_label = name_num_mapping[self._str_label]

    @property
    def time_span(self):
        return self._starting, self._ending

    @property
    def covering_ratio(self):
        return self._starting / float(self._total_duration), self._ending / float(self._total_duration)

    @property
    def num_label(self):
        return self._num_label

    @property
    def label(self):
        return self._str_label

    @property
    def name(self):
        return '{}_{}'.format(self._vid_id, self._idx)

    @property
    def path(self):
        if self._file_path is None:
            raise ValueError("This instance is not associated to a file on disk. Maybe the file is missing?")
        return self._file_path

    @path.setter
    def path(self, path):
        self._file_path = path


class Video(object):
    """
    This class represents one video in the activity-net db
    """
    def __init__(self, key, info, name_idx_mapping=None):
        self._id = key
        self._info_dict = info
        self._instances = [Instance(i, x, self._id, self._info_dict, name_idx_mapping)
                           for i, x in enumerate(self._info_dict['annotations'])]
        self._file_path = None

    @property
    def id(self):
        return self._id

    @property
    def url(self):
        return self._info_dict['url']

    @property
    def instances(self):
        return self._instances

    @property
    def duration(self):
        return self._info_dict['duration']

    @property
    def subset(self):
        return self._info_dict['subset']

    @property
    def instance(self):
        return self._instances

    @property
    def path(self):
        if self._file_path is None:
            raise ValueError("This video is not associated to a file on disk. Maybe the file is missing?")
        return self._file_path

    @path.setter
    def path(self, path):
        self._file_path = path
