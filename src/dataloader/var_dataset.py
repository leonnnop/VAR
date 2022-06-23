import os
import math
import nltk
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from src.utils import load_json, flat_list_of_lists

import logging
log_format = "%(asctime)-10s: %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


class VARDataset(Dataset):
    PAD_TOKEN = "[PAD]"  # padding of the whole sequence, note
    CLS_TOKEN = "[CLS]"  # leading token of the joint sequence
    SEP_TOKEN = "[SEP]"  # a separator for video and text
    VID_TOKEN = "[VID]"  # used as placeholder in the clip+sentence+text joint sequence

    BOS_TOKEN = "[BOS]"  # beginning of the sentence
    EOS_TOKEN = "[EOS]"  # ending of the sentence
    UNK_TOKEN = "[UNK]"

    SEN_TOKEN = "[SEN]" # used as placeholder in the clip+sentence+text joint sequence
    MSK_TOKEN = "[MSK]" 

    PAD = 0
    CLS = 1
    SEP = 2
    VID = 3
    BOS = 4
    EOS = 5
    UNK = 6
    SEN = 7
    MSK = 8
    IGNORE = -1  # used to calculate loss

    VID_TYPE = 0
    TEX_TYPE = 1
    SEN_TYPE = 2

    def __init__(self, dset_name, data_dir, 
                 max_t_len, max_v_len, max_n_len, K, mode="train", sample_mode='uniform',
                 word2idx_path=None):
        self.dset_name = dset_name
        self.data_dir = data_dir  # containing training data
        self.mode = mode
        self.sample_mode = sample_mode
        self.K = K # * number of cascade decoders
        assert K >= 1

        meta_dir = os.path.join(self.data_dir, 'data')

        if (word2idx_path is None) or (not os.path.exists(word2idx_path)):
            logging.info('[WARNING] word2idx_path load failed, use default path.')
            word2idx_path = os.path.join(data_dir, 'vocab_feature', 'word2idx.json')
        self.word2idx = load_json(word2idx_path)
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.video_feature_dir = os.path.join(self.data_dir, 'video_feature') 
        self.duration_file = os.path.join(meta_dir, 'var_video_duration_v1.0.csv')
        self.frame_to_second = self._load_duration()
        
        self.max_seq_len = max_v_len + max_t_len
        self.max_v_len = max_v_len
        self.max_t_len = max_t_len  # sentence
        self.max_n_len = max_n_len

        # data entries
        self.data = None
        self.set_data_mode(mode=mode)
        self.fix_missing()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        items, meta = self.convert_example_to_features(self.data[index])
        return items, meta


    def set_data_mode(self, mode):
        """mode: `train` or `val` or `test`"""
        assert mode in ['train', 'val', 'test']
        logging.info("Mode {}".format(mode))
        data_path = os.path.join(self.data_dir, 'data', "var_{}_v1.0.json".format(mode))
            
        self._load_data(data_path, mode)


    def _load_data(self, data_path, mode):
        logging.info("Loading data from {}".format(data_path))
        raw_data = load_json(data_path)
        data = []
        for k, line in raw_data.items():
            if line['split'] != mode: continue
            _line = {}
            valid_n = min(len(line["events"]), self.max_n_len)

            _line["name"] = k
            _line["video_ids"] = [e['video_id'] for e in line['events']]
            _line["events"] = line["events"][:valid_n]
            _line["hypothesis"] = line["hypothesis"]

            _line["video_ids"] = list(set(_line["video_ids"]))

            data.append(_line.copy())

        self.data = data

        logging.info("Loading complete! {} examples in total.".format(len(self)))


    def fix_missing(self):
        """filter our videos with no feature file"""
        missing_video_names = []
        missing_examples = []
        for e in self.data:
            for video_name in e['video_ids']:
                cur_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
                cur_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
                for p in [cur_path_bn, cur_path_resnet]:
                    if not os.path.exists(p):
                        missing_video_names.append(video_name)
                        missing_examples.append(e['name'])

        missing_video_names = list(set(missing_video_names))
        missing_examples = list(set(missing_examples))
        if len(missing_examples) > 0 or len(missing_video_names) > 0:
            logging.info("Missing {} features (clips/sentences) from {} videos".format(
                len(missing_video_names), len(set(missing_video_names))))
            logging.info("Missing {}".format(set(missing_video_names)))
        
            # * remove missing video in training
            self.data = [e for e in self.data if e["name"] not in missing_examples]


    def _load_duration(self):
        """https://github.com/salesforce/densecap/blob/master/data/anet_dataset.py#L120
        Since the features are extracted not at the exact 0.5 secs. To get the real time for each feature,
        use `(idx + 1) * frame_to_second[vid_name] `
        """
        frame_to_second = {}
        sampling_sec = 0.5  # hard coded, only support 0.5
        with open(self.duration_file, "r") as f:
            for line in f:
                vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                frame_to_second[vid_name] = float(vid_dur)*math.ceil(float(vid_frame)*1./float(vid_dur)*sampling_sec)*1./float(vid_frame)

        return frame_to_second


    def convert_example_to_features(self, example):

        example_name = example["name"]

        video_names = example['video_ids']
        video_features = {}
        for video_name in video_names: 
            feat_path_resnet = os.path.join(self.video_feature_dir, "{}_resnet.npy".format(video_name))
            feat_path_bn = os.path.join(self.video_feature_dir, "{}_bn.npy".format(video_name))
            video_feature = np.concatenate([np.load(feat_path_resnet), np.load(feat_path_bn)], axis=1)
            video_features[video_name] = video_feature

        num_sen = len(example["events"])
        single_video_features = []
        single_video_metas = []
        for clip_idx in range(num_sen):
            cur_data, cur_meta = self.clip_sentence_to_feature(example_name, example["events"][clip_idx], video_features)
            single_video_features.append(cur_data)
            single_video_metas.append(cur_meta)

        _mask_idx = example['hypothesis']
        assert _mask_idx < num_sen

        # * visual feature for aux loss
        _, f_video_all_mask, f_feat, f_video_temporal_tokens = self._construct_entire_video_features(single_video_features)

        # mask visual input
        single_video_features[_mask_idx]['video_feature'] = np.zeros_like(single_video_features[_mask_idx]['video_feature'])
        single_video_features[_mask_idx]['video_mask'] = [1] * len(single_video_features[_mask_idx]['video_mask'])
        single_video_features[_mask_idx]['video_tokens'] = [self.CLS_TOKEN] + [self.VID_TOKEN] * (self.max_v_len-2) + [self.SEP_TOKEN]
        single_video_metas[_mask_idx]['is_hypothesis'] = True
            
        _, video_all_mask, feat, video_temporal_tokens = self._construct_entire_video_features(single_video_features)
        
        input_labels_list = [[] for _ in range(self.K)]
        token_type_ids_list = [[] for _ in range(self.K)]
        input_mask_list = [[] for _ in range(self.K)]
        input_ids_list = [[] for _ in range(self.K)]

        def _fill_data(_idx):
            video_tokens = single_video_features[_idx]['video_tokens']
            video_mask = single_video_features[_idx]['video_mask']
            text_tokens = single_video_features[_idx]['text_tokens']
            text_mask = single_video_features[_idx]['text_mask']
            return video_tokens, video_mask, text_tokens, text_mask

        for _idx in range(self.max_n_len):

            video_tokens, video_mask, text_tokens, text_mask = _fill_data(_idx if _idx<num_sen else 0)

            # * prepare for dec1
            _input_tokens = video_tokens + text_tokens
            _input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in _input_tokens]

            _input_mask = video_mask + text_mask
            _token_type_ids = [self.VID_TYPE] * self.max_v_len + [self.TEX_TYPE] * self.max_t_len

            input_ids_list[0].append(np.array(_input_ids).astype(np.int64))
            token_type_ids_list[0].append(np.array(_token_type_ids).astype(np.int64))
            input_mask_list[0].append(np.array(_input_mask).astype(np.float32))

            # * shifted right, `-1` is ignored when calculating CrossEntropy Loss
            _input_labels = \
                [self.IGNORE] * len(video_tokens) + \
                [self.IGNORE if m == 0 else tid for tid, m in zip(_input_ids[-len(text_mask):], text_mask)][1:] + \
                [self.IGNORE]      

            input_labels_list[0].append(np.array(_input_labels).astype(np.int64))

            # * prepare for dec2+
            for k_idx in range(1, self.K):
                sen_tokens = [self.SEN_TOKEN]*self.max_n_len
                _input_tokens = video_tokens + sen_tokens + text_tokens
                _input_ids = [self.word2idx.get(t, self.word2idx[self.UNK_TOKEN]) for t in _input_tokens]

                sen_mask = [1]*(num_sen) + [0]*(self.max_n_len-num_sen)
                _input_mask = video_mask + sen_mask + text_mask

                _token_type_ids = [self.VID_TYPE] * self.max_v_len + [self.SEN_TYPE]*self.max_n_len + [self.TEX_TYPE] * self.max_t_len

                input_ids_list[k_idx].append(np.array(_input_ids).astype(np.int64))
                token_type_ids_list[k_idx].append(np.array(_token_type_ids).astype(np.int64))
                input_mask_list[k_idx].append(np.array(_input_mask).astype(np.float32))

                _input_labels = \
                    [self.IGNORE] * len(video_tokens) + \
                    [self.IGNORE] * len(sen_tokens) + \
                    [self.IGNORE if m == 0 else tid for tid, m in zip(_input_ids[-len(text_mask):], text_mask)][1:] + \
                    [self.IGNORE]      

                input_labels_list[k_idx].append(np.array(_input_labels).astype(np.int64))


        # * ignore all padded sentence
        for k_idx in range(self.K):
            for n_idx in range(num_sen, self.max_n_len):
                input_labels_list[k_idx][n_idx][:] = self.IGNORE


        data = dict(
            example_name=example_name,
            num_sen=num_sen,
            # * encoder input
            encoder_input=dict(
                video_features   = feat.astype(np.float32),
                temporal_tokens = np.array(video_temporal_tokens).astype(np.int64),
                video_mask  = np.array(video_all_mask).astype(np.float32),
            ),
            unmasked_encoder_input=dict(
                video_features   = f_feat.astype(np.float32),
                temporal_tokens = np.array(f_video_temporal_tokens).astype(np.int64),
                video_mask  = np.array(f_video_all_mask).astype(np.float32),
            ),
            # * decoder inputs
            decoder_input=dict(
                input_ids      = input_ids_list,
                input_mask     = input_mask_list,
                token_type_ids = token_type_ids_list
            ),
            # * gts for cascaded decoder
            gt=input_labels_list,
        )

        return data, single_video_metas


    def _construct_entire_video_features(self, single_video_features):
        video_tokens = []
        video_mask = []
        feats = []
        video_temporal_tokens = []
        for idx, clip_feat in enumerate(single_video_features):
            video_tokens += clip_feat['video_tokens'].copy()
            video_mask += clip_feat['video_mask'].copy()
            feats.append(clip_feat['video_feature'].copy())

        # * pad videos to max_n_len
        if len(single_video_features) < self.max_n_len:
            pad_v_n = self.max_n_len - len(single_video_features)

            video_tokens += [self.PAD_TOKEN] * self.max_v_len * pad_v_n
            video_mask += [0] * self.max_v_len * pad_v_n

            _feat = [np.zeros_like(single_video_features[0]['video_feature'])] * pad_v_n

            feats.extend(_feat)

        for idx in range(self.max_n_len):
            video_temporal_tokens += [idx]*len(clip_feat['video_tokens'])

        feat = np.concatenate(feats, axis=0)

        return video_tokens, video_mask, feat, video_temporal_tokens


    def clip_sentence_to_feature(self, name, event, video_features):
        """ 
            get video clip and sentence feature and tokens/masks
        """
        event['example_name'] = name
        event['is_hypothesis'] = False

        video_name = event['video_id']
        timestamp = event['timestamp']
        sentence = event['sentence']

        frm2sec = self.frame_to_second[video_name]

        # video + text tokens
        feat, video_tokens, video_mask = self._load_indexed_video_feature(video_features[video_name], timestamp, frm2sec)
        text_tokens, text_mask = self._tokenize_pad_sentence(sentence)

        data = dict(
            video_tokens=video_tokens,
            text_tokens=text_tokens,
            video_mask=video_mask,
            text_mask=text_mask,
            # 
            video_feature=feat.astype(np.float32)
        )

        meta = event

        return data, meta


    @classmethod
    def _convert_to_feat_index_st_ed(cls, feat_len, timestamp, frm2sec):
        """convert wall time st_ed to feature index st_ed"""
        st = int(math.floor(timestamp[0] / frm2sec))
        ed = int(math.ceil(timestamp[1] / frm2sec))
        ed = min(ed, feat_len-1)
        st = min(st, ed-1)
        assert st <= ed <= feat_len, "st {} <= ed {} <= feat_len {}".format(st, ed, feat_len)
        return st, ed


    def _load_indexed_video_feature(self, raw_feat, timestamp, frm2sec):
        """ 
        All non-PAD tokens are valid, will have a mask value of 1.
        Returns:
            feat: self.max_v_len
            video_tokens: self.max_v_len
            mask: self.max_v_len
        """
        max_v_l = self.max_v_len - 2
        feat_len = len(raw_feat)
        st, ed = self._convert_to_feat_index_st_ed(feat_len, timestamp, frm2sec)
        indexed_feat_len = ed - st + 1

        feat = np.zeros((self.max_v_len, raw_feat.shape[1]))

        if indexed_feat_len > max_v_l:
            # * linear (uniform) sample
            downsamlp_indices = np.linspace(st, ed, max_v_l, endpoint=True).astype(np.int).tolist()
            assert max(downsamlp_indices) < feat_len
            feat[1:max_v_l+1] = raw_feat[downsamlp_indices]

            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * max_v_l + [self.SEP_TOKEN]
            mask = [1] * (max_v_l + 2)
        else:
            valid_l = ed - st + 1
            feat[1:valid_l+1] = raw_feat[st:ed + 1]
            video_tokens = [self.CLS_TOKEN] + [self.VID_TOKEN] * valid_l + \
                           [self.SEP_TOKEN] + [self.PAD_TOKEN] * (max_v_l - valid_l)
            mask = [1] * (valid_l + 2) + [0] * (max_v_l - valid_l)

        return feat, video_tokens, mask

    def _tokenize_pad_sentence(self, sentence):
        """[BOS], [WORD1], [WORD2], ..., [WORDN], [EOS], [PAD], ..., [PAD], len == max_t_len
        All non-PAD values are valid, with a mask value of 1
        """
        max_t_len = self.max_t_len
        sentence_tokens = nltk.tokenize.word_tokenize(sentence.lower())[:max_t_len - 2]
        sentence_tokens = [self.BOS_TOKEN] + sentence_tokens + [self.EOS_TOKEN]

        # pad
        valid_l = len(sentence_tokens)
        mask = [1] * valid_l + [0] * (max_t_len - valid_l)
        sentence_tokens += [self.PAD_TOKEN] * (max_t_len - valid_l)
        return sentence_tokens, mask

    def convert_ids_to_sentence(self, ids, rm_padding=True, return_sentence_only=True):
        """A list of token ids"""
        rm_padding = True if return_sentence_only else rm_padding
        if rm_padding:
            raw_words = [self.idx2word[wid] for wid in ids if wid not in [self.PAD, self.IGNORE]]
        else:
            raw_words = [self.idx2word[wid] for wid in ids if wid != self.IGNORE]

        # get only sentences, the tokens between `[BOS]` and the first `[EOS]`
        if return_sentence_only:
            words = []
            for w in raw_words[1:]:  # no [BOS]
                if w != self.EOS_TOKEN:
                    words.append(w)
                else:
                    break
        else:
            words = raw_words
        return " ".join(words)


def prepare_batch_inputs(batch, device, non_blocking=False):
    def _recursive_to_device(v):
        if isinstance(v[0], list):
            return [_recursive_to_device(_v) for _v in v]
        elif isinstance(v[0], torch.Tensor):
            return [_v.to(device, non_blocking=non_blocking) for _v in v]
        else: return v

    batch_inputs = dict()

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_inputs[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(v, list):
            batch_inputs[k] = _recursive_to_device(v)
        elif isinstance(v, dict):
            batch_inputs[k] = prepare_batch_inputs(v, device, non_blocking)

    return batch_inputs


def sentences_collate(batch):
    """get rid of unexpected list transpose in default_collate
    https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py#L66
    """
    batch_meta = [[{"name": e['example_name'],
                   "clip_idx": e['clip_idx'],
                   "gt_sentence": e["sentence"],
                   "is_hypothesis": e["is_hypothesis"],
                   } for e in _batch[1]] for _batch in batch]  # change key

    padded_batch = default_collate([e[0] for e in batch])
    return padded_batch, batch_meta

def cal_performance(pred, gold):
    gold = gold[:, -pred.shape[1]:]
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(VARDataset.IGNORE)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum()

    return n_correct