import argparse
import json
import random
import string
import sys
import time

from eval_kit.bertscore.bertscore import BertScore
from eval_kit.bleu.bleu import Bleu

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import numpy as np


def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

class VAREvaluator(object):

    def __init__(self, prediction_filename=None, verbose=False):
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')

        self.verbose = verbose
        self.prediction = self.import_prediction(prediction_filename)

        self.tokenizer = PTBTokenizer()
        self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(),"METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
            ]
        if self.verbose:
            self.bertscorer = (BertScore(), "BertScore")

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print("Loading submission...")
        submission = json.load(open(prediction_filename))
        results = {}
        for vid_id in submission:
            results[vid_id] = submission[vid_id]
        return results

    def separate_evaluate(self):
        prediction_observed = {}
        prediction_hypothesis = {}
        for vid in self.prediction:
            prediction_observed[vid] = []
            prediction_hypothesis[vid] = []

            for _idx, _res in enumerate(self.prediction[vid]):
                if _res['is_hypothesis']:
                    prediction_hypothesis[vid].append(_res)
                else:
                    prediction_observed[vid].append(_res)

            assert len(prediction_hypothesis[vid]) != 0
            assert len(prediction_observed[vid]) != 0

        scores_o = {}
        scores_h = {}
        self.scores = {}
        if self.verbose: 
            print('Evaluating observed events...')
        scores_observed = self.example_evaluate(prediction_observed)
        if self.verbose: 
            print('Evaluating hypothesis events...')
        scores_hypothesis = self.example_evaluate(prediction_hypothesis)

        _res_list = [('observed', scores_observed, scores_o), ('hypothesis', scores_hypothesis, scores_h)]
        for _name, _scores, _dict in _res_list:
            for metric, score in _scores.items():
                if metric not in _dict:
                    _dict[metric] = []
                _dict[metric].append(score)

            self.scores[_name] = _dict


    def evaluate(self):
        aggregator = {}
        self.scores = {}
        scores = self.example_evaluate(self.prediction)
        for metric, score in scores.items():
            if metric not in self.scores:
                self.scores[metric] = []
            self.scores[metric].append(score)
        

    def example_evaluate(self, prediction):
        res = {}
        gts = {}
        unique_index = 0
        vid2capid = {}
        cur_res = {}
        cur_gts = {}

        for vid_id in prediction.keys():
            vid2capid[vid_id] = []
            for pred in prediction[vid_id]:
                cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                cur_gts[unique_index] = [{'caption': remove_nonascii(pred['gt_sentence'])}]
                vid2capid[vid_id].append(unique_index)
                unique_index += 1 

        all_scores = {}
        tokenize_res = self.tokenizer.tokenize(cur_res)
        tokenize_gts = self.tokenizer.tokenize(cur_gts)
        for vid in vid2capid.keys():
            res[vid] = {index:tokenize_res[index] for index in vid2capid[vid]}
            gts[vid] = {index:tokenize_gts[index] for index in vid2capid[vid]}
        output = {}
        for scorer, method in self.scorers:
            if self.verbose:
                print('computing %s score...'%(scorer.method()))

            kargs = {'gts':tokenize_gts, 'res':tokenize_res}
            score, scores = scorer.compute_score(**kargs)

            if type(method) == list: 
                for sc, scs, m in zip(score, scores, method):
                    output[m] = float(sc)
                    if self.verbose: 
                        print("Calculated %s: %0.5f"%(m, sc))
            else:
                output[method] = np.mean(list(scores))
                if self.verbose: 
                    print("Calculated %s: %0.3f" % (method, output[method]))

        if self.verbose: 
            scorer, method = self.bertscorer
            kargs = {'gts':gts, 'res':res}
            score, scores = scorer.compute_score(**kargs)
            output[method] = score 
        
        return output 


def main(args):
    evaluator = VAREvaluator(prediction_filename=args.submission, verbose=args.verbose)
    if args.separate:
        evaluator.separate_evaluate()
    else: evaluator.evaluate()

    return evaluator.scores

