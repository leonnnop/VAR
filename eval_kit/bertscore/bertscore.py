from bert_score import score
import re
import math

class BertScore:
    def __init__(self):
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res):

        assert(gts.keys() == res.keys())
        vidIds = gts.keys()

        hyp_input = []
        ref_input = []
        same_indices = []
        for id in vidIds:
            hypo = sum(res[id].values(), [])
            ref = sum(gts[id].values(), [])

            # Sanity check.
            assert(type(hypo) is list)
            assert(type(ref) is list)
            assert(len(hypo) == len(ref))

            hyp_input += hypo
            ref_input += ref
            # * average over event - > video
            same_indices.append(len(ref_input))

        p, r, f_scores = score(hyp_input, ref_input, \
            model_type='./eval_kit/roberta_large_619fd8c/', num_layers=17, baseline_path='./eval_kit/roberta_large_619fd8c/roberta-large.tsv',\
            lang='en', rescale_with_baseline=True, verbose=True)
        prev_idx = 0 
        aggreg_f1_scores = [] 
        for idx in same_indices: 
            aggreg_f1_scores.append(f_scores[prev_idx: idx].mean().cpu().item())
            prev_idx = idx 

        # * ignore missing sentence with garbage 
        _sum = sum([x for x in aggreg_f1_scores if not math.isnan(x)])
        return _sum/len(aggreg_f1_scores), aggreg_f1_scores

    def method(self):
        return "BertScore" 
