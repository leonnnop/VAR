import argparse
import os
import os.path as osp
import sys
from eval_kit.evaluate2021 import main as eval2021

def parse_args():
    parser = argparse.ArgumentParser(description='VAR evaluation')
    parser.add_argument('submission', type=str, help='path to prediction file (json)')
    parser.add_argument('--separate', action='store_true', default=False,
                        help='evaluate premise and hypothesis separately')
    parser.add_argument('--verbose', action='store_false', default=True,
                        help='verbose logging')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    score = eval2021(args)

    if not args.separate:
        print("[All] METEOR {m:.2f} Bleu@4 {b:.2f} CIDEr {c:.2f} ROUGE_L {r:.2f} BERT_S {s:.2f}"
                                .format(m=score["METEOR"][0]*100,
                                        b=score["Bleu_4"][0]*100,
                                        c=score["CIDEr"][0]*100,
                                        r=score["ROUGE_L"][0]*100,
                                        s=score["BertScore"][0]*100))
    else:
        print("[Separate Observed] METEOR {m:.2f} Bleu@4 {b:.2f} CIDEr {c:.2f} ROUGE_L {r:.2f} BERT_S {s:.2f}"
                                .format(m=score['observed']["METEOR"][0]*100,
                                        b=score['observed']["Bleu_4"][0]*100,
                                        c=score['observed']["CIDEr"][0]*100,
                                        r=score['observed']["ROUGE_L"][0]*100,
                                        s=score['observed']["BertScore"][0]*100))
        print("[Separate Hypothesis] METEOR {m:.2f} Bleu@4 {b:.2f} CIDEr {c:.2f} ROUGE_L {r:.2f} BERT_S {s:.2f}"
                                .format(m=score['hypothesis']["METEOR"][0]*100,
                                        b=score['hypothesis']["Bleu_4"][0]*100,
                                        c=score['hypothesis']["CIDEr"][0]*100,
                                        r=score['hypothesis']["ROUGE_L"][0]*100,
                                        s=score['hypothesis']["BertScore"][0]*100))
