# (Video) Feature Extraction Code for Visual Abductive Reasoning
Partially forked code from [anet2016-cuhk-feature](https://github.com/LuoweiZhou/anet2016-cuhk-feature) and [caffemodel2pytorch](https://github.com/vadimkantorov/caffemodel2pytorch). See also [here](https://github.com/yjxiong/anet2016-cuhk) (CUHK & ETH & SIAT Solution to ActivityNet Challenge 2016) for more details.

## Feature extraction
- RGB ResNet-200 feature
- Optical flow BN-Inception feature

## Brief instruction
1. Download pretrained models:
```
bash models/get_reference_models.sh
```
2. Extract feature through:
```
python extract_feature.py 
```
Note: See more configurations in `extract_feature.py `.