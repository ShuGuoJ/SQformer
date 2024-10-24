# SQformer

This repository contains the official implementation for SQformer introduced in the following paper:

[**SQformer: Spectral-Query transformer for hyperspectral image arbitrary-scale super-resolution**](https://arxiv.org/abs/2012.09161)
<br>
[Shuguo Jiang](https://yinboc.github.io/), [Nanying Li](https://www.sifeiliu.net/), [Meng Xu](https://xiaolonw.github.io/), [Shuyu Zhang](https://xiaolonw.github.io/), [Sen Jia](https://xiaolonw.github.io/)
<br>
IEEE TGRS 2024

### Citation

If you find our work useful in your research, please cite:

```
@ARTICLE{10684259,
  author={Jiang, Shuguo and Li, Nanying and Xu, Meng and Zhang, Shuyu and Jia, Sen},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={SQformer: Spectral-Query Transformer for Hyperspectral Image Arbitrary-Scale Super-Resolution}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
```

## Requirements
* See `requirements.txt`

## Data Preparation
### GF5
* Structure the data in the following format:
```
├──GF5
│  ├──Train_Spec
│  │  ├──xxx.mat
│  │  ...
│  ├──Train_Spec_LR_bicubic_X2
│  │  ├──xxx.mat
│  │  ...
│  ├──Train_Spec_LR_bicubic_X3
│  │  ├──xxx.mat
│  │  ...
│  ├──Train_Spec_LR_bicubic_X4
│  │  ├──xxx.mat
│  │  ...
│  ├──Valid_Spec
│  │  ├──xxx.mat
│  │  ...
│  ├──Valid_Spec_LR_bicubic_X2
│  │  ├──xxx.mat
│  │  ...
│  ├──Valid_Spec_LR_bicubic_X3
│  │  ├──xxx.mat
│  │  ...
│  ├──Valid_Spec_LR_bicubic_X4
│  │  ├──xxx.mat
│  │  ...
```

### Chikusei
* Structure the data in the following format:
```
├──Chikusei
│  ├──train_HR
│  │  ├──xxx.mat
│  │  ...
│  ├──valid_HR
│  │  ├──xxx.mat
│  │  ...
```

## Training
Before training, set the value of `os.environ[os.MY_DATASETS]` on line 17 of `train.py` to your root path of `GF5` and `Chikusei`:
```
os.environ['MY_DATASETS'] = [ROOT_PATH]
```
Then run the below command to train `SQformer` on two GPUs. Taking `GF5` as an example:
```
python train.py --config configs/train-GF5/[CONFIG_FILE] --name [EXP_NAME] --gpu 0,1
```
For other experiments, simply replace `[CONFIG_FILE]` by other configs (see `./configs`).

## Testing
Before testing, set the value of `os.environ[os.MY_DATASETS]` on line 10 of `test.py` to your root path of `GF5` and `Chikusei`:
```
os.environ['MY_DATASETS'] = [ROOT_PATH]
```
Then run the below command to get the results on the test set:
```
python test.py --config configs/test/[CONFIG_FILE] --model [MODEL_PATH] --gpu 0
```
