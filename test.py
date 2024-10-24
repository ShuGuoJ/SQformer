import argparse
import os
import math
parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--model')
parser.add_argument('--gpu', default='0')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['MY_DATASETS'] = '/remote-home/share/data'
from functools import partial
import h5py
from skimage import __version__
if __version__ <= "0.15.0":
    from skimage.measure import compare_ssim
    compare_ssim = partial(compare_ssim, multichannel=True)
else:
    from skimage.metrics import structural_similarity as compare_ssim
    compare_ssim = partial(compare_ssim, channel_axis=2)
import xlwt
import time

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

import datasets
import models
import utils


if __name__ == '__main__':
    n_gpus = len(args.gpu.split(","))

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    def add_root(cfg):
        if cfg['dataset']['name'] == 'hsi-folder':
            path = cfg['dataset']['args']['root_path']
            cfg['dataset']['args']['root_path'] = os.path.join(os.getenv('MY_DATASETS', 'datasets'), path)
        elif cfg['dataset']['name'] == 'paired-hsi-folders':
            path_1 = cfg['dataset']['args']['root_path_1']
            cfg['dataset']['args']['root_path_1'] = os.path.join(os.getenv('MY_DATASETS', 'datasets'), path_1)
            path_2 = cfg['dataset']['args']['root_path_2']
            cfg['dataset']['args']['root_path_2'] = os.path.join(os.getenv('MY_DATASETS', 'datasets'), path_2)

    add_root(config['test_dataset'])

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=2*n_gpus, pin_memory=True)
    # loader = DataLoader(dataset, batch_size=spec['batch_size'], pin_memory=True)

    try:
        save_root = os.path.dirname(args.model) + "/" + spec["dataset"]["args"]["root_path_1"].split("/")[-1]
    except KeyError:
        save_root = os.path.dirname(args.model) + "/" + f'{spec["dataset"]["args"]["root_path"].split("/")[-1]}_LR_bicubic_X{dataset.scale_min}'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res, ssim = utils.eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True,
        save_root=save_root)
    print('result: {:.4f}, ssim: {:.4f}'.format(res, ssim))
