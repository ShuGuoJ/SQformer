import os
import time
import shutil
import math
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
from tqdm import tqdm

import torch
import numpy as np
from torch.optim import SGD, Adam, AdamW
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torchvision import transforms
import random
from thop import profile
from thop import clever_format
import time


class Regularizer():
    def __init__(self):
        super().__init__()
        self.output = []

    def forward_hook(self, module, fea_in, fea_out):
        self.output.append(fea_out)

    def __call__(self, *args, **kwargs):
        l2_loss = 0.
        for x in self.output:
            l2_loss = l2_loss + 0.5 * torch.mean(torch.pow(x, 2)).cpu()
        self.output.clear()
        return l2_loss


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W) or (1, D, H, W)
    """
    if img.ndim == 3:
        coord = make_coord(img.shape[-2:])
        rgb = img.view(img.shape[0], -1).permute(1, 0)
    elif img.ndim == 4:
        coord = make_coord(img.shape[-3:])
        rgb = img.view(img.shape[0], -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def calc_sam(x1, x2):
    sim = F.cosine_similarity(x1, x2, dim=1)
    return torch.acos(sim)


def get_sincos_pos_embed(coord, n_dim):
    # coord: [*, n_spa] -> [*, n_dim]
    n_spa = coord.shape[-1]
    assert n_dim % n_spa == 0
    d_model = n_dim // n_spa
    indices = torch.arange(d_model, device=coord.device).to(torch.float32)
    pre_shape = coord.shape[:-1]
    pos_set = []
    for i in range(n_spa):
        tmp = torch.ones(pre_shape + (d_model,), dtype=torch.float, device=coord.device)
        tmp = 1000**(tmp * torch.div(indices, 2, rounding_mode='floor') / d_model / 2)
        tmp = coord[..., i].unsqueeze(-1) / tmp
        torch.sin_(tmp[..., ::2])
        torch.cos_(tmp[..., 1::2])
        pos_set.append(tmp)
    pos = torch.cat(pos_set, dim=-1)
    # pos = pos.to(coord.device)
    return pos


def get_customized_collate_fn(scale_max, scale_min=1):

    def customized_collate_fn(batch_list):
        inp_dict = {}
        for sample in batch_list:
            for k, v in sample.items():
                if inp_dict.get(k) is None:
                    inp_dict[k] = [v]
                else:
                    inp_dict[k].append(v)
        for k, v in inp_dict.items():
            inp_dict[k] = torch.stack(v, dim=0)
        scale = random.uniform(scale_min, scale_max)
        # shape = inp_dict['inp'].shape
        # tmp = F.interpolate(inp_dict['inp'].permute(0, 1, 3, 4, 2).reshape(shape[0], -1, shape[2]), round(shape[2]/scale), mode='linear')
        # inp_dict['inp'] = tmp.reshape(shape[0], 1, shape[3], shape[4], -1).permute(0, 1, 4, 2, 3)
        shape = inp_dict['inp'].shape[-3:]
        shape[0] = round(shape[0]/scale)
        inp_dict['inp'] = F.interpolate(inp_dict['inp'], shape, mode='trilinear')
        return inp_dict

    return customized_collate_fn


def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        # feat = model.feat.clone()
        # feat_norm = feat.norm(dim=1)
        # feat_norm = feat_norm.cpu().numpy()
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            preds.append(pred)
            ql = qr

            # calculate flops
            # flops, params = profile(model, inputs=(inp, coord[:, ql: qr, :], cell[:, ql: qr, :]))
            # # flops, params = profile(model.query_rgb, inputs=(coord[:, ql: qr, :], cell[:, ql: qr, :]))
            # flops_, params_ = clever_format([flops, params], '%.6f')
            # print(f'flops: {flops_}, params: {params_}')
            # exit(0)

            # calculate time
            # begin = time.time()
            # # pred = model.query_rgb(coord[:, ql: qr, :], cell[:, ql: qr, :])
            # pred = model(inp, coord[:, ql: qr, :], cell[:, ql: qr, :])
            # end = time.time()
            # print(f'runing time: {end-begin}')
            # time.sleep(10)
            # exit(0)

        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, save_root=None):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_psnr = Averager()
    val_ssim = Averager()

    if save_root:
        wb = xlwt.Workbook()
        sh = wb.add_sheet("mySheet")
        sh.write(0, 0, label="file_name")
        sh.write(0, 1, label="psnr")
        sh.write(0, 2, label="ssim")

    prefetcher = data_prefetcher(loader)
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    batch = prefetcher.next()

    i = 0
    while batch is not None:
        inp = (batch['inp'] - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, batch['coord'], batch['cell'])
        else:
            pred = batched_predict(model, inp,
                                   batch['coord'], batch['cell'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None:  # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['gt'])
        val_psnr.add(res.item(), inp.shape[0])

        if save_root:
            if eval_type is not None:
                ssim = compare_ssim(batch["gt"][0].permute(1, 2, 0).cpu().numpy(),
                                    pred[0].permute(1, 2, 0).cpu().numpy())  # assume b is 1
                val_ssim.add(ssim, inp.shape[0])
                basename = os.path.basename(loader.dataset.dataset.dataset_1.files[i])
                save_path = f"{save_root}/{basename}"
                rgb_img = transforms.ToPILImage()(pred[0].cpu())
                rgb_img.save(save_path)
            else:
                try:
                    h = w = int(math.sqrt(pred.shape[1]))  # assume h is equal to w
                    ssim = compare_ssim(batch["gt"].reshape(h, w, -1).cpu().numpy(),
                                        pred.reshape(h, w, -1).cpu().numpy(), data_range=1)  # assume b is 1
                    val_ssim.add(ssim, inp.shape[0])
                    try:
                        basename = os.path.basename(loader.dataset.dataset.dataset_1.files[i])
                    except AttributeError:
                        basename = os.path.basename(loader.dataset.dataset.files[i])
                    save_path = f"{save_root}/{basename}"
                    with h5py.File(save_path, "w") as fw:
                        fw.create_dataset("cube",
                                          data=pred.reshape(h, w, -1).permute(2, 0, 1).cpu().numpy())  # assume b is 1
                except:
                    dep = inp.shape[-3]
                    h = w = int(math.sqrt(pred.shape[1] // dep))
                    ssim = compare_ssim(batch["gt"].reshape(dep, h, w).permute(1, 2, 0).cpu().numpy(),
                                        pred.reshape(dep, h, w).permute(1, 2, 0).cpu().numpy())  # assume b is 1
                    val_ssim.add(ssim, inp.shape[0])
                    basename = os.path.basename(loader.dataset.dataset.dataset_1.files[i])
                    save_path = f"{save_root}/{basename}"
                    with h5py.File(save_path, "w") as fw:
                        fw.create_dataset("cube",
                                          data=pred.reshape(dep, h, w).cpu().numpy())  # assume b is 1
                finally:
                    pass

            sh.write(i + 1, 0, label=basename)
            sh.write(i + 1, 1, label=res.item())
            sh.write(i + 1, 2, label=float(ssim))

        if verbose:
            if save_root:
                pbar.set_description('psnr {:.4f}, ssim {:.4f}'.format(val_psnr.item(), val_ssim.item()))
            else:
                pbar.set_description('psnr {:.4f}'.format(val_psnr.item()))

        batch = prefetcher.next()
        pbar.update(1)
        i += 1


    if save_root:
        wb.save(f"{save_root}/metric.xls")
        return val_psnr.item(), val_ssim.item()
    else:
        return val_psnr.item()


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return

        with torch.cuda.stream(self.stream):
            for key in self.next_data.keys():
                self.next_data[key] = self.next_data[key].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        if data is not None:
            for key in data.keys():
                data[key].record_stream(torch.cuda.current_stream())
        self.preload()
        return data



