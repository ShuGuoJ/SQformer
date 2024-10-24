import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import pywt

Pad_Mode = ['constant', 'reflect', 'replicate', 'circular']

def default_conv2d(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def default_conv3d(in_channels, out_channels,  kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride, 
        padding, bias=bias)        

class ResBlock(nn.Module):
    def __init__(self, conv2d, wn,  n_feats, kernel_size=3, bias=True, bn=False, act=nn.ReLU(inplace=True)):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(wn(conv2d(n_feats, n_feats, kernel_size, bias=bias)))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = x
        x = self.body(x)
        x = torch.add(x, res)
        return x   

class threeUnit(nn.Module):
    def __init__(self, conv3d, wn, n_feats, bias=True, bn=False, act=nn.ReLU(inplace=True)):
        super(threeUnit, self).__init__()    

        self.spatial = wn(conv3d(n_feats, n_feats, kernel_size=(1,3,3), padding=(0,1,1), bias=bias))
        self.spectral = wn(conv3d(n_feats, n_feats, kernel_size=(3,1,1), padding=(1,0,0), bias=bias))

        self.spatial_one = wn(conv3d(n_feats, n_feats, kernel_size=(1,3,3), padding=(0,1,1), bias=bias))
        self.relu = act
                 
    def forward(self, x):
        out = self.spatial(x) + self.spectral(x)
        out = self.relu(out)
        out = self.spatial_one(out)
                                
        return out




class Upsampler(nn.Sequential):
    def __init__(self, conv2d, wn, scale, n_feats, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(wn(conv2d(n_feats, 4 * n_feats, 3, bias)))
                m.append(nn.PixelShuffle(2))

                if act == 'relu':
                    m.append(nn.ReLU(inplace=True))

        elif scale == 3:
            m.append(wn(conv2d(n_feats, 9 * n_feats, 3, bias)))
            m.append(nn.PixelShuffle(3))

            if act == 'relu':
                m.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class DWT_3D(nn.Module):
    def __init__(self, pad_type = 'replicate', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups=None,
                 kernel_size = None, trainable = False):
        """
           
        """
        super(DWT_3D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = (1,2,2)
        # assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_lll = self.filt_low[1, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_llh = self.filt_low[1, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_lhl = self.filt_low[1, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_lhh = self.filt_low[1, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_hll = self.filt_high[1, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_hlh = self.filt_high[1, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_hhl = self.filt_high[1, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_hhh = self.filt_high[1, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_lll = self.filter_lll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_llh = self.filter_llh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhl = self.filter_lhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhh = self.filter_lhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hll = self.filter_hll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hlh = self.filter_hlh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhl = self.filter_hhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhh = self.filter_hhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        if torch.cuda.is_available():
            self.filter_lll = nn.Parameter(self.filter_lll).cuda()
            self.filter_llh = nn.Parameter(self.filter_llh).cuda()
            self.filter_lhl = nn.Parameter(self.filter_lhl).cuda()
            self.filter_lhh = nn.Parameter(self.filter_lhh).cuda()
            self.filter_hll = nn.Parameter(self.filter_hll).cuda()
            self.filter_hlh = nn.Parameter(self.filter_hlh).cuda()
            self.filter_hhl = nn.Parameter(self.filter_hhl).cuda()
            self.filter_hhh = nn.Parameter(self.filter_hhh).cuda()
        if self.trainable:
            self.filter_lll = nn.Parameter(self.filter_lll)
            self.filter_llh = nn.Parameter(self.filter_llh)
            self.filter_lhl = nn.Parameter(self.filter_lhl)
            self.filter_lhh = nn.Parameter(self.filter_lhh)
            self.filter_hll = nn.Parameter(self.filter_hll)
            self.filter_hlh = nn.Parameter(self.filter_hlh)
            self.filter_hhl = nn.Parameter(self.filter_hhl)
            self.filter_hhh = nn.Parameter(self.filter_hhh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2,
                              self.kernel_size // 2, self.kernel_size // 2,
                              self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 5
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad = self.pad_sizes, mode = self.pad_type)
        return F.conv3d(input, self.filter_lll, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_llh, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_lhl, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_lhh, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hll, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hlh, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hhl, stride = self.stride, groups = self.groups), \
               F.conv3d(input, self.filter_hhh, stride = self.stride, groups = self.groups)


class IDWT_3D(nn.Module):
    def __init__(self, pad_type = 'replicate', wavename = 'haar',
                 stride = 2, in_channels = 1, out_channels = None, groups=None,
                 kernel_size = None, trainable=False):
        """
            参照 DWT_1D 中的说明
            理论上，使用简单上采样和卷积实现的 IDWT 要比矩阵法计算量小、速度快，
            然而由于 Pytorch 中没有实现简单上采样，在实现 IDWT 只能用与 [[[1,0],[0,0]], [[0,0],[0,0]]] 做反卷积 Deconvolution 来实现简单上采样
            这使得该方法比矩阵法实现 IDWT 速度慢非常多。
        """
        super(IDWT_3D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None, '若训练过程中不更新滤波器组，请将 kernel_size 设置为默认值 None'
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = 2
        self.stride_ = (1,2,2)
        # assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.dec_lo)
        band_high = torch.tensor(wavelet.dec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band, '参数 kernel_size 的取值不能小于 初始化所用小波的滤波器长度'
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_lll = self.filt_low[:, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_llh = self.filt_low[:, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_lhl = self.filt_low[:, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_lhh = self.filt_low[:, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_hll = self.filt_high[:, None, None] * self.filt_low[None, :, None] * self.filt_low[None, None, :]
        self.filter_hlh = self.filt_high[:, None, None] * self.filt_low[None, :, None] * self.filt_high[None, None, :]
        self.filter_hhl = self.filt_high[:, None, None] * self.filt_high[None, :, None] * self.filt_low[None, None, :]
        self.filter_hhh = self.filt_high[:, None, None] * self.filt_high[None, :, None] * self.filt_high[None, None, :]
        self.filter_lll = self.filter_lll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_llh = self.filter_llh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhl = self.filter_lhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_lhh = self.filter_lhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hll = self.filter_hll[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hlh = self.filter_hlh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhl = self.filter_hhl[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        self.filter_hhh = self.filter_hhh[None, None, :, :, :].repeat((self.out_channels, self.in_channels // self.groups, 1, 1, 1))
        up_filter = torch.zeros((self.stride))
        up_filter[0] = 1
        up_filter = up_filter[:, None, None] * up_filter[None,:, None] * up_filter[None, None, :]
        up_filter = up_filter[None, None, :, :, :].repeat(self.out_channels, 1, 1, 1, 1)
        self.register_buffer('up_filter', up_filter)
        if torch.cuda.is_available():
            self.filter_lll = nn.Parameter(self.filter_lll).cuda()
            self.filter_llh = nn.Parameter(self.filter_llh).cuda()
            self.filter_lhl = nn.Parameter(self.filter_lhl).cuda()
            self.filter_lhh = nn.Parameter(self.filter_lhh).cuda()
            self.filter_hll = nn.Parameter(self.filter_hll).cuda()
            self.filter_hlh = nn.Parameter(self.filter_hlh).cuda()
            self.filter_hhl = nn.Parameter(self.filter_hhl).cuda()
            self.filter_hhh = nn.Parameter(self.filter_hhh).cuda()
            self.up_filter = self.up_filter.cuda()
        if self.trainable:
            self.filter_lll = nn.Parameter(self.filter_lll)
            self.filter_llh = nn.Parameter(self.filter_llh)
            self.filter_lhl = nn.Parameter(self.filter_lhl)
            self.filter_lhh = nn.Parameter(self.filter_lhh)
            self.filter_hll = nn.Parameter(self.filter_hll)
            self.filter_hlh = nn.Parameter(self.filter_hlh)
            self.filter_hhl = nn.Parameter(self.filter_hhl)
            self.filter_hhh = nn.Parameter(self.filter_hhh)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 0, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 0, self.kernel_size // 2 - 1,
                              self.kernel_size // 2 - 0, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 + 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 + 1,
                              self.kernel_size // 2 - 1, self.kernel_size // 2 + 1]

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        assert len(LLL.size()) == len(LLH.size()) == len(LHL.size()) == len(LHH.size()) == len(HLL.size()) == len(HLH.size()) == len(HHL.size()) == len(HHH.size()) == 5
        assert LLL.size()[0] == LLH.size()[0] == LHL.size()[0] == LHH.size()[0] == HLL.size()[0] == HLH.size()[0] == HHL.size()[0] == HHH.size()[0]
        assert LLL.size()[1] == LLH.size()[1] == LHL.size()[1] == LHH.size()[1] == HLL.size()[1] == HLH.size()[1] == HHL.size()[1] == HHH.size()[1] == self.in_channels
        LLL = F.pad(F.conv_transpose3d(LLL, self.up_filter, stride = self.stride_, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        LLH = F.pad(F.conv_transpose3d(LLH, self.up_filter, stride = self.stride_, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        LHL = F.pad(F.conv_transpose3d(LHL, self.up_filter, stride = self.stride_, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        LHH = F.pad(F.conv_transpose3d(LHH, self.up_filter, stride = self.stride_, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HLL = F.pad(F.conv_transpose3d(HLL, self.up_filter, stride = self.stride_, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HLH = F.pad(F.conv_transpose3d(HLH, self.up_filter, stride = self.stride_, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HHL = F.pad(F.conv_transpose3d(HHL, self.up_filter, stride = self.stride_, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        HHH = F.pad(F.conv_transpose3d(HHH, self.up_filter, stride = self.stride_, groups = self.in_channels), pad = self.pad_sizes, mode = self.pad_type)
        return F.conv3d(LLL, self.filter_lll, stride = 1, groups = self.groups) + \
               F.conv3d(LLH, self.filter_llh, stride = 1, groups = self.groups) + \
               F.conv3d(LHL, self.filter_lhl, stride = 1, groups = self.groups) + \
               F.conv3d(LHH, self.filter_lhh, stride = 1, groups = self.groups) + \
               F.conv3d(HLL, self.filter_hll, stride = 1, groups = self.groups) + \
               F.conv3d(HLH, self.filter_hlh, stride = 1, groups = self.groups) + \
               F.conv3d(HHL, self.filter_hhl, stride = 1, groups = self.groups) + \
               F.conv3d(HHH, self.filter_hhh, stride = 1, groups = self.groups)
