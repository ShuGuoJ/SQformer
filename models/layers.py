import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from utils import get_sincos_pos_embed

from models import register


class Identity(nn.Identity):
    def forward(self, *args):
        return args[0]


class Mlp(nn.Module):
    def __init__(self, in_features, expandsion=2, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        hidden_features = in_features * expandsion
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim, bias=False) if num_heads > 1 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, q, d):
        # q: [b, n_q, c]
        # d: [b, n_q, n_d, c]
        # w: [b, n_q, n_d]
        B, N1, C = q.shape
        N2 = d.shape[2]
        q = self.proj_q(q).reshape(B*N1*self.num_heads, self.head_dim).unsqueeze(1)
        kv = self.proj_kv(d).reshape(B, N1, N2, self.num_heads, 2*self.head_dim).permute(0, 1, 3, 4, 2).reshape(B*N1*self.num_heads, 2*self.head_dim, N2)
        k, v = kv.chunk(2, dim=-2)
        attn = q @ (k * self.scale)
        attn = attn.softmax(dim=-1)
        v = v.permute(0, 2, 1)
        attn = self.attn_drop(attn)
        x = (attn @ v).reshape(B, N1, self.num_heads*self.head_dim)
        x = self.proj(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mask=False):
        super().__init__()
        self.hidden_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.mask = mask

        self.proj_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim, bias=False) if num_heads > 1 else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x):
        # q: [b, n_q, c]
        B, N, C = x.shape
        q = self.proj_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.proj_kv(x).reshape(B, N, self.num_heads, 2*self.head_dim).permute(0, 2, 3, 1)
        k, v = kv.chunk(2, dim=-2)
        attn = q @ (k * self.scale)
        if self.mask:
            mask_mat = torch.zeros_like(attn, device=attn.device).to(torch.bool)
            mask_mat[..., 1:, 0] = True
            attn.masked_fill_(mask_mat, float('-inf'))
        attn = attn.softmax(dim=-1)
        v = v.permute(0, 1, 3, 2)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, -1)
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, ndim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., expandsion=2, mlp_drop=0., norm=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(ndim) if norm else nn.Identity()
        self.attention = Attention(dim=ndim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ln2 = nn.LayerNorm(ndim) if norm else nn.Identity()
        self.mlp = Mlp(in_features=ndim, expandsion=expandsion, drop=mlp_drop)

    def forward(self, q, d):
        x = q
        x = self.ln1(x + self.attention(x, d))
        x = self.ln2(x + self.mlp(x))
        return x


class SelfBlock(nn.Module):
    def __init__(self, ndim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., expandsion=2, mlp_drop=0., norm=False, mask=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(ndim) if norm else nn.Identity()
        self.attention = SelfAttention(dim=ndim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, mask=mask)
        self.ln2 = nn.LayerNorm(ndim) if norm else nn.Identity()
        self.mlp = Mlp(in_features=ndim, expandsion=expandsion, drop=mlp_drop)

    def forward(self, x):
        x = self.ln1(x + self.attention(x))
        x = self.ln2(x + self.mlp(x))
        return x


@register('SpectralQuerier')
class SpectralQuerier(nn.Module):
    def __init__(self, hlayers, nlayers, in_dim, hid_dim, out_dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., expandsion=2, mlp_drop=0., norm=False):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.selfblocks = nn.ModuleList([])
        self.token_head_proj = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim, bias=False)
        )
        self.ln = nn.LayerNorm(hid_dim) if norm else nn.Identity()
        dt_head_proj = []
        for i in range(hlayers):
            if i == 0:
                dt_head_proj.append(nn.Linear(in_dim, hid_dim, bias=False))
            else:
                dt_head_proj.append(nn.Linear(hid_dim, hid_dim, bias=False))
            dt_head_proj.append(nn.ReLU())
        dt_head_proj.append(nn.Linear(hid_dim, hid_dim, bias=False))
        self.dt_head_proj = nn.Sequential(*dt_head_proj)
        for i in range(nlayers):
            self.blocks.append(Block(ndim=hid_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, expandsion=expandsion, mlp_drop=mlp_drop, norm=norm))
        for i in range(nlayers-1):
            self.selfblocks.append(SelfBlock(ndim=hid_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop, expandsion=expandsion, mlp_drop=mlp_drop, norm=norm))
        self.selfblocks.append(nn.Identity())
        self.tail_proj = nn.Linear(hid_dim, out_dim, bias=False)

    def forward(self, q, d):
        # q: [b, n_q, c]
        # d: [b, n_q, n_d, c]
        # w: [b, n_q, n_d]
        # B, N = d.shape[:2]
        x = self.token_head_proj(q)
        d = self.ln(self.dt_head_proj(d))
        for block1, block2 in zip(self.blocks, self.selfblocks):
            x = block1(x, d)
            x = block2(x)
        return self.tail_proj(x)
