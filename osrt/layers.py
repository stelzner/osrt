import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

import math
from einops import rearrange


__USE_DEFAULT_INIT__ = False


class JaxLinear(nn.Linear):
    """ Linear layers with initialization matching the Jax defaults """
    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            input_size = self.weight.shape[-1]
            std = math.sqrt(1/input_size)
            init.trunc_normal_(self.weight, std=std, a=-2.*std, b=2.*std)
            if self.bias is not None:
                init.zeros_(self.bias)


class ViTLinear(nn.Linear):
    """ Initialization for linear layers used by ViT """
    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.normal_(self.bias, std=1e-6)


class SRTLinear(nn.Linear):
    """ Initialization for linear layers used in the SRT decoder """
    def reset_parameters(self):
        if __USE_DEFAULT_INIT__:
            super().reset_parameters()
        else:
            init.xavier_uniform_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves=8, start_octave=0):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave

    def forward(self, coords, rays=None):
        embed_fns = []
        batch_size, num_points, dim = coords.shape

        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves)
        octaves = octaves.float().to(coords)
        multipliers = 2**octaves * math.pi
        coords = coords.unsqueeze(-1)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        scaled_coords = coords * multipliers

        sines = torch.sin(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)

        result = torch.cat((sines, cosines), -1)
        return result


class RayEncoder(nn.Module):
    def __init__(self, pos_octaves=8, pos_start_octave=0, ray_octaves=4, ray_start_octave=0):
        super().__init__()
        self.pos_encoding = PositionalEncoding(num_octaves=pos_octaves, start_octave=pos_start_octave)
        self.ray_encoding = PositionalEncoding(num_octaves=ray_octaves, start_octave=ray_start_octave)

    def forward(self, pos, rays):
        if len(rays.shape) == 4:
            batchsize, height, width, dims = rays.shape
            pos_enc = self.pos_encoding(pos.unsqueeze(1))
            pos_enc = pos_enc.view(batchsize, pos_enc.shape[-1], 1, 1)
            pos_enc = pos_enc.repeat(1, 1, height, width)
            rays = rays.flatten(1, 2)

            ray_enc = self.ray_encoding(rays)
            ray_enc = ray_enc.view(batchsize, height, width, ray_enc.shape[-1])
            ray_enc = ray_enc.permute((0, 3, 1, 2))
            x = torch.cat((pos_enc, ray_enc), 1)
        else:
            pos_enc = self.pos_encoding(pos)
            ray_enc = self.ray_encoding(rays)
            x = torch.cat((pos_enc, ray_enc), -1)

        return x


# Transformer implementation based on ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            ViTLinear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ViTLinear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., selfatt=True, kv_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        if selfatt:
            self.to_qkv = JaxLinear(dim, inner_dim * 3, bias=False)
        else:
            self.to_q = JaxLinear(dim, inner_dim, bias=False)
            self.to_kv = JaxLinear(kv_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            JaxLinear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, z=None):
        if z is None:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim=-1)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., selfatt=True, kv_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout, selfatt=selfatt, kv_dim=kv_dim)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, z=None):
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x) + x
        return x



class SlotAttention(nn.Module):
    """
    Slot Attention as introduced by Locatello et al.
    """
    def __init__(self, num_slots, input_dim=768, slot_dim=1536, hidden_dim=3072, iters=3, eps=1e-8,
                 randomize_initial_slots=False):
        super().__init__()

        self.num_slots = num_slots
        self.iters = iters
        self.scale = slot_dim ** -0.5
        self.slot_dim = slot_dim

        self.randomize_initial_slots = randomize_initial_slots
        self.initial_slots = nn.Parameter(torch.randn(num_slots, slot_dim))

        self.eps = eps

        self.to_q = JaxLinear(slot_dim, slot_dim, bias=False)
        self.to_k = JaxLinear(input_dim, slot_dim, bias=False)
        self.to_v = JaxLinear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.mlp = nn.Sequential(
            JaxLinear(slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            JaxLinear(hidden_dim, slot_dim)
        )

        self.norm_input   = nn.LayerNorm(input_dim)
        self.norm_slots   = nn.LayerNorm(slot_dim)
        self.norm_pre_mlp = nn.LayerNorm(slot_dim)

    def forward(self, inputs):
        """
        Args:
            inputs: set-latent representation [batch_size, num_inputs, dim]
        """
        batch_size, num_inputs, dim = inputs.shape

        inputs = self.norm_input(inputs)
        if self.randomize_initial_slots:
            slot_means = self.initial_slots.unsqueeze(0).expand(batch_size, -1, -1)
            slots = torch.distributions.Normal(slot_means, self.embedding_stdev).rsample()
        else:
            slots = self.initial_slots.unsqueeze(0).expand(batch_size, -1, -1)

        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots
            norm_slots = self.norm_slots(slots)

            q = self.to_q(norm_slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            # shape: [batch_size, num_slots, num_inputs]
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(updates.flatten(0, 1), slots_prev.flatten(0, 1))
            slots = slots.reshape(batch_size, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_pre_mlp(slots))

        return slots


