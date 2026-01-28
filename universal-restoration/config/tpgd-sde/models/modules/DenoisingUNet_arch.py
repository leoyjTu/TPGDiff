import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import functools

from .module_util import (
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock, Upsampler,
    LinearAttention, Attention,
    PreNorm, Residual)

from .attention import SpatialTransformer


class StructFiLMAdapter(nn.Module):
    def __init__(self, feat_dim: int, struct_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(struct_dim),
            nn.Linear(struct_dim, feat_dim * 2)
        )

    def forward(self, feat: torch.Tensor, struct_tokens: torch.Tensor):
        if struct_tokens is None:
            return feat

        B, C, H, W = feat.shape
        pooled = struct_tokens.mean(dim=1)          # [B, D]

        gamma_beta = self.mlp(pooled)               # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=1)    # [B, C], [B, C]

        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)

        return feat * (1.0 + gamma) + beta


class ConditionalUNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, ch_mult=[1, 2, 4, 4], 
                    context_dim=512, use_degra_context=True, use_image_context=False, upscale=1, use_struct_context: bool = False, struct_context_dim: int = 512):
        super().__init__()
        self.depth = len(ch_mult)
        self.upscale = upscale # not used
        self.context_dim = -1 if context_dim is None else context_dim
        self.use_image_context = use_image_context
        self.use_degra_context = use_degra_context

        self.use_struct_context = use_struct_context
        self.struct_context_dim = struct_context_dim if use_struct_context else -1

        num_head_channels = 32
        dim_head = num_head_channels

        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())

        self.init_conv = default_conv(in_nc*2, nf, 7)
        
        # time embeddings
        time_dim = nf * 4

        self.random_or_learned_sinusoidal_cond = False

        if self.random_or_learned_sinusoidal_cond:
            learned_sinusoidal_dim = 16
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, False)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(nf)
            fourier_dim = nf

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        if self.context_dim > 0 and use_degra_context: 
            self.prompt = nn.Parameter(torch.rand(1, time_dim))
            self.text_mlp = nn.Sequential(
                nn.Linear(context_dim, time_dim), NonLinearity(),
                nn.Linear(time_dim, time_dim))
            self.prompt_mlp = nn.Linear(time_dim, time_dim)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        ch_mult = [1] + ch_mult

        if self.use_struct_context and self.struct_context_dim > 0:
            self.struct_adapters_down = nn.ModuleList([])
            self.struct_adapters_up = nn.ModuleList([])
        else:
            self.struct_adapters_down = None
            self.struct_adapters_up = None

        for i in range(self.depth):
            dim_in = nf * ch_mult[i]
            dim_out = nf * ch_mult[i+1]

            num_heads_in = dim_in // num_head_channels
            num_heads_out = dim_out // num_head_channels
            dim_head_in = dim_in // num_heads_in

            if use_image_context and context_dim > 0:
                att_down = LinearAttention(dim_in) if i < 3 else SpatialTransformer(dim_in, num_heads_in, dim_head, depth=1, context_dim=context_dim)
                att_up = LinearAttention(dim_out) if i < 3 else SpatialTransformer(dim_out, num_heads_out, dim_head, depth=1, context_dim=context_dim)
            else:
                att_down = LinearAttention(dim_in) # if i < 2 else Attention(dim_in)
                att_up = LinearAttention(dim_out) # if i < 2 else Attention(dim_out)

            self.downs.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                block_class(dim_in=dim_in, dim_out=dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, att_down)),
                Downsample(dim_in, dim_out) if i != (self.depth-1) else default_conv(dim_in, dim_out)
            ]))

            if self.struct_adapters_down is not None:
                self.struct_adapters_down.append(
                    StructFiLMAdapter(dim_in, struct_dim=self.struct_context_dim)
                )

            self.ups.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, att_up)),
                Upsample(dim_out, dim_in) if i!=0 else default_conv(dim_out, dim_in)
            ]))

            if self.struct_adapters_up is not None:
                self.struct_adapters_up.insert(0,
                    StructFiLMAdapter(dim_out, struct_dim=self.struct_context_dim)
                )

        mid_dim = nf * ch_mult[-1]
        num_heads_mid = mid_dim // num_head_channels
        self.mid_block1 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)
        if use_image_context and context_dim > 0:
            self.mid_attn = Residual(PreNorm(mid_dim, SpatialTransformer(mid_dim, num_heads_mid, dim_head, depth=1, context_dim=context_dim)))
        else:
            self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = block_class(dim_in=mid_dim, dim_out=mid_dim, time_emb_dim=time_dim)

        self.final_res_block = block_class(dim_in=nf * 2, dim_out=nf, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(nf, out_nc, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, xt, cond, time, deg_context=None, content_context=None, struct_tokens=None):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(xt.device)
        
        x = xt - cond
        x = torch.cat([x, cond], dim=1)

        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)

        x = self.init_conv(x)
        x_ = x.clone()

        t = self.time_mlp(time) 
        if self.context_dim > 0:
            if self.use_degra_context and deg_context is not None:
                prompt_embedding = torch.softmax(self.text_mlp(deg_context), dim=1) * self.prompt
                prompt_embedding = self.prompt_mlp(prompt_embedding)
                t = t + prompt_embedding

            if self.use_image_context and content_context is not None:
                content_context = content_context.unsqueeze(1)

        h = []

        for idx, (b1, b2, attn, downsample) in enumerate(self.downs):
            x = b1(x, t)

            if (self.struct_adapters_down is not None and struct_tokens is not None and  idx < self.depth-1):
                x = self.struct_adapters_down[idx](x, struct_tokens)

            h.append(x)

            x = b2(x, t)
            x = attn(x, context=content_context)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x, context=content_context)
        x = self.mid_block2(x, t)

        for up_idx, (b1, b2, attn, upsample) in enumerate(self.ups):
            x = torch.cat([x, h.pop()], dim=1)
            x = b1(x, t)

            if (self.struct_adapters_up is not None and struct_tokens is not None and  up_idx >= 1):
                x = self.struct_adapters_up[up_idx](x, struct_tokens)

            x = torch.cat([x, h.pop()], dim=1)
            x = b2(x, t)

            x = attn(x, context=content_context)
            x = upsample(x)

        x = torch.cat([x, x_], dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = x[..., :H, :W].contiguous()
        
        return x



