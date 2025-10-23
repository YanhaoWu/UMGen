import logging
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_func
from torch.nn import functional as F
import deepspeed
logger = logging.getLogger(__name__)



def get_norm(norm_type, dim, **kwargs):
    if norm_type == "layer_norm":
        return LayerNorm(dim, **kwargs)
    elif norm_type == "rms_norm":
        return RMSNorm(dim, **kwargs)
    else:
        raise ValueError(f"norm type {norm_type} not supported")


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(
            input, self.weight.shape, self.weight, self.bias, 1e-5
        ).to(input.dtype)


class RMSNorm(torch.nn.Module):
    # from https://github.com/meta-llama/llama/blob/main/llama/model.py#L34
    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config, causal=True, block_size=1024):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=not config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=not config.bias
        )
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.causal = causal
        # causal mask to ensure that attention is only applied to the left
        # in the input sequence
        if self.causal:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, kvcache=None, mask_index=None):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and
        # move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        if isinstance(kvcache, list) and kvcache[0].size(0) > 0:
            prev_k, prev_v = kvcache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)

        new_kvcache = [k, v]
        curr_T = k.shape[1]

        k = k.view(B, curr_T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, curr_T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention;
        # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if kvcache:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                att = att.masked_fill(
                    torch.ones_like(self.bias[:, :, :T, :curr_T]) == 0,
                    float("-inf"),
                )
            elif mask_index is not None:
                att = att.masked_fill(
                    mask_index[:, None, None, :curr_T] == 0, float("-inf")
                )

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                att = att.masked_fill(
                    self.bias[:, :, :T, :T] == 0, float("-inf")
                )
            elif mask_index is not None:
                att = att.masked_fill(
                    mask_index[:, None, None, :curr_T] == 0, float("-inf")
                )

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kvcache


class CausalFlashAttention(nn.Module):
    def __init__(self, config, causal=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=not config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=not config.bias
        )
        # regularization
        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "scale", torch.tensor(1.0 / math.sqrt(self.n_embd / self.n_head))
        )
        self.causal = causal

    def forward(self, x, kvcache=None):
        # print("Using CaualFlashAttention")
        B, T, C = x.size()  # b, n, c
        # calculate query, key, values for all heads in batch and
        # move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # b, n, c
        if isinstance(kvcache, list) and kvcache[0].size(0) > 0:
            prev_k, prev_v = kvcache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)

        new_kvcache = [k, v]
        curr_T = k.shape[1]
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, curr_T, self.n_head, C // self.n_head)
        v = v.view(B, curr_T, self.n_head, C // self.n_head)

        y = flash_attn_func(
            q=q,
            k=k,
            v=v,
            dropout_p=self.attn_dropout_p,
            softmax_scale=self.scale,
            causal=self.causal,
        )  # (batch_size, seqlen, nheads, headdim).
        # y = y.transpose(1, 2).contiguous().view(B, T, C) # b, t, c
        y = y.view(B, T, C).contiguous()
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kvcache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embd, 4 * config.n_embd, bias=config.bias
        )
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(
            4 * config.n_embd, config.n_embd, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, causal=True, norm_type="layer_norm"):
        super().__init__()
        norm_kwargs = (
            {"bias": config.bias} if norm_type == "layer_norm" else {}
        )
        self.ln_1 = get_norm(norm_type, config.n_embd, **norm_kwargs)
        if config.flash_attention:
            self.attn = CausalFlashAttention(config, causal=causal)
        else:
            self.attn = CausalSelfAttention(
                config, causal=causal, block_size=config.block_size
            )
        self.ln_2 = get_norm(norm_type, config.n_embd, **norm_kwargs)
        self.mlp = MLP(config)
        self.torch_cp_enabled = bool(
            int(os.environ.get("HAT_USE_CHECKPOINT", "0"))
        )

    def forward_func(self, x, kvcache=None):
        attn_out, cache_ele = self.attn(self.ln_1(x), kvcache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if kvcache is None:
            return x, None
        else:
            return x, cache_ele

    def forward(self, x, kvcache=None):
        if self.torch_cp_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.forward_func, x, kvcache, use_reentrant=False
            )
        elif deepspeed.checkpointing.is_configured() and self.training:
            return deepspeed.checkpointing.non_reentrant_checkpoint(
                self.forward_func, x, kvcache
            )
        else:
            return self.forward_func(x, kvcache)




class BlockTAR(nn.Module):
    def __init__(self, config, norm_type="layer_norm"):
        super().__init__()
        norm_kwargs = (
            {"bias": config.bias} if norm_type == "layer_norm" else {}
        )
        self.ln_1 = get_norm(norm_type, config.n_embd, **norm_kwargs)
        if config.flash_attention:
            self.spatial_attn_1 = CausalFlashAttention(config, causal=False)
        else:
            self.spatial_attn_1 = CausalSelfAttention(config, causal=False)
        self.ln_2 = get_norm(norm_type, config.n_embd, **norm_kwargs)
        self.mlp1 = MLP(config)

        self.ln_3 = get_norm(norm_type, config.n_embd, **norm_kwargs)
        if config.flash_attention:
            self.temporal_attn = CausalFlashAttention(config, causal=True)
        else:
            self.temporal_attn = CausalSelfAttention(
                config, causal=True, block_size=config.block_size
            )
        self.ln_4 = get_norm(norm_type, config.n_embd, **norm_kwargs)
        self.mlp2 = MLP(config)

        self.ln_5 = get_norm(norm_type, config.n_embd, **norm_kwargs)
        if config.flash_attention:
            self.spatial_attn_2 = CausalFlashAttention(config, causal=False)
        else:
            self.spatial_attn_2 = CausalSelfAttention(config, causal=False)
        self.ln_6 = get_norm(norm_type, config.n_embd, **norm_kwargs)
        self.mlp3 = MLP(config)

        self.torch_cp_enabled = bool(
            int(os.environ.get("HAT_USE_CHECKPOINT", "0"))
        )

    def forward_func(self, x, kvcache=None, mask_temporal_flag=False):
        # x: bs, t, s, c, where s = h * w
        B, T, S, C = x.shape[:]

        x = rearrange(x, "b t s c-> (b t) s c", b=B, t=T, s=S, c=C)
        attn_out, _ = self.spatial_attn_1(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp1(self.ln_2(x))

        if not mask_temporal_flag:
            x = rearrange(x, "(b t) s c-> (b s) t c", b=B, t=T, s=S, c=C)
            attn_out, cache_ele_t = self.temporal_attn(self.ln_3(x), kvcache)
            x = x + attn_out
            x = x + self.mlp2(self.ln_4(x))
        else:
            x = rearrange(x, "(b t) s c-> (b s) t c", b=B, t=T, s=S, c=C)

        x = rearrange(x, "(b s) t c-> (b t) s c", b=B, t=T, s=S, c=C)
        attn_out, _ = self.spatial_attn_2(self.ln_5(x))
        x = x + attn_out
        x = x + self.mlp3(self.ln_6(x))

        x = rearrange(x, "(b t) s c-> b t s c", b=B, t=T, s=S, c=C)

        if kvcache is None:
            return x, None
        else:
            return x, cache_ele_t

    def forward(self, x, kvcache=None, mask_temporal_flag=False):
        if self.torch_cp_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.forward_func,
                x,
                kvcache,
                mask_temporal_flag,
                use_reentrant=False,
            )
        elif deepspeed.checkpointing.is_configured() and self.training:
            return deepspeed.checkpointing.non_reentrant_checkpoint(
                self.forward_func, x, kvcache, mask_temporal_flag
            )
        else:
            return self.forward_func(x, kvcache, mask_temporal_flag)


class BlockOAR(nn.Module):
    def __init__(self, config, causal=True, norm_type="layer_norm"):
        super().__init__()
        norm_kwargs = (
            {"bias": config.bias} if norm_type == "layer_norm" else {}
        )
        self.ln_1 = get_norm(norm_type, config.n_embd, **norm_kwargs)

        if config.flash_attention:
            self.temporal_attn = CausalFlashAttention(config, causal=causal)
        else:
            self.temporal_attn = CausalSelfAttention(
                config, causal=causal, block_size=config.block_size
            )
            # self.temporal_attn = CausalSelfAttention(
            #     config, causal=causal, block_size=config.ar_context_size
            # )

        self.ln_2 = get_norm(norm_type, config.n_embd, **norm_kwargs)
        self.mlp = MLP(config)
        self.torch_cp_enabled = bool(
            int(os.environ.get("HAT_USE_CHECKPOINT", "0"))
        )

    def forward_func(self, x, kvcache=None):
        # x: bs, t, s, c, where s = h * w
        B, T, S, C = x.shape[:]

        # do temporal attention
        x = rearrange(x, "b t s c-> (b t) s c", b=B, t=T, s=S, c=C)
        attn_out, cache_ele = self.temporal_attn(self.ln_1(x), kvcache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        x = rearrange(x, "(b t) s c-> b t s c", b=B, t=T, s=S, c=C)

        if kvcache is None:
            return x, None
        else:
            return x, cache_ele

    def forward(self, x, kvcache=None):
        if self.torch_cp_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.forward_func, x, kvcache, use_reentrant=False
            )
        elif deepspeed.checkpointing.is_configured() and self.training:
            return deepspeed.checkpointing.non_reentrant_checkpoint(
                self.forward_func, x, kvcache
            )
        else:
            return self.forward_func(x, kvcache)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    assert (
        dim % 4
    ) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.unsqueeze(-1) * omega[None, None, :]
    x = x.unsqueeze(-1) * omega[None, None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)
    return pe.type(dtype)


def checkpoint_func(modules):
    def exec_func(x):
        for module in modules:
            x = module(x)
        return x

    return exec_func


class FlashCrossAttention(nn.Module):
    def __init__(self, config, causal=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.q_attn = nn.Linear(
            config.n_embd, config.n_embd, bias=not config.bias
        )
        self.k_attn = nn.Linear(
            config.n_embd, config.n_embd, bias=not config.bias
        )
        self.v_attn = nn.Linear(
            config.n_embd, config.n_embd, bias=not config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=not config.bias
        )
        # regularization
        self.attn_dropout_p = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "scale", torch.tensor(1.0 / math.sqrt(self.n_embd / self.n_head))
        )
        self.causal = causal

    def forward(self, q, p, kvcache=None):
        B, T, C = q.size()  # b, t, c
        # calculate query, key, values for all heads
        q = self.q_attn(q)
        k = self.k_attn(p)
        v = self.v_attn(p)
        if isinstance(kvcache, list) and kvcache[0].size(0) > 0:
            prev_k, prev_v = kvcache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)
        new_kvcache = [k, v]
        curr_T = k.shape[1]
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = k.view(B, curr_T, self.n_head, C // self.n_head)
        v = v.view(B, curr_T, self.n_head, C // self.n_head)
        y = flash_attn_func(
            q=q,
            k=k,
            v=v,
            dropout_p=self.attn_dropout_p,
            softmax_scale=self.scale,
            causal=self.causal,
        )  # (batch_size, seqlen, nheads, headdim).
        # y = y.transpose(1, 2).contiguous().view(B, T, C) # b, t, c
        y = y.view(B, T, C).contiguous()
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kvcache


class CrossAttention(nn.Module):
    def __init__(self, config, causal=False, in_channels=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        if in_channels is None:
            in_channels = config.n_embd
        self.in_channels = in_channels

        # self.project_posi_emb = nn.Linear(
        #     in_channels+ 3 * config.n_posiembed, config.n_embd, bias=not config.bias
        # )

        # key, query, value projections for all heads, but in a batch
        self.q_attn_wp = nn.Linear(
            in_channels, config.n_embd, bias=not config.bias
        )
        self.k_attn_wp = nn.Linear(
            in_channels, config.n_embd, bias=not config.bias
        )
        self.v_attn_wp = nn.Linear(
            in_channels, config.n_embd, bias=not config.bias
        )
        # output projection
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=not config.bias
        )
        # regularization
        self.attn_dropout_p = config.dropout

        self.resid_dropout = nn.Dropout(config.dropout)

        self.attn_dropout = nn.Dropout(
            config.dropout
        )  # 如果有mask_index，就不需要这个dropout

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "scale", torch.tensor(1.0 / math.sqrt(self.n_embd / self.n_head))
        )
        self.causal = causal

    def forward(self, q, p, kvcache=None, mask_index=None, relative_pe=None):
        B, T, C = q.size()  # b, t, c

        if relative_pe is not None and relative_pe.shape[-2] == 1:  # 全局位置编码
            q = torch.cat(
                (q, relative_pe[:, : q.shape[1], :, :].squeeze(2)), dim=-1
            )
            p = torch.cat((p, relative_pe.squeeze(2)), dim=-1)

        # calculate query, key, values for all heads
        # print(q.dtype)

        q = self.q_attn_wp(q)
        k = self.k_attn_wp(p)
        v = self.v_attn_wp(p)
        if isinstance(kvcache, list) and kvcache[0].size(0) > 0:
            prev_k, prev_v = kvcache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)
        new_kvcache = [k, v]
        curr_T = k.shape[1]
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T1, hs)
        k = k.view(B, curr_T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T2, hs)
        v = v.view(B, curr_T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T2, hs)

        if kvcache:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                att = att.masked_fill(
                    torch.ones_like(self.bias[:, :, :T, :curr_T]) == 0,
                    float("-inf"),
                )
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
            elif mask_index is not None:
                att = att.masked_fill(
                    mask_index[:, None, None, :curr_T] == 0, float("-inf")
                )

                att = F.softmax(att, dim=-1)

            y = att @ v  # (B, nh, T1, T2) x (B, nh, T2, hs) -> (B, nh, T1, hs)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                att = att.masked_fill(
                    self.bias[:, :, :T, :T] == 0, float("-inf")
                )
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)

            elif mask_index is not None:
                att = att.masked_fill(mask_index, float("-inf"))

                att = F.softmax(
                    att, dim=-1
                )  # (B, nh, T1, T2), 有mask的时候先就不用dropout了

            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kvcache



class Decoder(nn.Module):
    def __init__(self, config, causal=False):
        super().__init__()

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        if config.flash_attention:  #
            self.self_attn = CausalFlashAttention(config, causal=causal)
        else:
            self.self_attn = CausalSelfAttention(
                config, causal=causal, block_size=config.block_size
            )

        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)

        if config.flash_attention:
            self.cross_attn = FlashCrossAttention(config, causal=False)
        else:
            self.cross_attn = CrossAttention(
                config,
                causal=False,
                in_channels=config.n_embd + 3 * config.n_posiembed,
            )

        self.ln_4 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp1 = MLP(config)
        self.torch_cp_enabled = bool(
            int(os.environ.get("HAT_USE_CHECKPOINT", "0"))
        )

        self.config = config

    def forward_func(
        self, x, p, kvcache=None, mask_index=None, relative_pe=None
    ):
        # x: bs, t, s, c, where s = h * w
        B, T, S, C = x.shape[:]
        x = rearrange(x, "b t s c-> (b t) s c", b=B, t=T, s=S, c=C)

        attn_out, _ = self.self_attn(self.ln_1(x))
        x = x + attn_out

        p = rearrange(p, "b t s c-> (b t) s c")

        attn_out, cache_ele_t = self.cross_attn(self.ln_2(x), self.ln_3(p))

        x = x + attn_out
        x = x + self.mlp1(self.ln_4(x))
        x = rearrange(x, "(b t) s c-> b t s c", b=B, t=T, s=S, c=C)

        if kvcache is None:
            return x, None
        else:
            return x, cache_ele_t



    def forward(self, x, p, kvcache=None, mask_index=None, relative_pe=None):
        if self.torch_cp_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.forward_func,
                x,
                p,
                kvcache,
                mask_index,
                relative_pe,
                use_reentrant=False,
            )
        elif deepspeed.checkpointing.is_configured() and self.training:
            # return deepspeed.checkpointing.checkpoint(
            return deepspeed.checkpointing.non_reentrant_checkpoint(
                self.forward_func, x, p, kvcache, mask_index, relative_pe
            )
        else:
            return self.forward_func(
                x, p, kvcache, mask_index=mask_index, relative_pe=relative_pe
            )



class GMLP(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, bias=False, dropout=0.0):
        super().__init__()
        self.c_fc = nn.Linear(in_dim, mlp_dim, bias=bias)
        # 输出self.c_fc的数据类型
        # print("dtype of self.c_fc is ", self.c_fc.weight.dtype)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(mlp_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.torch_cp_enabled = bool(
            int(os.environ.get("HAT_USE_CHECKPOINT", "0"))
        )

    def forward_func(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        if self.torch_cp_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.forward_func,
                x,
                use_reentrant=False,
            )
        elif deepspeed.checkpointing.is_configured() and self.training:
            return deepspeed.checkpointing.non_reentrant_checkpoint(
                self.forward_func,
                x,
            )
        else:
            return self.forward_func(x)


def position_encoding_init(n_position, emb_dim, start_index=0):
    """Init the sinusoid position encoding table"""

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array(
        [
            [
                (pos + start_index) / np.power(10000, 2 * (j // 2) / emb_dim)
                for j in range(emb_dim)
            ]
            if pos != 0
            else np.zeros(emb_dim)
            for pos in range(n_position)
        ]
    )

    position_enc[1:, 0::2] = np.sin(
        position_enc[1:, 0::2]
    )  # apply sin on 0th,2nd,4th...emb_dim
    position_enc[1:, 1::2] = np.cos(
        position_enc[1:, 1::2]
    )  # apply cos on 1st,3rd,5th...emb_dim
    return torch.from_numpy(position_enc).type(torch.bfloat16)
