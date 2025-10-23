#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# TODO: This file need to be refactored.

import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .base import QuantizedToken, TokenizerModuleMixin
from .quantize import NormEMAVectorQuantizer, VectorQuantizer
from .vq_modules import Decoder, Encoder

logger = logging.getLogger(__name__)


class VQModel(nn.Module):
    def __init__(
        self,
        ddconfig,
        n_embed,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=True,  # tell vector quantizer to return indices as bhw
    ):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.image_key = image_key
        self.ddconfig = ddconfig
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(
            n_embed,
            embed_dim,
            beta=0.25,
            remap=remap,
            sane_index_shape=sane_index_shape,
        )
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(
            embed_dim, ddconfig["z_channels"], 1
        )
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer(
                "colorize", torch.randn(3, colorize_nlabels, 1, 1)
            )
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        load_res = self.load_state_dict(sd, strict=False)
        if load_res.missing_keys:
            logger.warning(
                f"Missing keys: {load_res.missing_keys} ! Is it intended ?"
            )
        print(f"Restored from {path}. ")

    def encode(self, x):
        # x : B, C, H, W, value in [-1, 1]
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, indices = self.quantize(h)
        return quant, emb_loss, indices

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embedding(code_b)
        quant_b = rearrange(quant_b, "b h w c-> b c h w")
        dec = self.decode(quant_b)
        return dec

    def indices_to_quant(self, indices):
        quant = self.quantize.embedding(indices)
        quant = rearrange(quant, "b h w c -> b c h w")
        return quant

    def decode_from_idx(self, idx, device="cuda"):
        h = idx.shape[-2]
        w = idx.shape[-1]
        idx = torch.from_numpy(idx).to(device)
        quant = self.indices_to_quant(idx)
        dec = self.decode(quant)
        return self.tensor_to_np(dec)

    def tensor_to_np(self, tensor):
        tensor = (tensor + 1.0) / 2.0
        tensor = tensor * 255.0
        tensor = torch.clamp(tensor, min=0, max=255)
        np_img = tensor.cpu().detach().numpy().astype(np.uint8)
        if len(np_img.shape) == 4:
            np_img = np_img[0]
        # (C, H, W)->(H, W, C)
        np_img = np.transpose(np_img, (1, 2, 0))
        return np_img


class NormVQModel(VQModel):
    def __init__(self, *args, **kwargs):
        ckpt_path = kwargs["ckpt_path"]
        kwargs["ckpt_path"] = None
        stride = kwargs.pop("stride", 3)
        padding = kwargs.pop("padding", 1)
        super().__init__(*args, **kwargs)
        self.quantize = NormEMAVectorQuantizer(
            n_embed=self.n_embed,
            embedding_dim=self.embed_dim,
            beta=1.0,
            kmeans_init=True,
            decay=0.99,
        )
        self.post_quant_conv = torch.nn.Conv2d(
            self.embed_dim,
            self.ddconfig["z_channels"],
            stride,
            padding=padding,
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=[])




normvq_dim16_res512_f16 = dict(
    n_embed=8192,
    embed_dim=16,
    ddconfig=dict(
        double_z=False,
        z_channels=256,
        resolution=512,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 1, 2, 2, 4],
        num_res_blocks=2,
        attn_resolutions=[32],
        dropout=0.0,
    ),
    ckpt_path="data/image_decoder.pt",  # noqa
)


def get_normvq_dim16_res512_f16(
    device: str = "cuda", ckpt=None
) -> NormVQModel:
    config = deepcopy(normvq_dim16_res512_f16)
    if ckpt is not None:
        config["ckpt_path"] = ckpt
    return NormVQModel(**config).to(device)


def get_map_normvq_dim16_res256_f8(
    device: str = "cuda", ckpt=None
) -> NormVQModel:
    config = dict(
        n_embed=8192,
        embed_dim=16,
        ddconfig=dict(
            double_z=False,
            z_channels=16,
            resolution=256,
            in_channels=5,
            out_ch=5,
            ch=128,
            ch_mult=[1, 2, 2, 4],
            num_res_blocks=2,
            attn_resolutions=[16],
            dropout=0.0,
        ),
        stride=1,
        padding=0,
        ckpt_path="data/weights/map_vq",
    )
    if ckpt is not None:
        config["ckpt_path"] = ckpt
    return NormVQModel(**config).to(device)
