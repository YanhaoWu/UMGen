#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from copy import deepcopy
from typing import Optional

from torch import Tensor

from .base import QuantizedToken, TokenizerModuleMixin
from .vq_model import NormVQModel, normvq_dim16_res512_f16


class NormVQModelTokenizer(NormVQModel, TokenizerModuleMixin):
    """A VQModel that inherits TokenizerModuleMixin interface.

    Note:
        User is recommended to use this class instead of NormVQModel directly
        because it is refactored to be more user-friendly.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, x: Tensor) -> QuantizedToken:
        """Encode the input tensor.

        The input tensor is a tensor image of shape (..., 3, H, W) with values
        in the range [-1, 1].

        Args:
            x (Tensor): The input tensor.

        Returns:
            QuantizedToken: The quantized token with quantized embedding,
                encoding indices, and quantization loss.
                encoding indices shape will be (..., h, w).
                quantized embedding shape will be (..., c, h, w)


        """
        quant, emb_loss, indices = NormVQModel.encode(self, x)
        return QuantizedToken(
            quantized_embedding=quant,
            encoding_indices=indices,
            quantization_loss=emb_loss,
        )

    def decode(
        self,
        quantized_embedding: Optional[Tensor] = None,
        encoding_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode the quantized tensor or the encoding indices.

        Args:
            quantized_embedding (Optional[Tensor]): The quantized
                embedding tensor.
            encoding_indices (Optional[Tensor]): The encoding indices.

        Returns:
            Tensor: The decoded tensor from the quantized embedding or
                the encoding indices.

        """
        if [quantized_embedding, encoding_indices].count(None) != 1:
            raise ValueError(
                "Only one of quantized_embedding and encoding_indices should be provided"
            )
        if quantized_embedding is not None:
            return NormVQModel.decode(self, quantized_embedding)
        else:
            assert encoding_indices is not None
            quant = self.indices_to_quant(encoding_indices)
            return NormVQModel.decode(self, quant)


def get_normvq_dim16_res512_f16(
    ckpt: Optional[str] = None,
) -> NormVQModelTokenizer:
    cfg = deepcopy(normvq_dim16_res512_f16)
    if ckpt is not None:
        cfg["ckpt_path"] = ckpt
    return NormVQModelTokenizer(**cfg)
