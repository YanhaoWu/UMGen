#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from torch import Tensor


@dataclass
class QuantizedToken:
    """A dataclass for quantized token.

    Args:
        quantized_embedding (Optional[Tensor]): The quantized embedding.
        encoding_indices (Optional[Tensor]): The encoding indices.
        quantization_loss (Optional[Tensor]): The quantization loss.

    """

    quantized_embedding: Optional[Tensor] = None
    encoding_indices: Optional[Tensor] = None
    quantization_loss: Optional[Tensor] = None


class TokenizerModuleMixin:
    """A mixin class for tokenizer module.

    The tokenizer module is used to encode and decode the input tensor.

    """

    def encode(self, x: Tensor) -> QuantizedToken:
        """Encode the input tensor."""
        raise NotImplementedError

    def decode(
        self,
        quantized_embedding: Optional[Tensor] = None,
        encoding_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Decode the quantized embedding or the encoding indices."""
        raise NotImplementedError
