import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

class AttentionQKV(nn.Module):
    """
    Computes attention based on provided similarity metric.
    """

    def __init__(self):
        pass

    def forward(self, queries, keys, values, mask=None):
        """Fast scaled dot product attention.

            :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
            :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
            :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]
            :param mask:    Tensor with shape [batch_size, n_queries, n_queries]

            :return: output: Tensor with shape [batch_size, heads (optional), n_queries, depth_v]
        """
        pass


class MultiHeadProjection(nn.Module):

    def __init__(self, n_heads, feature_sizes):
        """Map the multi-headed attention across the map

        Arguments:
            n_heads {int} -- The number of heads in the attention map
            feature_sizes {int} -- The size of the feature dimensions for key, query, and value

        """

        pass

    def forward(self, inputs, mask=None):
        """Fast multi-head attention.

        :param queries: Tensor with shape [batch_size, n_queries, depth_k]
        :param keys:    Tensor with shape [batch_size, n_keyval, depth_k]
        :param values:  Tensor with shape [batch_size, n_keyval, depth_v]

        :return: output: Tensor with shape [batch_size, n_queries, depth_v]
        """
        pass

    def _split_heads(self, tensor):
        pass

    def _combine_heads(self, tensor):
        pass

class MultiHeadAttention(nn.Module):
    """
    Fast multi-head attention. Based on the Attention is All You Need paper.

    https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, n_heads, input_shapes):
        pass


    def forward(self, inputs, mask=None):
        """Fast multi-head self attention.

            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        pass