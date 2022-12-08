import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

from transformer_utils import ApplyAttentionMask


class AttentionQKV(nn.Module):
    """
    Computes attention based on provided similarity metric.
    """

    def setup(self):
        self.apply_mask = ApplyAttentionMask()

    def __call__(self, queries, keys, values, mask=None):
        """Fast scaled dot product attention.

            :param queries: Tensor with shape [batch_size, heads (optional), n_queries, depth_k]
            :param keys:    Tensor with shape [batch_size, heads (optional), n_keyval, depth_k]
            :param values:  Tensor with shape [batch_size, heads (optional), n_keyval, depth_v]
            :param mask:    Tensor with shape [batch_size, n_queries, n_queries]

            :return: output: Tensor with shape [batch_size, heads (optional), n_queries, depth_v]
        """
        key_dim = jnp.array(keys.shape[-1], dtype=jnp.float32)
        similarity = jnp.matmul(queries, keys.transpose(1,2))
        masked_similarity = self.apply_mask(similarity, mask=mask)
        weights = nn.softmax(similarity / jnp.sqrt(key_dim), axis=-1)
        output = jnp.matmul(weights, values)

        return output, weights


class MultiHeadProjection(nn.Module):
    n_heads : int
    feature_sizes : int
    def setup(self):
        """Map the multi-headed attention across the map

        Arguments:
            n_heads {int} -- The number of heads in the attention map
            feature_sizes {int} -- The size of the feature dimensions for key, query, and value

        """

        self.attention_map = AttentionQKV()

        for size in self.feature_sizes:
            assert size % self.n_heads == 0

    def __call__(self, inputs, mask=None):
        """Fast multi-head attention.

        :param queries: Tensor with shape [batch_size, n_queries, depth_k]
        :param keys:    Tensor with shape [batch_size, n_keyval, depth_k]
        :param values:  Tensor with shape [batch_size, n_keyval, depth_v]

        :return: output: Tensor with shape [batch_size, n_queries, depth_v]
        """
        queries, keys, values = inputs
        queries_split = self._split_heads(queries)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        attention_output_split, _ = self.attention_map(queries_split, keys_split, values_split, mask=mask)

        output = self._combine_heads(attention_output_split)
        return output

    def _split_heads(self, array):
        assert len(array.shape) == 3

        batch_size, arraylen = array.shape[0], array.shape[1]
        feature_size = array.shape[2]

        new_feature_size = feature_size / self.n_heads
        array = array.reshape(batch_size, self.n_heads, arraylen, new_feature_size)
        array = array.transpose(1,3)
        return array

    def _combine_heads(self, array):
        assert len(array.shape) == 4
        array = array.transpose(1, 3)
        batch_size, arraylen = array.shape[0], array.shape[1]
        feature_size = array.shape[-1]

        new_feature_size = feature_size * self.n_heads
        array = array.reshape(batch_size, arraylen, new_feature_size)
        return array

class MultiHeadAttention(nn.Module):
    """
    Fast multi-head attention. Based on the Attention is All You Need paper.

    https://arxiv.org/pdf/1706.03762.pdf
    """
    n_heads : int
    input_shapes : int
    def setup(self):
        self.qa_channels, self.ma_channels = self.input_shapes
        self.attention_layer = MultiHeadProjection()

        assert self.qa_channels % self.n_heads == 0 and self.ma_channels % self.n_heads == 0 and \
                                                        'Feature size must be divisible by n_heads'
        assert self.qa_channels == self.ma_channels and 'Cannot combine tensors with different shapes'

        self.query_layer = nn.LayerNorm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))
        self.key_layer = nn.LayerNorm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))
        self.value_layer = nn.LayerNorm(nn.Linear(self.ma_channels, self.ma_channels, bias=False))

        self.output_layer = nn.LayerNorm(nn.Linear(self.qa_channels, self.qa_channels, bias=False))

    def weights_init(m):
        nn.initializers.normal(stddev=0.02)
        self.query_layer.apply(weights_init)
        self.key_layer.apply(weights_init)
        self.value_layer.apply(weights_init)
        self.output_layer.apply(weights_init)

    def __call__(self, inputs, mask=None):
        """Fast multi-head self attention.

            :param inputs: tuple of (query_antecedent, memory_antecedent)
                query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
                memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
        """
        query_antecedent, memory_antecedent = inputs
        q = self.query_layer(query_antecedent)
        k = self.key_layer(memory_antecedent)
        v = self.value_layer(memory_antecedent)

        attention_output = self.attention_layer((q, k, v), mask=mask)
        output = self.output_layer(attention_output)
        return output
