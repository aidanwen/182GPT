import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

class PositionEmbedding(nn.Module):
    def setup(self, hidden_size) -> None:
        pass

    def __call__(self, inputs, start=1):
        pass

class TransformerFeed__call__(nn.Module):
    def setup(self, input_size,
                 filter_size,
                 hidden_size,
                 dropout):
        pass

    def __call__(self, inputs):
        pass

class TransformerDecoderBlock(nn.Module):
    """A decoding block from the paper Attention Is All You Need (https://arxiv.org/pdf/1706.03762.pdf).

    :param inputs: two Tensors encoder_outputs, decoder_inputs
                    encoder_outputs -> a Tensor with shape [batch_size, sequence_length, channels]
                    decoder_inputs -> a Tensor with shape [batch_size, decoding_sequence_length, channels]

    :return: output: Tensor with same shape as decoder_inputs
    """

    def setup(self,
                 input_size,
                 n_heads,
                 filter_size,
                 hidden_size,
                 dropout = None) -> None:
        pass

    def __call__(self, decoder_inputs, encoder_outputs, self_attention_mask=None, cross_attention_mask=None):
        pass

class TransformerDecoder(nn.Module):
    """
        Stack of TransformerDecoderBlocks. Performs initial embedding to d_model dimensions, then repeated self-attention
        followed by attention on source sequence. Defaults to 6 layers of self-attention.
    """

    def setup(self,
                 embedding_layer,
                 output_layer,
                 n_layers,
                 n_heads,
                 d_model,
                 d_filter,
                 dropout = None) -> None:
        pass

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def __call__(self, target_input, encoder_output, encoder_mask=None, decoder_mask=None, mask_future=False,
        shift_target_sequence_right=False):
        """
            Args:
                inputs: a tuple of (encoder_output, target_embedding)
                    encoder_output: a float32 Tensor with shape [batch_size, sequence_length, d_model]
                    target_input: either a int32 or float32 Tensor with shape [batch_size, target_length, ndims]
                    cache: Used for fast decoding, a dictionary of tf.TensorArray. None during training.
                mask_future: a boolean for whether to mask future states in target self attention

            Returns:
                a tuple of (encoder_output, output)
                    output: a Tensor with shape [batch_size, sequence_length, d_model]
        """
        pass

    def shift_target_sequence_right(self, target_sequence):
        pass

    def get_future_mask(self, batch_size, sequence_length):
        """Mask future targets and padding

            :param batch_size: a Tensor dimension
            :param sequence_length: a Tensor dimension
            :param padding_mask: None or bool Tensor with shape [batch_size, sequence_length]

            :return mask Tensor with shape [batch_size, sequence_length, sequence_length]
        """
        pass

    def get_self_attention_mask(self, batch_size, sequence_length, decoder_mask, mask_future):
        pass

    # This is an upper left block matrix which masks the attention for things that don't
    # exist within the internals.
    def get_cross_attention_mask(self, encoder_output, decoder_input, encoder_mask, decoder_mask):
        pass
        
        
class TransformerInputEmbedding(nn.Module):

    def setup(self,
                 embed_size,
                 vocab_size = None,
                 dropout = None,
                 batch_norm = False,
                 embedding_initializer=None) -> None:
        pass

    def __call__(self, inputs, start=1):
        pass

class Transformer(nn.Module):

    def setup(self,
                 vocab_size = None,
                 n_layers = 6,
                 n_heads = 8,
                 d_model = 512,
                 d_filter = 2048,
                 dropout = None,
                 embedding_initializer=None,
                 **kwargs) -> None:
        pass

    def __call__(self, source_sequence, target_sequence, encoder_mask, decoder_mask, mask_future=True, shift_target_sequence_right=True):
        pass