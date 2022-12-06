import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

import transformer_utils
from transfomer_attention import MultiHeadAttention

class PositionEmbedding(nn.Module):
    def setup(self, hidden_size) -> None:
        pass

    def __call__(self, inputs, start=1):
        pass

class TransformerFeedForward(nn.Module):
    def setup(self, input_size,
                 filter_size,
                 hidden_size,
                 dropout):
        self.fc = nn.Dense(hidden_size)
        self.gelu = TransformerGELU()
        self.proj = nn.Dense(input_size)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, inputs):
        x = self.fc(inputs)
        x = self.gelu(x)  # gelu activation function
        x = self.proj(x)
        x = self.dropout(x)
        return x

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
                 dropout = 0.1) -> None:
        self.norm_1 = nn.LayerNorm(input_size)
        self.attention = MultiHeadAttention(n_heads,[input_size,input_size])
        self.norm_2 = nn.LayerNorm(input_size)
        self.feed_forward = TransformerFeedForward(input_size, filter_size, hidden_size, dropout)

    def __call__(self, inputs, self_attention_mask=None):
        norm_inputs = self.norm_1(inputs)
        attention = self.attention(norm_inputs)
        res_attention = attention + inputs
        output = res_attention + self.feed_forward(self.norm_2(res_attention))
        return output

class TransformerDecoder(nn.Module):
    """
        Stack of TransformerDecoderBlocks. Performs initial embedding to d_model dimensions, then repeated self-attention
        followed by attention on source sequence. Defaults to 6 layers of self-attention.
    """

    def setup(self,
                 embedding_layer,
                 output_layer,
                 n_layers = 6,
                 n_heads,
                 d_model,
                 d_filter,
                 dropout = None) -> None:
        self.embedding_layer = embedding_layer
        self.output_layer = output_layer
        embed_size = self.embedding_layer.embed_size
        self.decoding_stack = []
        for i in range(n_layers):
            decoder = TransformerDecoderBlock(embed_size, n_heads, d_filter, d_model, dropout)
            setattr(self,f"decoder{i}",decoder)
            self.decoding_stack.append(decoder)
        self.output_layer = output_layer

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def __call__(self, input, decoder_mask=None, mask_future=False):
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
        input_embedding = self.embedding_layer(input)
        decoder_output = input_embedding
        for decoder in self.decoding_stack:
            decoder_output = decoder(decoder_output)
        output = self.output_layer(decoder_output)
        return output


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

    def __call__(self, source_sequence, target_sequence, decoder_mask, mask_future=True, shift_target_sequence_right=True):
        pass
