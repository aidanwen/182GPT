import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

from attention import MultiHeadAttention

import transformer_utils

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

    :param inputs: Tensor of decoder_inputs
\                    decoder_inputs -> a Tensor with shape [batch_size, decoding_sequence_length, channels]

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
        attention = self.attention(norm_inputs, mask=self_attention_mask)
        res_attention = attention + inputs
        output = res_attention + self.feed_forward(self.norm_2(res_attention))
        return output

class TransformerDecoder(nn.Module):
    """
        Stack of TransformerDecoderBlocks. Performs initial embedding to d_model dimensions, then repeated self-attention
        followed by attention on source sequence. Defaults to 6 layers of self-attention.
    """

    def setup(self,
                 embed_size,
                 vocab_size,
                 # output_layer,
                 n_layers = 6,
                 n_heads = 8,
                 d_model = 512,
                 d_filter = 2048,
                 dropout = 0.1) -> None:

        self.embed_size = embed_size
        self.token_embedding = nn.Embed(vocab_size, self.embed_size)
        self.pos_embedding = nn.Embed(d_model, self.embed_size)

        self.output_layer = nn.Dense(vocab_size, use_bias=False)

        self.decoding_stack = []
        for i in range(n_layers):
            decoder = TransformerDecoderBlock(embed_size, n_heads, d_filter, d_model, dropout)
            setattr(self,f"decoder{i}",decoder)
            self.decoding_stack.append(decoder)
        self.output_layer = output_layer
        self.attention_mask = jnp.reshape(jnp.tril(jnp.ones(d_model, d_model)), (1,1,d_model,d_model))
        self.norm = nn.LayerNorm(embed_size)
        self.drop = nn.Dropout(dropout)

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def __call__(self, input, mask_future=True):
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
        seq_len = jnp.size(decoder_output,1)

        pos = jnp.expand_dims(jnp.arange(0, stop=seq_len,dtype=jnp.long),0)

        tok_embed = self.token_embedding(input) # (batch_size, sequence_length, d_model)
        pos_embed = self.pos_embedding(pos) # (1, sequence_length, d_model)

        decoder_input = self.dropout(tok_embed + pos_embed)

        self_attention_mask = (self.attention_mask[:,:,:seq_len,:seq_len] == 0)

        for decoder in self.decoding_stack:
            decoder_output = decoder(decoder_output, self_attention_mask = self_attention_mask)

        decoder_output = self.norm(decoder_output)
        output = self.output_layer(decoder_output)
        return output


