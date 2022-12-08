import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn
from flax.training import train_state, checkpoints

from attention import MultiHeadAttention

import transformer_utils

class PositionEmbedding(nn.Module):
    d_model : int
    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pass

    def __call__(self, x):
        seq_len = jnp.size(x,1)

        pe = jnp.zeros((seq_len, self.d_model))
        position = jnp.arange(0, seq_len, dtype=jnp.float32)[:,None]
        div_term = jnp.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = jnp.sin(position * div_term)
        pe[:, 1::2] = jnp.cos(position * div_term)
        pe = pe[None]
        self.pe = jax.device_put(pe)

        x = x + self.pe[:, :x.shape[1]]
        return x

class TransformerFeedForward(nn.Module):
    intput_size : int
    filter_size : int
    hidden_size : int
    dropout : float
    def setup(self):
        self.fc = nn.Dense(self.hidden_size)
        self.gelu = TransformerGELU()
        self.proj = nn.Dense(self.input_size)
        self.dropout = nn.Dropout(self.dropout)

    def __call__(self, x):
        x = self.fc(x)
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
    input_size : int
    n_heads : int
    filter_size : int
    hidden_size : int
    dropout : float
    def setup(self):
        self.norm_1 = nn.LayerNorm(self.input_size)
        self.attention = MultiHeadAttention(self.n_heads,[self.input_size,self.input_size])
        self.norm_2 = nn.LayerNorm(self.input_size)
        self.feed_forward = TransformerFeedForward(self.input_size, self.filter_size, self.hidden_size, self.dropout)

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
    # embed_size,
    # vocab_size,
    # # output_layer,
    # n_layers = 6,
    # n_heads = 8,
    # d_model = 512,
    # d_filter = 2048,
    # dropout = 0.1
    embed_size : int
    vocab_size : int
    n_layers : int
    n_heads : int
    d_model : int
    d_filter : int
    dropout : float
    def setup(self):

        self.token_embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.pos_embedding = PositionEmbedding(self.d_model, self.embed_size)
        # self.pos_embedding = nn.Embed(d_model, self.embed_size)

        self.output_layer = nn.Dense(self.vocab_size, use_bias=False)

        self.decoding_stack = []
        for i in range(self.n_layers):
            decoder = TransformerDecoderBlock(self.embed_size, self.n_heads, self.d_filter, self.d_model, self.dropout)
            setattr(self,f"decoder{i}",decoder)
            self.decoding_stack.append(decoder)
        # self.output_layer = output_layer
        self.attention_mask = jnp.reshape(jnp.tril(jnp.ones(self.d_model, self.d_model)), (1,1,self.d_model,self.d_model))
        self.norm = nn.LayerNorm(self.embed_size)
        self.drop = nn.Dropout(self.dropout)

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def __call__(self, input, fine_tune = False):
        """
            Args:
                inputs: a tuple of (encoder_output, target_embedding)
                    encoder_output: a float32 Tensor with shape [batch_size, sequence_length, d_model]
                    target_input: either a int32 or float32 Tensor with shape [batch_size, target_length, ndims]
                    cache: Used for fast decoding, a dictionary of tf.TensorArray. None during training.
                mask_future: a boolean for whether to mask future states in target self attention

            Returns:
                a tuple of (embedding_output, output)
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

        embedding_output = self.token_embedding.attend(decoder_output)
        output = None
        if fine_tune:
            output = self.output_layer(decoder_output)

        return embedding_output, output
