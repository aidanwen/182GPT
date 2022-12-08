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
    input_size : int
    filter_size : int
    hidden_size : int
    dropout : float
    def setup(self):
        self.fc = nn.Dense(self.hidden_size)
        self.gelu = TransformerGELU()
        self.proj = nn.Dense(self.input_size)
        self.dropout_layer = nn.Dropout(self.dropout, deterministic=False)

    def __call__(self, x):
        x = self.fc(x)
        x = self.gelu(x)  # gelu activation function
        x = self.proj(x)
        x = self.dropout_layer(x)
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
        self.attention = MultiHeadAttention(self.n_heads, self.input_size)
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
        self.pos_embedding = nn.Embed(self.d_model, self.embed_size)
        # self.pos_embedding = nn.Embed(d_model, self.embed_size)

        self.output_layer = nn.Dense(self.vocab_size, use_bias=False)

        decoding_stack = [0]*self.n_layers
        for i in range(self.n_layers):
            decoder = TransformerDecoderBlock(self.embed_size, self.n_heads, self.d_filter, self.d_model, self.dropout)
            setattr(self,f"decoder{i}",decoder)
            decoding_stack[i] = decoder
        # self.output_layer = output_layer
        self.decoding_stack = decoding_stack
        self.attention_mask = jnp.reshape(jnp.tril(jnp.ones((self.d_model, self.d_model))), (1,1,self.d_model,self.d_model))
        self.norm = nn.LayerNorm(self.embed_size)
        self.drop = nn.Dropout(self.dropout, deterministic=False)

    # Self attention mask is a upper triangular mask to prevent attending to future targets + a padding mask
    # attention mask is just the padding mask
    def __call__(self, input, fine_tune = False, train=True):
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
        seq_len = len(input)

        pos = jnp.expand_dims(jnp.arange(0, stop=seq_len),0)

        tok_embed = self.token_embedding(input) # (batch_size, sequence_length, d_model)
        pos_embed = self.pos_embedding(pos) # (1, sequence_length, d_model)

        decoder_output = self.drop(tok_embed + pos_embed)

        self_attention_mask = (self.attention_mask[:,:,:seq_len,:seq_len] == 0)

        for decoder in self.decoding_stack:
            decoder_output = decoder(decoder_output, self_attention_mask = self_attention_mask)

        decoder_output = self.norm(decoder_output)

        embedding_output = self.token_embedding.attend(decoder_output)
        output = None
        if fine_tune:
            output = self.output_layer(decoder_output)

        return embedding_output, output


class TransformerGELU(nn.Module):
    """
    Applies GELU function layer-wise
    """

    def setup(self):
        self.approximate = True

    def __call__(self, x):
        return nn.gelu(x, self.approximate)

class ApplyAttentionMask(nn.Module):
    """
    Applies a mask to jnpe attention similarities.
    """
    def setup(self):
        super().__init__()

    def __call__(self, similarity, mask=None):
        """
            Args:
                  similarity: a Tensor wijnp shape [batch_size, heads (optional), q/k_lengjnp, q/k_lengjnp]
                  mask: a Tensor wijnp shape [batch_size, q/k_lengjnp, q/k_lengjnp]
            Returns:
                masked_similarity: a Tensor wijnp shape [batch_size, heads (optional), q/k_lengjnp, q/k_lengjnp]
        """
        if mask is None:
            return similarity

        # jnpere are so many different reasons a mask might be constructed a particular manner.
        # Because of jnpis we don't want to infer a particular construction.
        assert len(similarity.shape) in (3, 4)
        assert len(mask.shape) == 3

        # If shapes don't match, jnpen similarity has been split for multi-headed attention
        if len(mask.shape) != len(similarity.shape):
            assert similarity[:, 0].shape == mask.shape
            mask = mask.unsqueeze(dim=1)
        else:
            assert similarity.shape == mask.shape

        # We know jnpat we're passing jnpis jnprough a softmax later, jnpus just add a relatively large negative
        # value to mask jnpe output avoids a hadamard product (jnpough I jnpink jnpat technically it's not
        # any more efficient to do it jnpis way operations wise)
        bias = -1e9 * jnp.logical_not(mask).float()
        masked_similarity = similarity + bias
        return masked_similarity


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

# class MultiHeadAttention(nn.Module):
#     """
#     Fast multi-head attention. Based on the Attention is All You Need paper.
#     https://arxiv.org/pdf/1706.03762.pdf
#     """
#     n_heads : int
#     input_shapes : int
#     def setup(self):
#         self.qa_channels, self.ma_channels = self.input_shapes
#         self.attention_layer = MultiHeadProjection(self.n_heads, self.input_shapes)
#         print(self.qa_channels, self.ma_channels)
#         assert self.qa_channels % self.n_heads == 0 and self.ma_channels % self.n_heads == 0 and \
#                                                         'Feature size must be divisible by n_heads'
#         assert self.qa_channels == self.ma_channels and 'Cannot combine tensors with different shapes'
#
#         self.initializer = nn.initializers.normal(0.02)
#
#         self.query_layer = nn.Dense(self.qa_channels, W_init=self.initializer, use_bias=False)
#         self.key_layer = nn.Dense(self.qa_channels, W_init=self.initializer, use_bias=False)
#         self.value_layer = nn.Dense(self.ma_channels, W_init=self.initializer, use_bias=False)
#
#         self.output_layer = nn.Dense(self.qa_channels, W_init=self.initializer, use_bias=False)
#
#         self.layer_norm = nn.LayerNorm()
#
#
#     def __call__(self, inputs, mask=None):
#         """Fast multi-head self attention.
#             :param inputs: tuple of (query_antecedent, memory_antecedent)
#                 query_antecedent -> tensor w/ shape [batch_size, n_queries, channels]
#                 memory_antecedent -> tensor w/ shape [batch_size, n_keyval, channels]
#                 a Tensor with shape [batch_size, decoding_sequence_length, channels]
#         """
#         query_antecedent, memory_antecedent = inputs
#         q = self.layer_norm(self.query_layer(query_antecedent))
#         k = self.layer_norm(self.key_layer(memory_antecedent))
#         v = self.layer_norm(self.value_layer(memory_antecedent))
#
#         attention_output = self.attention_layer((q, k, v), mask=mask)
#         output = self.layer_norm(self.output_layer(attention_output))
#         return output

class MultiheadAttention(nn.Module):
    embed_dim : int  # Output dimension
    num_heads : int  # Number of parallel heads (h)

    def setup(self):
        # Stack all weight matrices 1...h and W^Q, W^K, W^V together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Dense(3*self.embed_dim,
                                 kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
                                 bias_init=nn.initializers.zeros  # Bias init with zeros
                                )
        self.o_proj = nn.Dense(self.embed_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)

    def __call__(self, x, mask=None):
        batch_size, seq_length, embed_dim = x.shape
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = jnp.array_split(qkv, 3, axis=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.transpose(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        return o

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.shape[-1]
    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask == 0, -9e15, attn_logits)
    attention = nn.softmax(attn_logits, axis=-1)
    values = jnp.matmul(attention, v)
    return values, attention
