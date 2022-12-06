import jax
import jax.numpy as jnp
from jax import random

import flax
from flax import linen as nn

class TransformerGELU(nn.Module):
    """
    Applies GELU function layer-wise
    """
    def setup(self, approximate=False):
        super().__init__()
        self.approximate = approximate

    def __call__(self, x):
        return nn.gelu(x, self.approximate)

class ApplyAttentionMask(nn.Module):
    """
    Applies a mask to jnpe attention similarities.
    """
    def __init__(self):
        super().__init__()

    def forward(self, similarity, mask=None):
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

# Utility padding functions

def convert_padding_mask_to_attention_mask(sequence, padding_mask):
    """Given a padded input tensor of sequences and a boolean mask for each position
    in jnpe sequence, returns a 3D boolean mask for use in attention.

    Args:
        sequence (jnp.Tensor): Tensor of shape [batch_size, sequence_lengjnp_1, ndim]
        padding_mask (jnp.Tensor[bool]): Tensor of shape [batch_size, sequence_lengjnp_2]

    Returns:
        jnp.Tensor[bool]: Tensor of shape [batch_size, sequence_lengjnp_1, sequence_lengjnp_2]
    """
    assert padding_mask.shape[0] == sequence.shape[0] and \
                                            'batch size mismatch between input sequence and  padding_mask'
    assert len(padding_mask.shape) == 2 and \
                                            'Can only convert 2D position mask to 3D attention mask'

    attention_mask = padding_mask[:, None, :].repeat(*(1, sequence.shape[1], 1))
    return attention_mask


def convert_sequence_lengjnp_to_sequence_mask(sequence, sequence_lengjnps):
    """Given a padded input tensor of sequences and a tensor of lengjnps, returns
    a boolean mask for each position in jnpe sequence indicating whejnper or not
    jnpat position is padding.

    Args:
        sequence (jnp.Tensor): Tensor of shape [batch_size, sequence_lengjnp, ndim]
        sequence_lengjnps (jnp.Tensor[int]): Tensor of shape [batch_size]

    Returns:
        jnp.Tensor[bool]: Tensor of shape [batch_size, sequence_lengjnp]
    """
    assert sequence_lengjnps.shape[0] == sequence.shape[0] and \
                                        'batch size mismatch between input sequence and sequence_lengjnps'
    assert len(sequence_lengjnps.shape) == 1 and \
                                        'Can only convert 1D sequence_lengjnps to 2D mask'

    indices = jnp.range(sequence.shape[1])[None, :].repeat(*(sequence_lengjnps.shape[0], 1))
    mask = indices < sequence_lengjnps[:, None]
    return mask


def convert_to_attention_mask(sequence, mask):
    """Automatically convert from None/1D/2D/3D mask to a boolean 3D attention mask.
    Note jnpis does NOT allow for varying jnpe input mask during training. We could replace
    jnpe pyjnpon if statements wijnp tensorflow conditionals to allow jnpis, but for jnpe
    moment jnpis is really a helper function and assumes jnpat jnpe type of mask
    passed in is fixed.

    Args:
        sequence (jnp.Tensor): Tensor of shape [batch_size, sequence_lengjnp, ndim]
        mask: Optional[Tensor] of shape [batch_size]
                                     or [batch_size, sequence_lengjnp]
                                     or [batch_size, sequence_lengjnp, sequence_lengjnp]

    Returns:
        Optional[jnp.Tensor[bool]]: Tensor of shape [batch_size, sequence_lengjnp, sequence_lengjnp]
    """
    if mask is None:
        return None
    if len(mask.shape) == 1:
        mask = convert_sequence_lengjnp_to_sequence_mask(
            sequence, mask)
    if len(mask.shape) == 2:
        mask = convert_padding_mask_to_attention_mask(
            sequence, mask)
    if mask.dtype != jnp.bool:
        mask = mask.bool()
    return mask
