"""Tests"""

import pytest
import os
import functools
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map
from jax.experimental import mesh_utils

import einops
from jaxtyping import Float, Int, Array

from models.attention.kernels import tpa

jax.config.update("jax_traceback_filtering", "off")


def reference(
    q: Float[Array, "b lq rq (h + dh)"],
    k: Float[Array, "b lk rk (h + dh)"],
    v: Float[Array, "b lk rk (h + dv)"],
    q_segment_ids: Int[Array, "b lq"],
    kv_segment_ids: Int[Array, "b lk"],
    num_heads: int,
    sm_scale: float = 1.0,
    causal: bool = False,
) -> Float[Array, "b lq h dv"]:
    """Reference implementation of Tensor Product Attention."""

    qa, qb = q[:, :, :, :num_heads], q[:, :, :, num_heads:]
    ka, kb = k[:, :, :, :num_heads], k[:, :, :, num_heads:]
    va, vb = v[:, :, :, :num_heads], v[:, :, :, num_heads:]

    batch_size, lq, rq, _ = qa.shape
    _, lk, rk, _ = ka.shape

    bb = einops.einsum(qb, kb, " b lq rq d, b lk rk d -> b lq lk rq rk")
    x = einops.einsum(qa, ka, bb, "b lq rq h, b lk rk h, b lq lk rq rk -> b h lq lk")
    x *= sm_scale / (rq * rk)

    segment_mask = q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]
    x += jnp.where(segment_mask[:, None, :, :], 0, -jnp.inf)

    if causal:
        mask = jnp.tril(jnp.ones(shape=(lq, lk), dtype=bool))
        x += jnp.where(mask, 0, -jnp.inf)

    p = jax.nn.softmax(x, axis=-1)
    assert p.shape == (batch_size, num_heads, lq, lk)

    return einops.einsum(p, va, vb, "b h lq lk, b lk rk h, b lk rk dv -> b lq h dv")


@pytest.mark.parametrize("rank_q", [1, 2, 6])
@pytest.mark.parametrize("rank_k", [1, 2, 4])
@pytest.mark.parametrize(
    "lq,lk,h,dk,dv",
    [
        (8, 8, 1, 4, 6),
        (1024, 128, 4, 32, 8),
        (128, 1024, 4, 32, 8),
    ],
)
def test_ring_attention_forward(
    rank_q: int,
    rank_k: int,
    lq: int,
    lk: int,
    h: int,
    dk: int,
    dv: int,
):
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)

    batch_size = 2

    q = jax.random.normal(keys[0], shape=(batch_size, lq, rank_q, (h + dk)))
    k = jax.random.normal(keys[0], shape=(batch_size, lk, rank_k, (h + dk)))
    v = jax.random.normal(keys[0], shape=(batch_size, lk, rank_k, (h + dv)))

    q_segment_ids = jnp.ones(shape=(batch_size, lq), dtype=int)
    kv_segment_ids = jnp.ones(shape=(batch_size, lk), dtype=int)

    out_ref = reference(
        q,
        k,
        v,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        num_heads=h,
    )
    assert out_ref.shape == (batch_size, lq, h, dv), out_ref.shape

    out = tpa.tpa(
        q,
        k,
        v,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        num_heads=h,
        debug=True,
        interpret=True,
    )

    np.testing.assert_allclose(out, out_ref, rtol=1e-3, atol=1e-3)
