"""Tests for Tensor Product Attention.

uv run python -m pytest -s --pdb models/attention/kernels/tpa_test.py
"""

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

    # just test by fully forming q,k,v
    q_ = einops.einsum(qa, qb, "b lq rq h, b lq rq dk -> b lq h dk") / rq
    k_ = einops.einsum(ka, kb, "b lk rk h, b lk rk dk -> b lk h dk") / rk
    v_ = einops.einsum(va, vb, "b lk rk h, b lk rk dv -> b lk h dv") / rk

    x = einops.einsum(q_, k_, "b lq h dk, b lk h dk -> b h lq lk")
    x *= sm_scale

    segment_mask = q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]
    x += jnp.where(segment_mask[:, None, :, :], 0, -jnp.inf)

    if causal:
        mask = jnp.tril(jnp.ones(shape=(lq, lk), dtype=bool))
        x += jnp.where(mask, 0, -jnp.inf)

    p = jax.nn.softmax(x, axis=-1)
    assert p.shape == (batch_size, num_heads, lq, lk)

    return einops.einsum(p, v_, "b h lq lk, b lk h dv -> b lq h dv")


def mha_jax(
    q: Float[Array, "b lq h dk"],
    k: Float[Array, "b lk h dk"],
    v: Float[Array, "b lk h dv"],
    sm_scale: Float[Array, "b lq"],
    bias: Float[Array, "b lq lk"] | None = None,
) -> Float[Array, "b lq h dv"]:
    """Batched multi-head attention."""
    qk = einops.einsum(q, k, "b lq h dk, b lk h dk -> b h lq lk")
    qk *= sm_scale[:, None, :, None]
    if bias is not None:
        qk += bias
    p = jax.nn.softmax(qk, axis=-1)
    return einops.einsum(p, v, "b h lq lk, b lk h dv -> b lq h dv")


@pytest.mark.parametrize("rank_q", [1, 2, 6])
@pytest.mark.parametrize("rank_k", [1, 2, 4])
@pytest.mark.parametrize(
    "batch_size,lq,lk,h,dk,dv",
    [
        (1, 8, 8, 1, 4, 6),
        (2, 1024, 128, 4, 32, 8),
        (2, 128, 1024, 4, 32, 8),
    ],
)
def test_tpa_forward(
    batch_size: int,
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

    q = jax.random.normal(keys[0], shape=(batch_size, lq, rank_q, (h + dk)))
    k = jax.random.normal(keys[1], shape=(batch_size, lk, rank_k, (h + dk)))
    v = jax.random.normal(keys[2], shape=(batch_size, lk, rank_k, (h + dv)))

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
        debug=False,
        interpret=True,
    )
    assert out.shape == (batch_size, lq, h, dv), out_ref.shape
    np.testing.assert_allclose(out, out_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("rank_q", [1, 2, 6])
@pytest.mark.parametrize("rank_k", [1, 2, 4])
@pytest.mark.parametrize(
    "batch_size,lq,lk,h,dk,dv",
    [
        (1, 8, 8, 1, 4, 6),
        (2, 1024, 128, 4, 32, 8),
        (2, 128, 1024, 4, 32, 8),
    ],
)
def test_tpa_backwards(
    batch_size: int,
    rank_q: int,
    rank_k: int,
    lq: int,
    lk: int,
    h: int,
    dk: int,
    dv: int,
):
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)

    q = jax.random.normal(keys[0], shape=(batch_size, lq, rank_q, (h + dk)))
    k = jax.random.normal(keys[1], shape=(batch_size, lk, rank_k, (h + dk)))
    v = jax.random.normal(keys[2], shape=(batch_size, lk, rank_k, (h + dv)))

    q_segment_ids = jnp.ones(shape=(batch_size, lq), dtype=int)
    kv_segment_ids = jnp.ones(shape=(batch_size, lk), dtype=int)

    do = jax.random.normal(keys[4], shape=(batch_size, lq, h, dv))

    def _ref(q, k, v):
        return do * reference(
            q,
            k,
            v,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            num_heads=h,
            causal=False,
        )

    dq_, dk_, dv_ = jax.grad(_ref, argnums=(0, 1, 2))(q, k, v)

    dq, dk, dv = jax.grad(
        lambda q, k, v: jnp.sum(
            do
            * tpa.tpa(
                q,
                k,
                v,
                q_segment_ids=q_segment_ids,
                kv_segment_ids=kv_segment_ids,
                causal=False,
            )
        ),
        argnums=(0, 1, 2),
    )(q, k, v)

    np.testing.assert_allclose(dq, dq_, atol=1e-4)
    np.testing.assert_allclose(dk, dk_, atol=1e-4)
    np.testing.assert_allclose(dv, dv_, atol=1e-4)
