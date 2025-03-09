"""Tests for Tensor Product Attention.

uv run python -m pytest -s --pdb models/attention/kernels/tpa_test.py
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

import einops
from jaxtyping import Float, Int, Array

from models.attention.kernels import tpa
from models.attention.mha import mha


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

    # easiest implementation is fully materialize q,k,v by tensor product.
    q_ = einops.einsum(qa, qb, "b lq rq h, b lq rq dk -> b lq h dk") / rq
    k_ = einops.einsum(ka, kb, "b lk rk h, b lk rk dk -> b lk h dk") / rk
    v_ = einops.einsum(va, vb, "b lk rk h, b lk rk dv -> b lk h dv") / rk

    segment_mask = q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]
    bias = jnp.where(segment_mask, 0, -jnp.inf)
    if causal:
        mask = jnp.tril(jnp.ones(shape=(lq, lk), dtype=bool))
        bias += jnp.where(mask, 0, -jnp.inf)[None, :, :]
    assert bias.shape == (batch_size, lq, lk)

    return mha(q_, k_, v_, bias=bias, sm_scale=sm_scale)


@pytest.mark.parametrize("rank_q", [1, 6])
@pytest.mark.parametrize("rank_k", [1, 4])
@pytest.mark.parametrize("nomat", [False, True])
@pytest.mark.parametrize(
    "batch_size,lq,lk,h,dk,dv,segment_size",
    [
        (1, 8, 8, 1, 4, 6, 8),
        (2, 1024, 128, 2, 3, 5, 42),
        (2, 1024, 128, 2, 3, 5, None),
        (2, 128, 1024, 2, 3, 5, 42),
        (2, 128, 1024, 2, 3, 5, None),
        (1, 1024, 1024, 2, 3, 5, 321),
    ],
)
def test_tpa_forward(
    batch_size: int,
    rank_q: int,
    rank_k: int,
    nomat: bool,
    lq: int,
    lk: int,
    h: int,
    dk: int,
    dv: int,
    segment_size: int | None,
):
    # pytest.skip()
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 3)

    q = jax.random.uniform(keys[0], shape=(batch_size, lq, rank_q, (h + dk)))
    k = jax.random.uniform(keys[1], shape=(batch_size, lk, rank_k, (h + dk)))
    v = jax.random.uniform(keys[2], shape=(batch_size, lk, rank_k, (h + dv)))

    if segment_size is not None:
        q_segment_ids = einops.repeat(
            jnp.arange(lq) // segment_size, "l -> b l", b=batch_size
        )
        kv_segment_ids = einops.repeat(
            jnp.arange(lk) // segment_size, "l -> b l", b=batch_size
        )
    else:
        q_segment_ids = jnp.zeros(shape=(batch_size, lq), dtype=int)
        kv_segment_ids = jnp.zeros(shape=(batch_size, lk), dtype=int)
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
        nomat=nomat,
        debug=False,
        interpret=True,
    )
    assert out.shape == (batch_size, lq, h, dv), out_ref.shape
    np.testing.assert_allclose(out, out_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("rank_q", [1, 6])
@pytest.mark.parametrize("rank_k", [1, 4])
@pytest.mark.parametrize("nomat", [False, True])
@pytest.mark.parametrize(
    "batch_size,lq,lk,h,dim_k,dim_v,segment_size",
    [
        (1, 8, 8, 1, 4, 6, 8),
        (2, 1024, 128, 2, 3, 5, 42),
        (2, 1024, 128, 2, 3, 5, None),
        (2, 128, 1024, 2, 3, 5, 42),
        (2, 128, 1024, 2, 3, 5, None),
        (1, 1024, 1024, 2, 3, 5, 321),
    ],
)
def test_tpa_backwards(
    batch_size: int,
    rank_q: int,
    rank_k: int,
    nomat: bool,
    lq: int,
    lk: int,
    h: int,
    dim_k: int,
    dim_v: int,
    segment_size: int | None,
):
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)

    q = jax.random.normal(keys[0], shape=(batch_size, lq, rank_q, (h + dim_k)))
    k = jax.random.normal(keys[1], shape=(batch_size, lk, rank_k, (h + dim_k)))
    v = jax.random.normal(keys[2], shape=(batch_size, lk, rank_k, (h + dim_v)))

    if segment_size is not None:
        q_segment_ids = einops.repeat(
            jnp.arange(lq) // segment_size, "l -> b l", b=batch_size
        )
        kv_segment_ids = einops.repeat(
            jnp.arange(lk) // segment_size, "l -> b l", b=batch_size
        )
    else:
        q_segment_ids = jnp.zeros(shape=(batch_size, lq), dtype=int)
        kv_segment_ids = jnp.zeros(shape=(batch_size, lk), dtype=int)

    def _ref(q, k, v, impl: str) -> Float[Array, ""]:
        if impl == "ref":
            o = reference(
                q,
                k,
                v,
                q_segment_ids=q_segment_ids,
                kv_segment_ids=kv_segment_ids,
                num_heads=h,
                causal=False,
            )
        else:
            o = tpa.tpa(
                q,
                k,
                v,
                q_segment_ids=q_segment_ids,
                kv_segment_ids=kv_segment_ids,
                num_heads=h,
                nomat=nomat,
                causal=False,
                interpret=True,
            )
        return jnp.nansum(o)

    dq_, dk_, dv_ = jax.grad(_ref, argnums=(0, 1, 2))(q, k, v, impl="ref")

    try:
        dq, dk, dv = jax.grad(_ref, argnums=(0, 1, 2))(q, k, v, impl="kernel")
    except NotImplementedError:
        pytest.skip()

    np.testing.assert_allclose(dq, dq_, atol=1e-3, rtol=1e-2)
    np.testing.assert_allclose(dk, dk_, atol=1e-3, rtol=1e-2)
    np.testing.assert_allclose(dv, dv_, atol=1e-3, rtol=1e-2)
