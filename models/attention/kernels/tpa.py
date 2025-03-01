"""Tensor Product Attention Pallas Kernels."""

from __future__ import annotations

import functools
from typing import Any
import einops

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from jaxtyping import Int, Bool, Float, Array


def reference(
    aq: Float[Array, "b lq rq h"],
    bq: Float[Array, "b lq rq dh"],
    ak: Float[Array, "b lk rk h"],
    bk: Float[Array, "b lk rk dh"],
    av: Float[Array, "b lk rv h"],
    bv: Float[Array, "b lk rv dv"],
    sm_scale: Float,
) -> Float[Array, "b lq h dv"]:
    """Tensor product attention without materializing Q, K, V.

    TODO: make a pallas kernel for it.

    """
    b, lq, rq, h = aq.shape
    _, lk, rk, dh = bv.shape

    # lol
    bb = einops.einsum(bq, bk, " b lq rq d, b lk rk d -> b lq lk rq rk")

    qk = einops.einsum(
        aq, ak, bb, "b lq rq h, b lk rk h, b lq lk rq rk -> b h lq lk"
    ) / (rq * rk)

    qk *= sm_scale
    p = jax.nn.softmax(qk, axis=-1)
    return einops.einsum(p, av, bv, "b h lq lk, b lk rv h, b lk rv dv -> b h lq rv")


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def tpa(
    q: Float[Array, "b lq rq h (1 + dh)"],
    k: Float[Array, "b lk rk h (1 + dh)"],
    v: Float[Array, "b lk rk h (1 + dv)"],
    q_segment_ids: Int[Array, "b lq"],
    kv_segment_ids: Int[Array, "b lk"],
    sm_scale: Float,
    causal: bool = False,
) -> Float[Array, "b lq h dv"]:
    """Tensor Product Attention kernel."""
    batch_size, q_len, rank_q, dim_k = q.shape
    _, k_len, rank_k, dim_v = v.shape

    o, _ = jax.vmap(  # over batch
        jax.vmap(
            functools.partial(_fwd_kernel, causal=causal, sm_scale=sm_scale),
            in_axes=[2, 2, 2, None, None],
            out_axes=1,
        ),
        in_axes=[0, 0, 0, 0, 0],
        out_axes=0,
    )(q, k, v, q_segment_ids, kv_segment_ids)
    return o


def lse(x: jax.Array) -> jax.Array:
    """Log sum exp."""
    m = jnp.max(x, axis=-1)
    return m + jnp.log(jnp.sum(jnp.exp(x - m[..., None]), axis=-1))


def lse_combine(a: jax.Array, b: jax.Array) -> jax.Array:
    """Combine LSE.
    a > b
    log(exp(a) + exp(b))
    = a + log(exp(a - a) + exp(b - a))
    = a + log(1 + exp(b - a))
    = a + softplus(b - a)
    """
    max = jnp.maximum(a, b)
    min = jnp.minimum(a, b)
    return max + jnp.log(1 + jnp.exp(min - max))


def _fwd_kernel(
    qa_ref: Float[Array, "lq rq h"],
    qb_ref: Float[Array, "lq rq dk"],
    ka_ref: Float[Array, "lk rk h"],
    kb_ref: Float[Array, "lk rk dk"],
    va_ref: Float[Array, "lk rk h"],
    vb_ref: Float[Array, "lk rk dv"],
    q_segment_ids_ref: Int[Array, " lq"],
    kv_segment_ids_ref: Int[Array, " lk"],
    o_ref: Float[Array, "lq h dv"],
    sm_scale: float,
    causal: bool,
    block_k: int = 128,
) -> tuple[
    Float[Array, "lq h dv"],  # out
    Float[Array, "lq h"],  # lse
]:
    """TPA forward pallas kernel.

    the main question here is should we parallelize across query heads or ranks?
        - if we parallelize across heads, then we can't share the inner products across
          heads and will have to recompute. on the other hand

    this kernel parallelizes across query ranks

    TODO: parallelize across query ranks and atomic add into output?
        - atomic add would be slow on gpu
        - atomic add wouldn't be necessary on TPU

    Args:
        q_ref: factorized queries (block)
        k_ref: factorized keys
        v_ref: factorized values
        q_segment_ids_ref: segment ids for queries (block)
        kv_segment_ids_ref: segment ids for keys and values.

    Returns:

    """
    block_q, block_h, dv = o_ref.shape

    start_q = pl.program_id(1)
    qa = qa_ref[...]
    qb = qb_ref[...]
    q_segment_ids = q_segment_ids_ref[...]

    def _scan_fn(start_k: int, carry):
        o, l_prev = carry

        kv_slice = pl.dslice(start_k * block_k, block_k)

        ka = ka_ref[kv_slice, slice(None), slice(None)]
        kb = kb_ref[kv_slice, slice(None), slice(None)]

        bb = einops.einsum(qb, kb, "lq rq dk, lk rk dk -> lq lk rq rk")
        x = einops.einsum(qa, ka, bb, "lq rq h, lk rk h, lq lk rk -> h lq lk")

        kv_segment_ids = kv_segment_ids_ref[kv_slice]
        mask = q_segment_ids[:, None] == kv_segment_ids[None, :]
        if causal:
            span_q = start_q * block_q + jnp.arange(block_q)
            span_k = start_k * block_k + jnp.arange(block_k)
            causal_mask = span_q[:, None] >= span_k[None, :]
            mask &= causal_mask
        x = jnp.where(mask[..., None], x, -jnp.inf)

        if sm_scale != 1.0:
            x *= sm_scale  # [block_q, block_k]

        m = jnp.max(x, axis=-1)
        l_ = m + jnp.log(jnp.sum(jnp.exp(x - m), axis=-1))

        l = lse_combine(l_prev, l_)

        log_p = x - l[..., None]
        p = jnp.exp(log_p)

        va = va_ref[kv_slice, slice(None), slice(None)]
        vb = vb_ref[kv_slice, slice(None), slice(None)]

        pva = einops.einsum(p, va, "h lq lk, lk rk h -> lq h rk")
        o_ = einops.einsum(vb, pva, "lq rk dv, lq h rk -> lq h dv")
        o = jnp.exp(l_prev - l)[..., None] * o + o_

        return o, l

    if causal:
        num_kv_blocks = jax.lax.div(block_q * (start_q + 1) + block_k - 1, block_k)
    else:
        kv_len = ka_ref.shape[0]
        num_kv_blocks = pl.cdiv(kv_len, block_k)

    o = jnp.zeros(shape=(block_q, block_h, dv), dtype=jnp.float32)
    l = jnp.zeros(shape=(block_q, block_h), dtype=jnp.float32) - jnp.inf
    o, l = jax.lax.fori_loop(0, num_kv_blocks, _scan_fn, (o, l))

    # store final output
    o_ref[...] = o
    l_ref[...] = l


def _fwd(
    q: Float[Array, "lq rq (h + dk)"],
    k: Float[Array, "lk rk (h + dk)"],
    v: Float[Array, "lk rk (h + dv)"],
    q_segment_ids: Int[Array, " lq"],
    kv_segment_ids: Int[Array, " lk"],
    num_heads: int,
    sm_scale: Float,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    block_h: int = 4,
    debug: bool = False,
    interpret: bool = True,
) -> tuple[
    Float[Array, "lq h dv"],
    Float[Array, "lq h"],
]:
    """Forward pass through TPA.

    Args:
        ...
        block_q: size of query blocks
        block_k: size of key/value blocks
        block_h: number of heads to compute in parallel within each thread. Note that
           this number defines the number of heads which share the same computation of
           query/key inner products across ranks.
    """
    q_len, rank_q, dim_k = q.shape
    k_len, rank_k, dim_v = v.shape

    assert q_len % block_q == 0, (q_len, block_q)
    assert k_len % block_k == 0, (k_len, block_k)

    qa, qb = q[:, :, :num_heads], q[:, :, num_heads + 1 :]
    ka, kb = k[:, :, :num_heads], q[:, :, num_heads + 1 :]
    va, vb = v[:, :, :num_heads], q[:, :, num_heads + 1 :]

    return pl.pallas_call(
        functools.partial(
            _fwd_kernel,
            sm_scale=sm_scale,
            causal=causal,
            block_k=block_k,
        ),
        grid=(
            pl.cdiv(num_heads, block_h),
            pl.cdiv(q_len, block_q),
        ),
        in_specs=[
            pl.BlockSpec((block_q, rank_q, block_h), lambda h, lq: (lq, 0, h)),  # qa
            pl.BlockSpec((block_q, rank_q, dim_k), lambda h, lq: (lq, 0, 0)),  # qb
            pl.BlockSpec((k_len, rank_k, block_h), lambda h, lq: (0, 0, h)),  # ka
            pl.BlockSpec((k_len, rank_k, dim_k), lambda h, lq: (0, 0, 0)),  # kb
            pl.BlockSpec((k_len, rank_k, block_h), lambda h, lq: (0, 0, h)),  # va
            pl.BlockSpec((k_len, rank_k, dim_k), lambda h, lq: (0, 0, 0)),  # vb
            pl.BlockSpec((block_q,), lambda h, lq: (lq,)),  # q_segment_ids
            pl.BlockSpec((k_len,), lambda h, lq: (0,)),  # kv_segment_ids
        ],
        out_specs=[
            pl.BlockSpec((block_q, block_h, dim_v), lambda h, lq: (lq, h, 0)),  # out
            pl.BlockSpec((block_q, block_h), lambda h, lq: (lq, h)),  # lse
        ],
        out_shape=[
            jax.ShapeDtypeStruct(shape=(q_len, num_heads, dim_v), dtype=q.dtype),  # out
            jax.ShapeDtypeStruct(shape=(q_len, num_heads), dtype=jnp.float32),  # lse
        ],
        compiler_params=dict(
            triton=dict(
                num_warps=4 if dim_k <= 64 else 8,
                num_stages=2,
            )
        ),
        debug=debug,
        interpret=interpret,
        name="tpa_fwd",
    )(
        qa=qa,
        qb=qb,
        ka=ka,
        kb=kb,
        va=va,
        vb=vb,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
    )
