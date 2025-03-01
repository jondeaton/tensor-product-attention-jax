"""Tensor Product Attention Pallas Kernels."""

from __future__ import annotations

import functools
from typing import Any
import einops

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from jaxtyping import Int, Float, Array


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def tpa(
    q: Float[Array, "b lq rq (h + dh)"],
    k: Float[Array, "b lk rk (h + dh)"],
    v: Float[Array, "b lk rk (h + dv)"],
    q_segment_ids: Int[Array, "b lq"],
    kv_segment_ids: Int[Array, "b lk"],
    num_heads: int,
    sm_scale: float = 1.0,
    causal: bool = False,
    debug: bool = False,
    interpret: bool = False,
) -> Float[Array, "b lq h dv"]:
    """Tensor Product Attention kernel."""
    batch_size, q_len, rank_q, dim_k = q.shape
    _, k_len, rank_k, dim_v = v.shape

    o, _ = jax.vmap(  # over batch
        functools.partial(
            _fwd,
            num_heads=num_heads,
            causal=causal,
            sm_scale=sm_scale,
            debug=debug,
            interpret=interpret,
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
    assert a.shape == b.shape, f"mismatching shapes: {a.shape}, {b.shape}"
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
    l_ref: Float[Array, "lq h"],
    sm_scale: float,
    causal: bool,
    block_kv: int,
):
    """TPA forward pallas kernel.

    the main question here is should we parallelize across query heads or ranks?
        - if we parallelize across heads, then we can't share the inner products across
          heads and will have to recompute. on the other hand

    this kernel parallelizes across query ranks

    TODO: parallelize across query ranks and atomic add into output?
        - atomic add would be slow on gpu
        - atomic add wouldn't be necessary on TPU

    Args:
        q(a|b)_ref: factorized queries (block)
        k(a|b)_ref: factorized keys
        v(a|b)_ref: factorized values
        q_segment_ids_ref: segment ids for queries (block)
        kv_segment_ids_ref: segment ids for keys and values.
        o_ref: output array reference
        l_ref: log-sum-exp array reference to write normalization factors.
        sm_scale: softmax scale
        causal: whether to apply a causal kernel.
        block_kv: block size to split key/value arrays

    Returns:
        nothing its a kernel.
    """
    rank_q = qa_ref.shape[1]
    rank_k = ka_ref.shape[1]
    block_q, block_h, dv = o_ref.shape

    start_q = pl.program_id(1)
    qa = qa_ref[...]
    qb = qb_ref[...]
    q_segment_ids = q_segment_ids_ref[...]

    def _scan_fn(start_k: int, carry):
        o, l_prev = carry
        assert o.shape == (block_h, block_q, dv), o.shape
        assert l_prev.shape == (block_h, block_q), l_prev.shape

        kv_slice = pl.dslice(start_k * block_kv, block_kv)

        ka = ka_ref[kv_slice, slice(None), slice(None)]
        kb = kb_ref[kv_slice, slice(None), slice(None)]

        # jax.debug.print("{x}", x=kb)

        bb = einops.einsum(qb, kb, "lq rq dk, lk rk dk -> lq lk rq rk")

        x = einops.einsum(ka, bb, "lk rk h, lq lk rq rk -> lq lk rq h") / rank_k
        x = einops.einsum(qa, x, "lq rq h, lq lk rq h -> h lq lk") / rank_q
        # x = einops.einsum(qa, ka, bb, "lq rq h, lk rk h, lq lk rk -> h lq lk")
        x *= sm_scale

        kv_segment_ids = kv_segment_ids_ref[kv_slice]
        mask = q_segment_ids[:, None] == kv_segment_ids[None, :]
        if causal:
            span_q = start_q * block_q + jnp.arange(block_q)
            span_k = start_k * block_kv + jnp.arange(block_kv)
            causal_mask = span_q[:, None] >= span_k[None, :]
            mask &= causal_mask
        x = jnp.where(mask[None, ...], x, -jnp.inf)

        m = jnp.max(x, axis=-1)
        l_ = m + jnp.log(jnp.sum(jnp.exp(x - m[..., None]), axis=-1))
        assert l_.shape == (block_h, block_q), l_.shape

        l = lse_combine(l_prev, l_)

        log_p = x - l[..., None]
        p = jnp.exp(log_p)

        va = va_ref[kv_slice, slice(None), slice(None)]
        vb = vb_ref[kv_slice, slice(None), slice(None)]

        v = einops.einsum(va, vb, "lk rk h, lk rk dv -> lk h dv") / rank_k
        o_ = einops.einsum(p, v, "h lq lk, lk h dv -> h lq dv")
        assert o_.shape == (block_h, block_q, dv), o_.shape

        # honestly idk if this can fit into smem
        # pva = einops.einsum(p, va, "h lq lk, lk rk h -> h lq lk rk")
        # o_ = einops.einsum(pva, vb, "h lq lk rk, lk rk dv -> h lq dv")

        o = jnp.exp(l_prev - l)[..., None] * o + o_

        return o, l

    if causal:
        num_kv_blocks = jax.lax.div(block_q * (start_q + 1) + block_kv - 1, block_kv)
    else:
        kv_len = ka_ref.shape[0]
        num_kv_blocks = pl.cdiv(kv_len, block_kv)

    o = jnp.zeros(shape=(block_h, block_q, dv), dtype=jnp.float32)
    l = jnp.zeros(shape=(block_h, block_q), dtype=jnp.float32) - jnp.inf
    o, l = jax.lax.fori_loop(0, num_kv_blocks, _scan_fn, (o, l))

    # store final output
    o_ref[...] = einops.rearrange(o, "h l d -> l h d")
    l_ref[...] = einops.rearrange(l, "h l -> l h")


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
    block_kv: int = 128,
    block_h: int = 4,
    debug: bool = False,
    interpret: bool = False,
) -> tuple[
    Float[Array, "lq h dv"],
    Float[Array, "lq h"],
]:
    """Forward pass through TPA.

    Args:
        ...
        block_q: size of query blocks
        block_kv: size of key/value blocks
        block_h: number of heads to compute in parallel within each thread. Note that
           this number defines the number of heads which share the same computation of
           query/key inner products across ranks.
    """
    q_len, rank_q, h_dk = q.shape
    k_len, rank_k, h_dv = v.shape
    assert k.shape == (k_len, rank_k, h_dk), k.shape

    qa, qb = q[:, :, :num_heads], q[:, :, num_heads:]
    ka, kb = k[:, :, :num_heads], k[:, :, num_heads:]
    va, vb = v[:, :, :num_heads], v[:, :, num_heads:]

    # head dimensions.
    dim_k = kb.shape[-1]
    dim_v = vb.shape[-1]

    block_q = min(block_q, q_len)
    block_kv = min(block_kv, k_len)
    block_h = min(block_h, num_heads)

    assert q_len % block_q == 0, (q_len, block_q)
    assert k_len % block_kv == 0, (k_len, block_kv)

    return pl.pallas_call(
        functools.partial(
            _fwd_kernel,
            sm_scale=sm_scale,
            causal=causal,
            block_kv=block_kv,
        ),
        grid=(
            pl.cdiv(num_heads, block_h),
            pl.cdiv(q_len, block_q),
        ),
        in_specs=[
            pl.BlockSpec((block_q, rank_q, block_h), lambda h, lq: (lq, 0, h)),  # qa
            pl.BlockSpec((block_q, rank_q, dim_k), lambda _, lq: (lq, 0, 0)),  # qb
            pl.BlockSpec((k_len, rank_k, block_h), lambda h, _: (0, 0, h)),  # ka
            pl.BlockSpec((k_len, rank_k, dim_k), lambda *_: (0, 0, 0)),  # kb
            pl.BlockSpec((k_len, rank_k, block_h), lambda h, _: (0, 0, h)),  # va
            pl.BlockSpec((k_len, rank_k, dim_v), lambda *_: (0, 0, 0)),  # vb
            pl.BlockSpec((block_q,), lambda _, lq: (lq,)),  # q_segment_ids
            pl.BlockSpec((k_len,), lambda *_: (0,)),  # kv_segment_ids
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
    )(qa, qb, ka, kb, va, vb, q_segment_ids, kv_segment_ids)


def _bwd(*args, **kwargs):
    raise NotImplementedError("Backward pass not implemented.")


tpa.defvjp(_fwd, _bwd)
