"""Tensor Product Attention: Flash Pallas Kernels.

one issue im realizing with TPA in general:

    - if in order to gain benefits of reduced computation from non-materializing,
        we *must* use at least 16 heads per kernel. This could interact negatively
        with Tensor Parallelism, where we split heads across devices and then
        end up with generally a small number of heads per device.

    - this would generally push us towards more heads on each device, reducing
        tensor-parallel rank.

    - Perhaps there is a less explored design space in transformers where you have
      a really large number of heads with small head-dim?
"""

from __future__ import annotations

import functools
import dataclasses

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

import einops
from einops import einsum
from jaxtyping import Int, Float, Array


# TODO: support sparsity following Splash Attention pattern.
# from jax.experimental.pallas.ops.tpu import splash_attention


@dataclasses.dataclass
class BlockSizes:
    """Configures sizes of input array slices for kernels."""

    # Query and key/value block length sizes.
    block_q: int = 128
    block_kv: int = 128

    # The number of attention heads processed simultaneously in each program. This
    # defines the number of heads which share query/key factor inner products with the
    # "no materialize" strategy.
    #
    # NOTE: In order to actually benefit from reduction in FLOPs in the "nomat" case, we
    # must share a sufficient number of heads to satisfy the following condition:
    #
    #     rank_q * rank_k * (num_heads * d_head) <= num_heads * d_head
    #
    # Estimating from default sizes: rank_q: 6, rank_k: 2, d_head: 64
    # we have h * d / rq * rk * (d + h)
    # h 64: 2.66
    # h 32: 1.77
    # h 16: 1.00
    # h 8: 0.6
    #
    # So generally we need to share >16 heads per kernel program in order to benefit.
    # On TPUv4 architcture, we have 0.25MiB = 260k bytes of VREG.
    #
    # We need to cache intermediate arrays shaped (block_h, block_q, block_kv) within
    # the kernels. With bfloat6, these are the proportion of VREG occupied
    # block_h, block_q, block_kv, VREG%
    #       8      128       128    100%
    #      32       32        32     25%
    #
    # So generally we should reduce the q/kv block sizes down from the default values
    # of 128 in order to share
    block_h: int = 4


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 6, 7, 8, 9, 10, 11))
def tpa(
    q: Float[Array, "b lq rq (h + dk)"],
    k: Float[Array, "b lk rk (h + dk)"],
    v: Float[Array, "b lk rk (h + dv)"],
    num_heads: int,
    q_segment_ids: Int[Array, "b lq"] | None = None,
    kv_segment_ids: Int[Array, "b lk"] | None = None,
    sm_scale: float = 1.0,
    causal: bool = False,
    block_sizes: BlockSizes | None = None,
    nomat: bool = False,
    debug: bool = False,
    interpret: bool = False,
) -> Float[Array, "b lq h dv"]:
    """Tensor Product Attention - as Pallas kernel.

    Args:
        q: query factors concatneated along final dimension.
        k: key factors concatneated along final dimension.
        v: value factors concatneated along final dimension.
        q_segment_ids: query segment ids for packed samples.
        kv_segment_ids: key/value segment ids for packed samples.
        num_heads: number of attention heads.
        sm_scale: softmax scaling factor.
        causal: whether to apply causal attention mask.
        block_sizes: query/key/heads block sizes for partitioning works amongst threads.
        nomat: whether to avoid materializing query/key arrays even in the kernel. This
            strategy uses more memory in SRAM to reduce total FLOPS by sharing query/key
            inner products amongst heads in the same block.
        debug/interpret: debug flags for Pallas.
    Returns:
        Attention output array.
    """
    o, _ = _tpa_fwd(
        q,
        k,
        v,
        num_heads=num_heads,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        sm_scale=sm_scale,
        causal=causal,
        block_sizes=block_sizes,
        nomat=nomat,
        debug=debug,
        interpret=interpret,
    )
    return o


def _tpa_fwd(
    q: Float[Array, "b lq rq (h + dk)"],
    k: Float[Array, "b lk rk (h + dk)"],
    v: Float[Array, "b lk rk (h + dv)"],
    num_heads: int,
    q_segment_ids: Int[Array, "b lq"] | None,
    kv_segment_ids: Int[Array, "b lk"] | None,
    sm_scale: Float,
    causal: bool,
    block_sizes: BlockSizes | None,
    nomat: bool,
    debug: bool,
    interpret: bool,
) -> tuple[
    Float[Array, "b lq h dv"],
    tuple[Array | None, ...],
]:
    if block_sizes is None:
        block_sizes = BlockSizes()

    o, l = jax.vmap(  # over batch
        functools.partial(
            _fwd,
            num_heads=num_heads,
            causal=causal,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            nomat=nomat,
            debug=debug,
            interpret=interpret,
        ),
        in_axes=[
            0,
            0,
            0,
            0 if q_segment_ids is not None else None,
            0 if kv_segment_ids is not None else None,
        ],
        out_axes=0,
    )(q, k, v, q_segment_ids, kv_segment_ids)
    res = q, k, v, q_segment_ids, kv_segment_ids, o, l
    return o, res


def _tpa_bwd(
    num_heads: int,
    sm_scale: float,
    causal: bool,
    block_sizes: BlockSizes | None,
    nomat: bool,
    debug: bool,
    interpret: bool,
    residuals: tuple[Array, ...],
    do: Float[Array, "b lq h dv"],
) -> tuple[
    Float[Array, "b lq rq (h + dk)"],
    Float[Array, "b lk rk (h + dk)"],
    Float[Array, "b lk rk (h + dv)"],
    None,
    None,
]:
    """Backward pass implementation."""
    if block_sizes is None:
        block_sizes = BlockSizes()

    q, k, v, q_segment_ids, kv_segment_ids, o, l = residuals

    dq, dk, dv = jax.vmap(  # over batch
        functools.partial(
            _bwd,
            num_heads=num_heads,
            causal=causal,
            sm_scale=sm_scale,
            block_sizes=block_sizes,
            nomat=nomat,
            debug=debug,
            interpret=interpret,
        ),
        in_axes=[
            0,
            0,
            0,
            0 if q_segment_ids is not None else None,
            0 if kv_segment_ids is not None else None,
            0,
            0,
            0,
        ],
        out_axes=0,
    )(q, k, v, q_segment_ids, kv_segment_ids, o, l, do)
    return dq, dk, dv, None, None


tpa.defvjp(_tpa_fwd, _tpa_bwd)


def _lse(x: Float[Array, "... n"]) -> Float[Array, "..."]:
    """Logsumexp over final dimension."""
    m = jnp.max(x, axis=-1)
    l_ = m + jnp.log(jnp.sum(jnp.exp(x - m[..., None]), axis=-1))
    neginf = jnp.array(-jnp.inf, dtype=x.dtype)
    return jnp.where(jnp.isneginf(m), neginf, l_)


def _lse_accum(a: Float[Array, "..."], b: Float[Array, "..."]) -> Float[Array, "..."]:
    """Accumulat log-sum-exp.

    let a > b
    log(exp(a) + exp(b))
        = a + log(exp(a - a) + exp(b - a))
        = a + log(1 + exp(b - a))
        = a + softplus(b - a)
    """
    assert a.shape == b.shape, f"mismatching shapes: {a.shape}, {b.shape}"
    max = jnp.maximum(a, b)
    min = jnp.minimum(a, b)
    lse = max + jnp.log(1 + jnp.exp(min - max))
    neginf = jnp.array(-jnp.inf, dtype=a.dtype)
    return jnp.where(jnp.isneginf(max), neginf, lse)


def _fwd_kernel(
    qa_ref: Float[Array, "lq rq h"],
    qb_ref: Float[Array, "lq rq dk"],
    ka_ref: Float[Array, "lk rk h"],
    kb_ref: Float[Array, "lk rk dk"],
    va_ref: Float[Array, "lk rk h"],
    vb_ref: Float[Array, "lk rk dv"],
    q_segment_ids_ref: Int[Array, " lq"] | None,
    kv_segment_ids_ref: Int[Array, " lk"] | None,
    o_ref: Float[Array, "lq h dv"],
    l_ref: Float[Array, "lq h"],
    sm_scale: float,
    causal: bool,
    block_kv: int,
    block_h: int,
    nomat: bool,
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
        nomat: whether to avoid materializing q and k.

    Returns:
        nothing its a kernel.
    """
    rank_q = qa_ref.shape[1]
    rank_k = ka_ref.shape[1]
    block_q, num_heads, dv = o_ref.shape

    h_index: Int = pl.program_id(0)
    heads_slice = pl.dslice(h_index * block_h, block_h)

    q_index: Int = pl.program_id(1)

    qa = qa_ref[..., heads_slice]
    qb = qb_ref[...]
    q_segment_ids = q_segment_ids_ref[...] if q_segment_ids_ref is not None else None

    if not nomat:
        """
        NOTE: yeah looks like compiling einsum ain't gonna work HAHA

        jax._src.pallas.mosaic.lowering.LoweringException:
        NotImplementedError: Only 2D tensors supported in dot; received:
             [ShapedArray (bfloat16[128,6,11), ShapedArray(bfloat16[128,6,128])]

        Additional diagnostics:
        Failing jaxpr equation: a:bf16[128,1,128] = dot_general[
            dimension_numbers=(([1], [1]), ([0], [O]))
            preferred_element_type=bfloat16
        ] b c
        """
        q = einsum(qa, qb, "lq rq h, lq rq dk -> lq h dk") / rank_q
    else:
        q = None

    def _scan_fn(start_k: int, carry):
        o, l_prev = carry
        assert o.shape == (block_h, block_q, dv), o.shape
        assert l_prev.shape == (block_h, block_q), l_prev.shape

        kv_slice = pl.dslice(start_k * block_kv, block_kv)

        ka = ka_ref[kv_slice, slice(None), heads_slice]
        kb = kb_ref[kv_slice, slice(None), slice(None)]

        # Compute attention logits for all.
        x: Float[Array, "h lq lk"]
        if nomat:
            # NOTE: this strategy uses more SRAM per block but reduces the amount of
            # flops by sharding query/key inner products amongst heads.
            bb = einsum(qb, kb, "lq rq dk, lk rk dk -> lq lk rq rk")

            # NOTE: the following double scan is equivalent to the following reduciton
            # but uses less memory sinc jax.lax.scan without unrolling is guaranteed
            # to reuse input/output buffers. Prior implementation:
            # x = einsum(ka, bb, "lk rk h, lq lk rq rk -> lq lk rq h") / rank_k
            # x = einsum(qa, x, "lq rq h, lq lk rq h -> h lq lk") / rank_q

            def accum_logits(out, i: Int, j: Int) -> Float[Array, "h lq lk"]:
                return out + einsum(
                    qa[:, i], ka[:, j], bb[:, :, i, j], "lq h, lk h, lq lk -> h lq lk"
                )

            x, _ = jax.lax.scan(  # over rank_q
                lambda o, i: jax.lax.scan(  # over rank_k
                    lambda o, j: (accum_logits(o, i, j), None),
                    init=o,
                    xs=jnp.arange(rank_k),
                ),
                init=jnp.zeros(shape=(block_h, block_q, block_kv), dtype=ka.dtype),
                xs=jnp.arange(rank_q),
            )
            x /= rank_q * rank_k
            del bb
        else:
            # NOTE: this strategy materializes q and k in the highest memory cache
            # but still avoids materializing them in HBM.
            assert q is not None
            k = einsum(ka, kb, "lk rk h, lk rk dk -> lk h dk") / rank_k
            x = einsum(q, k, "lq h dk, lk h dk -> h lq lk")

        x *= sm_scale

        if kv_segment_ids_ref is not None:
            assert q_segment_ids is not None
            kv_segment_ids = kv_segment_ids_ref[kv_slice]
            mask = q_segment_ids[:, None] == kv_segment_ids[None, :]
        else:
            mask = None

        if causal:
            span_q = q_index * block_q + jnp.arange(block_q)
            span_k = start_k * block_kv + jnp.arange(block_kv)
            causal_mask = span_q[:, None] >= span_k[None, :]
            if mask is not None:
                mask &= causal_mask
            else:
                mask = causal_mask

        if mask is not None:
            x = jnp.where(mask[None, ...], x, -jnp.inf)

        # TODO: these two operaitons can probably be combined into one.
        l: Float[Array, "h lq"] = _lse_accum(l_prev, _lse(x))

        log_p: Float[Array, "h lq lk"] = x - l[..., None]
        p = jnp.exp(log_p)
        p = jnp.where(jnp.isneginf(l[..., None]), 0.0, p)  # no data yet -> zero.

        va = va_ref[kv_slice, slice(None), heads_slice]
        vb = vb_ref[kv_slice, slice(None), slice(None)]

        if nomat:

            def accum_out(
                o: Float[Array, "h lq dv"],
                i: Int,
            ) -> Float[Array, "h lq dv"]:
                pva = einsum(p, va[:, i], "h lq lk, lk h -> h lq lk")
                o_ = einsum(pva, vb[:, i], "h lq lk , lk dv -> h lq dv")
                o += o_ / rank_k
                return o

            o_, _ = jax.lax.scan(  # over rank_k
                lambda o, i: (accum_out(o, i), None),
                init=jnp.zeros(shape=(block_h, block_q, dv), dtype=va.dtype),
                xs=jnp.arange(rank_k),
            )

        else:
            # simple case just materialize V
            v = einsum(va, vb, "lk rk h, lk rk dv -> lk h dv") / rank_k
            o_ = einsum(p, v, "h lq lk, lk h dv -> h lq dv")

        assert o_.shape == (block_h, block_q, dv), o_.shape

        correction = jnp.exp(l_prev - l)
        correction = jnp.where(jnp.isneginf(l), 0, correction)
        o = correction[..., None] * o + o_
        return o, l

    if causal:
        num_kv_blocks = jax.lax.div(block_q * (q_index + 1) + block_kv - 1, block_kv)
    else:
        kv_len = ka_ref.shape[0]
        num_kv_blocks = pl.cdiv(kv_len, block_kv)

    o = jnp.zeros(shape=(block_h, block_q, dv), dtype=vb_ref.dtype)
    l = jnp.zeros(shape=(block_h, block_q), dtype=qa_ref.dtype) - jnp.inf
    o, l = jax.lax.fori_loop(0, num_kv_blocks, _scan_fn, (o, l))

    # TODO: consider defining empty attention to be zero instead of nan to avoid this
    # explicit case handling if this is slow.
    o = jnp.where(jnp.isneginf(l)[..., None], jnp.nan, o)

    # Store final output.
    o_ref[:, heads_slice, :] = einops.rearrange(o, "h l d -> l h d")
    l_ref[..., heads_slice] = einops.rearrange(l, "h l -> l h")


def _fwd(
    q: Float[Array, "lq rq (h + dk)"],
    k: Float[Array, "lk rk (h + dk)"],
    v: Float[Array, "lk rk (h + dv)"],
    q_segment_ids: Int[Array, " lq"] | None,
    kv_segment_ids: Int[Array, " lk"] | None,
    num_heads: int,
    sm_scale: Float,
    causal: bool,
    block_sizes: BlockSizes,
    nomat: bool,
    debug: bool,
    interpret: bool,
) -> tuple[
    Float[Array, "lq h dv"],
    Float[Array, "lq h"],
]:
    """Tensor Product Attention forward implementation.

    Args:
        ...
        block_q: size of query blocks
        block_kv: size of key/value blocks
        block_h: number of heads to compute in parallel within each thread. Note that
           this number defines the number of heads which share the same computation of
           query/key inner products across ranks.
    """
    q_len, rank_q, h_dk = q.shape
    k_len, rank_k, _ = v.shape
    assert k.shape == (k_len, rank_k, h_dk), k.shape

    qa, qb = q[:, :, :num_heads], q[:, :, num_heads:]
    ka, kb = k[:, :, :num_heads], k[:, :, num_heads:]
    va, vb = v[:, :, :num_heads], v[:, :, num_heads:]

    # head dimensions.
    dim_k = kb.shape[-1]
    dim_v = vb.shape[-1]

    block_q = min(block_sizes.block_q, q_len)
    block_kv = min(block_sizes.block_kv, k_len)
    block_h = min(block_sizes.block_h, num_heads)

    assert q_len % block_q == 0, (q_len, block_q)
    assert k_len % block_kv == 0, (k_len, block_kv)

    return pl.pallas_call(
        functools.partial(
            _fwd_kernel,
            sm_scale=sm_scale,
            causal=causal,
            block_kv=block_kv,
            block_h=block_h,
            nomat=nomat,
        ),
        grid=(
            # TODO: switch these around? idk
            pl.cdiv(num_heads, block_h),
            pl.cdiv(q_len, block_q),
        ),
        in_specs=[
            pl.BlockSpec((block_q, rank_q, num_heads), lambda h, lq: (lq, 0, 0)),  # qa
            pl.BlockSpec((block_q, rank_q, dim_k), lambda _, lq: (lq, 0, 0)),  # qb
            pl.BlockSpec((k_len, rank_k, num_heads), lambda h, _: (0, 0, 0)),  # ka
            pl.BlockSpec((k_len, rank_k, dim_k), lambda *_: (0, 0, 0)),  # kb
            pl.BlockSpec((k_len, rank_k, num_heads), lambda h, _: (0, 0, 0)),  # va
            pl.BlockSpec((k_len, rank_k, dim_v), lambda *_: (0, 0, 0)),  # vb
            (
                pl.BlockSpec((block_q,), lambda _, lq: (lq,))  # q_segment_ids
                if q_segment_ids is not None
                else None
            ),
            (
                pl.BlockSpec((k_len,), lambda *_: (0,))  # kv_segment_ids
                if kv_segment_ids is not None
                else None
            ),
        ],
        out_specs=[
            pl.BlockSpec((block_q, num_heads, dim_v), lambda h, lq: (lq, 0, 0)),  # out
            pl.BlockSpec((block_q, num_heads), lambda h, lq: (lq, 0)),  # lse
        ],
        out_shape=[
            jax.ShapeDtypeStruct(shape=(q_len, num_heads, dim_v), dtype=q.dtype),  # out
            jax.ShapeDtypeStruct(shape=(q_len, num_heads), dtype=q.dtype),  # lse
        ],
        compiler_params=dict(
            # TODO: tune these compiler params?
            # TODO: should be TPU params?
            triton=dict(
                num_warps=4 if dim_k <= 64 else 8,
                num_stages=2,
            )
        ),
        debug=debug,
        interpret=interpret,
        name="tpa_fwd",
    )(qa, qb, ka, kb, va, vb, q_segment_ids, kv_segment_ids)


def _bwd_kernel(
    qa_ref: Float[Array, "lq rq h"],
    qb_ref: Float[Array, "lq rq dk"],
    ka_ref: Float[Array, "lk rk h"],
    kb_ref: Float[Array, "lk rk dk"],
    va_ref: Float[Array, "lk rk h"],
    vb_ref: Float[Array, "lk rk dv"],
    q_segment_ids_ref: Int[Array, " lq"] | None,
    kv_segment_ids_ref: Int[Array, " lk"] | None,
    o_ref: Float[Array, "lq h dv"],
    l_ref: Float[Array, "lq h"],
    do_ref: Float[Array, "lq h dv"],
    delta_ref: Float[Array, "lq h"],
    dqa_alias: Float[Array, "lq rq h"],
    dqb_alias: Float[Array, "lq rq dk"],
    # outputs
    dqa_ref: Float[Array, "lq rq h"],
    dqb_ref: Float[Array, "lq rq dk"],
    dka_ref: Float[Array, "lk rk h"],
    dkb_ref: Float[Array, "lk rk dk"],
    dva_ref: Float[Array, "lk rk h"],
    dvb_ref: Float[Array, "lk rk dv"],
    # static
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_kv: int,
    block_h: int,
    nomat: bool,
):
    del dqa_alias, dqb_alias

    q_len, rank_q, _ = qa_ref.shape
    k_len, rank_k, dim_k = kb_ref.shape
    _, _, dim_v = o_ref.shape

    # grid is (heads, kv block)
    head_index: Int = pl.program_id(0)
    heads_slice = pl.dslice(start=head_index * block_h, size=block_h)

    start_k = pl.program_id(1)

    dka = jnp.zeros(shape=[block_kv, rank_k, block_h], dtype=ka_ref.dtype)
    dkb = jnp.zeros(shape=[block_kv, rank_k, dim_k], dtype=kb_ref.dtype)

    dva = jnp.zeros(shape=[block_kv, rank_k, block_h], dtype=va_ref.dtype)
    dvb = jnp.zeros(shape=[block_kv, rank_k, dim_v], dtype=vb_ref.dtype)

    ka = ka_ref[..., heads_slice]
    kb = kb_ref[...]

    va = va_ref[..., heads_slice]
    vb = vb_ref[...]

    kv_segment_ids = kv_segment_ids_ref[...] if kv_segment_ids_ref is not None else None

    if not nomat:
        k = einsum(ka, kb, "lk rk h, lk rk dk -> lk h dk") / rank_k
        v = einsum(va, vb, "lk rk h, lk rk dv -> lk h dv") / rank_k
    else:
        k = None
        v = None

    def _fn(start_q: int, carry):
        dka, dkb, dva, dvb = carry

        q_slice = pl.dslice(start=start_q * block_q, size=block_q)

        qa = qa_ref[q_slice, ..., heads_slice]
        qb = qb_ref[q_slice, ...]

        if q_segment_ids_ref is not None:
            q_segment_ids = q_segment_ids_ref[q_slice]
        else:
            q_segment_ids = None

        q = einsum(qa, qb, "lq rq h, lq rq dk -> lq h dk") / rank_q

        if nomat:
            bb = einsum(qb, kb, "lq rq dk, lk rk dk -> rq rk lq lk")

            # NOTE: the following double scan is equivalent to the following reduciton
            # but uses less memory sinc jax.lax.scan without unrolling is guaranteed
            # to reuse input/output buffers. Prior implementation:
            # x = einsum(ka, bb, "lk rk h, lq lk rq rk -> lq lk rq h") / rank_k
            # x = einsum(qa, x, "lq rq h, lq lk rq h -> h lq lk") / rank_q

            def accum_logits(out, i: Int, j: Int) -> Float[Array, "h lq lk"]:
                return out + einsum(
                    qa[:, i], ka[:, j], bb[i, j, :, :], "lq h, lk h, lq lk -> h lq lk"
                )

            x, _ = jax.lax.scan(  # over rank_q
                lambda o, i: jax.lax.scan(  # over rank_k
                    lambda o, j: (accum_logits(o, i, j), None),
                    init=o,
                    xs=jnp.arange(rank_k),
                ),
                init=jnp.zeros(shape=(block_h, block_q, block_kv), dtype=ka.dtype),
                xs=jnp.arange(rank_q),
            )
            x /= rank_q * rank_k

        else:
            bb = None
            x = einsum(q, k, "lq h dk, lk h dk -> h lq lk")

        assert x is not None
        x *= sm_scale

        if q_segment_ids is not None:
            assert kv_segment_ids is not None
            mask = q_segment_ids[:, None] == kv_segment_ids[None, :]
        else:
            mask = None

        if causal:
            span_q = start_q * block_q + jnp.arange(block_q)
            span_k = start_k * block_kv + jnp.arange(block_kv)
            causal_mask = span_q[:, None] >= span_k[None, :]
            if mask is not None:
                mask &= causal_mask
            else:
                mask = causal_mask
        if mask is not None:
            x = jnp.where(mask[None, ...], x, -jnp.inf)

        l = l_ref[q_slice, heads_slice]
        do = do_ref[q_slice, heads_slice, :]
        di = delta_ref[q_slice, heads_slice]

        l = einops.rearrange(l, "lq h -> h lq 1")
        p: Float[Array, "h lq lk"] = jnp.exp(x - l)
        p = jnp.where(jnp.isnan(p), 0, p)

        if nomat:
            # dv = p do
            # dva = p (do vb)
            # dvb = p (do va)
            def accumulate_dp(dp, dva, dvb, i: Int):
                dovb = einsum(do, vb[:, i], "lq h dv, lk dv -> h lq lk")

                dva = dva.at[:, i].add(
                    einsum(dovb, p, "h lq lk, h lq lk -> lk h") / rank_k
                )

                pva = einsum(p, va[:, i], "h lq lk, lk h -> h lq lk")
                dvb = dvb.at[:, i].add(
                    einsum(pva, do, "h lq lk, lq h dv -> lk dv") / rank_k
                )

                dp += einsum(dovb, va[:, i], "h lq lk, lk h -> h lq lk") / rank_k
                return dp, dva, dvb

            (dp, dva, dvb), _ = jax.lax.scan(  # over rank_k
                lambda dp_dva_dvb, i: (accumulate_dp(*dp_dva_dvb, i), None),
                init=(
                    einops.repeat(-di, "lq h -> h lq lk", lk=k_len),  # dp
                    dva,
                    dvb,
                ),
                xs=jnp.arange(rank_k),
            )
            ds = p * dp * sm_scale

            def accumulate_dq(
                dqa,
                dqb,
                i: Int,
                j: Int,
            ):
                """
                # dq = ds dk
                # dqa = ds (ka kb) qb = (ds ka) (kb qb)
                # dqb = ds (ka kb) qa = (ds qa ka) kb

                dq = einsum(ds, k, "h lq lk, lk h dk -> lq h dk")
                dqa = einsum(dq, qb, "lq h dk, lq rq dk -> lq rq h") / rank_q
                dqb = einsum(dq, qa, "lq h dk, lq rq h -> lq rq dk") / rank_q
                """
                assert bb is not None

                dska = einsum(ds, ka[:, j], "h lq lk, lk h -> h lq lk") / rank_k

                dqa = dqa.at[:, i].add(
                    einsum(dska, bb[i, j], "h lq lk, lq lk -> lq h") / rank_q
                )

                dsqaka = einsum(dska, qa[:, i], "h lq lk, lq h -> lq lk")
                dqb = dqb.at[:, i].add(
                    einsum(dsqaka, kb[:, j], "lq lk, lk dk -> lq dk") / rank_q
                )
                return (dqa, dqb), None

            (dqa, dqb), _ = jax.lax.scan(  # over rank_q
                lambda dqa_dqb, i: jax.lax.scan(  # over rank_k
                    lambda dqa_dqb, j: accumulate_dq(*dqa_dqb, i, j),  # type: ignore
                    init=dqa_dqb,
                    xs=jnp.arange(rank_k),
                ),
                init=jax.tree.map(jnp.zeros_like, (qa, qb)),
                xs=jnp.arange(rank_q),
            )

            def accumulate_dk(
                dka,
                dkb,
                i: Int,
                j: Int,
            ):
                assert bb is not None

                dsqa = einsum(ds, qa[:, i], "h lq lk, lq h -> h lq lk") / rank_q
                dka = dka.at[:, j].add(
                    einsum(dsqa, bb[i, j], "h lq lk, lq lk -> lk h") / rank_k
                )

                dskaqa = einsum(dsqa, ka[:, j], "h lq lk, lk h -> lq lk")
                dkb = dkb.at[:, j].add(
                    einsum(dskaqa, qb[:, i], "lq lk, lq dk -> lk dk") / rank_k
                )
                return (dka, dkb), None

            (dka, dkb), _ = jax.lax.scan(  # over rank_q
                lambda dka_dkb, i: jax.lax.scan(  # over rank_k
                    lambda dka_dkb, j: accumulate_dk(*dka_dkb, i, j),
                    init=dka_dkb,
                    xs=jnp.arange(rank_k),
                ),
                init=(dka, dkb),
                xs=jnp.arange(rank_q),
            )

        else:
            assert v is not None
            dv = einsum(p, do, "h lq lk, lq h dv -> lk h dv")
            dva += einsum(dv, vb, "lk h dv, lk rk dv -> lk rk h") / rank_k
            dvb += einsum(dv, va, "lk h dv, lk rk h -> lk rk dv") / rank_k

            dp = einops.repeat(-di, "lq h -> h lq lk", lk=k_len)
            dp += einsum(do, v, "lq h dv, lk h dv -> h lq lk")

            ds = p * dp
            ds *= sm_scale

            assert q is not None
            dk = einsum(ds, q, "h lq lk, lq h dk -> lk h dk")
            dka += einsum(dk, kb, "lk h dk, lk rk dk -> lk rk h") / rank_k
            dkb += einsum(dk, ka, "lk h dk, lk rk h -> lk rk dk") / rank_k

            assert k is not None
            dq = einsum(ds, k, "h lq lk, lk h dk -> lq h dk")
            dqa = einsum(dq, qb, "lq h dk, lq rq dk -> lq rq h") / rank_q
            dqb = einsum(dq, qa, "lq h dk, lq rq h -> lq rq dk") / rank_q

        # NOTE: on TPU no need for atomic add as long as we'll be processing all query
        # chunks sequentially. This might actually turn into an issue if we have so
        # many shared heads that we have insufficient parallelism to feed both cores
        # of megacore TPUs ...
        # As long as we # have micromatch size of 2 might be ok lmao
        pl.atomic_add(dqa_ref, (q_slice, slice(None), heads_slice), dqa)
        pl.atomic_add(dqb_ref, (q_slice, slice(None), slice(None)), dqb)

        return dka, dkb, dva, dvb

    end = start_k if causal else q_len
    num_blocks = pl.cdiv(end, block_q)

    dka, dkb, dva, dvb = jax.lax.fori_loop(0, num_blocks, _fn, (dka, dkb, dva, dvb))

    # Write outputs to HBM.
    dka_ref[..., heads_slice] = dka.astype(dka_ref.dtype)
    dkb_ref[...] = dkb.astype(dkb_ref.dtype)
    dva_ref[..., heads_slice] = dva.astype(dva_ref.dtype)
    dvb_ref[...] = dvb.astype(dvb_ref.dtype)


def _bwd(
    q: Float[Array, "lq rq (h + dk)"],
    k: Float[Array, "lk rk (h + dk)"],
    v: Float[Array, "lk rk (h + dv)"],
    q_segment_ids: Int[Array, " lq"] | None,
    kv_segment_ids: Int[Array, " lk"] | None,
    o: Float[Array, "lq h dv"],
    l: Float[Array, "lq h"],
    do: Float[Array, "lq h dv"],
    num_heads: int,
    sm_scale: Float,
    causal: bool,
    block_sizes: BlockSizes,
    nomat: bool,
    debug: bool,
    interpret: bool,
) -> tuple[
    Float[Array, "lq rq (h + dk)"],
    Float[Array, "lk rk (h + dk)"],
    Float[Array, "lk rk (h + dv)"],
]:
    q_len, rank_q, h_dk = q.shape
    k_len, rank_k, h_dv = v.shape
    assert k.shape == (k_len, rank_k, h_dk), k.shape
    assert h_dk > num_heads
    assert h_dv > num_heads

    qa, qb = q[:, :, :num_heads], q[:, :, num_heads:]
    ka, kb = k[:, :, :num_heads], k[:, :, num_heads:]
    va, vb = v[:, :, :num_heads], v[:, :, num_heads:]

    # head dimensions.
    dim_k = kb.shape[-1]
    dim_v = vb.shape[-1]

    block_q = min(block_sizes.block_q, q_len)
    block_kv = min(block_sizes.block_kv, k_len)
    block_h = min(block_sizes.block_h, num_heads)

    assert q_len % block_q == 0, (q_len, block_q)
    assert k_len % block_kv == 0, (k_len, block_kv)

    delta = jax.vmap(
        functools.partial(
            _precompute_delta,
            block_q=block_q,
            debug=debug,
            interpret=interpret,
        ),
        in_axes=[1, 1, 1],
        out_axes=1,
    )(o, do, l)

    # initialize dq and use input aliasing https://github.com/jax-ml/jax/discussions/23272
    # since we'll be writing to dq in parallel with atomic_add
    # NOTE: On GPU these will need to be float 32
    dqa = jnp.zeros_like(qa)
    dqb = jnp.zeros_like(qb)

    dqa, dqb, dka, dkb, dva, dvb = pl.pallas_call(
        functools.partial(
            _bwd_kernel,
            sm_scale=sm_scale,
            causal=causal,
            block_q=block_q,
            block_kv=block_kv,
            block_h=block_h,
            nomat=nomat,
        ),
        grid=(
            pl.cdiv(num_heads, block_h),
            pl.cdiv(k_len, block_kv),
        ),
        in_specs=[
            pl.BlockSpec((q_len, rank_q, num_heads), lambda h, lk: (0, 0, 0)),  # qa
            pl.BlockSpec((q_len, rank_q, dim_k), lambda _, lk: (0, 0, 0)),  # qb
            pl.BlockSpec((block_kv, rank_k, num_heads), lambda h, lk: (lk, 0, 0)),  # ka
            pl.BlockSpec((block_kv, rank_k, dim_k), lambda _, lk: (lk, 0, 0)),  # kb
            pl.BlockSpec((block_kv, rank_k, num_heads), lambda h, lk: (lk, 0, 0)),  # va
            pl.BlockSpec((block_kv, rank_k, dim_v), lambda _, lk: (lk, 0, 0)),  # vb
            # control
            (
                pl.BlockSpec((q_len,), lambda _, lk: (0,))  # q_segment_ids
                if q_segment_ids is not None
                else None
            ),
            (
                pl.BlockSpec((block_kv,), lambda _, lk: (lk,))  # kv_segment_ids
                if kv_segment_ids is not None
                else None
            ),
            # outputs
            pl.BlockSpec((q_len, num_heads, dim_v), lambda h, lk: (0, 0, 0)),  # o
            pl.BlockSpec((q_len, num_heads), lambda h, lk: (0, 0)),  # l
            pl.BlockSpec((q_len, num_heads, dim_v), lambda h, lk: (0, 0, 0)),  # do
            pl.BlockSpec((q_len, num_heads), lambda h, lk: (0, 0)),  # delta
            # aliases: dqa, dqb
            pl.BlockSpec((block_q, rank_q, num_heads), lambda h, lq: (lq, 0, 0)),  # dqa
            pl.BlockSpec((block_q, rank_q, dim_k), lambda _, lq: (lq, 0, 0)),  # dqb
        ],
        out_specs=[
            pl.BlockSpec((q_len, rank_q, num_heads), lambda h, lk: (0, 0, 0)),  # dqa
            pl.BlockSpec((q_len, rank_q, dim_k), lambda _, lk: (0, 0, 0)),  # dqb
            pl.BlockSpec(
                (block_kv, rank_k, num_heads), lambda h, lk: (lk, 0, 0)
            ),  # dka
            pl.BlockSpec((block_kv, rank_k, dim_k), lambda _, lk: (lk, 0, 0)),  # dkb
            pl.BlockSpec(
                (block_kv, rank_k, num_heads), lambda h, lk: (lk, 0, 0)
            ),  # dva
            pl.BlockSpec((block_kv, rank_k, dim_v), lambda _, lk: (lk, 0, 0)),  # dvb
        ],
        out_shape=[
            jax.ShapeDtypeStruct(shape=qa.shape, dtype=qa.dtype),
            jax.ShapeDtypeStruct(shape=qb.shape, dtype=qb.dtype),
            jax.ShapeDtypeStruct(shape=ka.shape, dtype=ka.dtype),
            jax.ShapeDtypeStruct(shape=kb.shape, dtype=kb.dtype),
            jax.ShapeDtypeStruct(shape=va.shape, dtype=va.dtype),
            jax.ShapeDtypeStruct(shape=vb.shape, dtype=vb.dtype),
        ],
        input_output_aliases={
            12 if q_segment_ids is not None else 10: 0,  # dqa
            13 if q_segment_ids is not None else 11: 1,  # dqb
        },
        compiler_params=dict(
            triton=dict(  # TODO: need to adjust this???
                num_warps=4 if dim_k <= 64 else 8,
                num_stages=2,
            )
        ),
        debug=debug,
        interpret=interpret,
        name="tpa_bwd",
    )(qa, qb, ka, kb, va, vb, q_segment_ids, kv_segment_ids, o, l, do, delta, dqa, dqb)

    dq = jnp.concatenate([dqa, dqb], axis=-1)
    dk = jnp.concatenate([dka, dkb], axis=-1)
    dv = jnp.concatenate([dva, dvb], axis=-1)

    return dq, dk, dv


def _precompute_delta(
    out: Float[Array, "lq dv"],
    do: Float[Array, "lq dv"],
    lse: Float[Array, " lq"],
    block_q: int,
    debug: bool,
    interpret: bool,
) -> Float[Array, " lq"]:
    """Precompute delta for backward pass."""
    seq_len, dv = out.shape

    def kernel(out_ref, dout_ref, delta_ref):
        o = out_ref[...].astype(jnp.float32)
        do = dout_ref[...].astype(jnp.float32)
        delta = jnp.nansum(o * do, axis=1)
        delta_ref[...] = delta.astype(delta_ref.dtype)

    return pl.pallas_call(
        kernel,
        grid=(pl.cdiv(seq_len, block_q),),
        in_specs=[
            pl.BlockSpec((block_q, dv), lambda l: (l, 0)),
            pl.BlockSpec((block_q, dv), lambda l: (l, 0)),
        ],
        out_specs=pl.BlockSpec((block_q,), lambda l: (l,)),
        compiler_params=dict(triton=dict(num_warps=4, num_stages=3)),
        out_shape=jax.ShapeDtypeStruct(lse.shape, lse.dtype),
        debug=debug,
        interpret=interpret,
        name="precompute_delta",
    )(out, do)
