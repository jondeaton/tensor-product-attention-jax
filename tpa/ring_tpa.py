"""Tensor Product Ring Attention.

This is an implementation of Tensor Product Attention (TPA) in a Ring Attention Context.

Plans also include:
    - local / sliding window variant
    - dillated variant
"""

from __future__ import annotations
from typing import Callable
import functools

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map

import einops
from jaxtyping import Float, Int, Array, PyTree

from .kernels import tpa


def _rotate_block(x: PyTree, axis_name: str, axis_size: int) -> PyTree:
    """Rotates an array block (ie query/key block) along the sharding axis.

    Args:
        x: array block to rotate
        axis_name: the name of the axis along which the array is sharded.
        axis_size: number of blocks/shards/slices that cut the axis.
    Returns:
        rotated block same shape as input block.
    """
    return jax.lax.ppermute(
        x,
        axis_name,
        perm=[(i, (i + 1) % axis_size) for i in range(axis_size)],
    )


def _ring_tpa_fwd(
    q: Float[Array, "b lq rq (h + dh)"],
    k: Float[Array, "b lk rk (h + dh)"],
    v: Float[Array, "b lk rk (h + dv)"],
    q_segment_ids: Int[Array, "b lq"],
    kv_segment_ids: Int[Array, "b lk"],
    axis_name: str,
    num_heads: int,
    sm_scale: float,
    causal: bool,
    fwd_block_fn: Callable[..., tuple[Array, ...]],
    bwd_block_fn: Callable[..., tuple[Array, ...]],
) -> tuple[
    Float[Array, "b lq h dv"],
    tuple[Array, ...],  # residuals
]:
    """Forward ring TPA."""
    del bwd_block_fn

    batch, q_len, _, h_dk = q.shape
    _, kv_len, rank_k, h_dv = v.shape
    dim_k = h_dk - num_heads
    dim_v = h_dv - num_heads
    assert k.shape == (batch, kv_len, rank_k, num_heads + dim_k)

    axis_size = jax.lax.psum(1, axis_name)
    rotate = functools.partial(_rotate_block, axis_name=axis_name, axis_size=axis_size)

    def scan_fn(carry, i: Int):
        del i
        o, l, k, v, kv_segment_ids = carry

        o_, l_ = fwd_block_fn(
            q,
            k,
            v,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
        )
        l_: Float[Array, "b lq h"]

        l_next = tpa._lse_accum(l, l_)

        # accumulator corrections, accounting for blocks with no data.
        o *= jnp.exp(l - l_next)[..., None]
        o = jnp.nan_to_num(o, nan=0)

        o_ *= jnp.exp(l_ - l_next)[..., None]
        o_ = jnp.nan_to_num(o_, nan=0)

        o += o_

        # no_data = jnp.isneginf(l_next)
        # o = jnp.where(no_data[..., None], 0, o)

        k, v, kv_segment_ids = rotate([k, v, kv_segment_ids])
        return (o, l_next, k, v, kv_segment_ids), None

    o = jnp.zeros(shape=(batch, q_len, num_heads, dim_v), dtype=v.dtype)
    l = jnp.full(shape=(batch, q_len, num_heads), dtype=q.dtype, fill_value=-jnp.inf)

    (o, l, k, v, kv_segment_ids), _ = jax.lax.scan(
        scan_fn,
        init=(o, l, k, v, kv_segment_ids),
        xs=jnp.arange(axis_size),
    )

    # Empty attention has value nan.
    o = jnp.where(jnp.isneginf(l[..., None]), jnp.nan, o)

    residuals = q, k, v, o, l, q_segment_ids, kv_segment_ids
    return o, residuals


def _ring_tpa_bwd(
    axis_name: str,
    num_heads: int,
    sm_scale: float,
    causal: bool,
    fwd_block_fn: Callable[..., tuple[Array, ...]],
    bwd_block_fn: Callable[..., tuple[Array, ...]],
    residuals: tuple[Array, ...],
    do: Float[Array, "b lq h dv"],
) -> tuple[
    Float[Array, "b lq rq (h + dk)"],
    Float[Array, "b lk rk (h + dk)"],
    Float[Array, "b lk rk (h + dv)"],
    None,
    None,
]:
    """Ring TPA backward pass implementation."""
    del fwd_block_fn
    q, k, v, o, l, q_segment_ids, kv_segment_ids = residuals

    batch, q_len, rank_q, h_dk = q.shape
    _, kv_len, rank_k, h_dv = v.shape
    dim_k = h_dk - num_heads
    assert k.shape == (batch, kv_len, rank_k, num_heads + dim_k)

    axis_size = jax.lax.psum(1, axis_name)
    rotate = functools.partial(_rotate_block, axis_name=axis_name, axis_size=axis_size)

    def scan_fn(carry, _):
        dq, dk, dv, k, v, kv_segment_ids = carry
        dq_, dk_, dv_ = bwd_block_fn(
            q,
            k,
            v,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            o=o,
            l=l,
            do=do,
        )
        dq += dq_
        dk += dk_
        dv += dv_

        dk, dv, k, v, kv_segment_ids = rotate([dk, dv, k, v, kv_segment_ids])
        carry = dq, dk, dv, k, v, kv_segment_ids
        return carry, None

    dq = jnp.zeros_like(q)
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    (dq, dk, dv, _, _, _), _ = jax.lax.scan(
        scan_fn,
        init=(dq, dk, dv, k, v, kv_segment_ids),
        xs=jnp.arange(axis_size),
    )
    return dq, dk, dv, None, None


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 11))
def _ring_tpa(
    q: Float[Array, "b lq rq (h + dh)"],
    k: Float[Array, "b lk rk (h + dh)"],
    v: Float[Array, "b lk rk (h + dv)"],
    q_segment_ids: Int[Array, "b lq"],
    kv_segment_ids: Int[Array, "b lk"],
    axis_name: str,
    num_heads: int,
    sm_scale: float,
    causal: bool,
    fwd_block_fn: Callable,
    bwd_block_fn: Callable,
) -> Float[Array, "b lq h dv"]:
    """Ring attention implementation."""
    o, _ = _ring_tpa_fwd(
        q,
        k,
        v,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        axis_name=axis_name,
        num_heads=num_heads,
        sm_scale=sm_scale,
        causal=causal,
        fwd_block_fn=fwd_block_fn,
        bwd_block_fn=bwd_block_fn,
    )
    return o


_ring_tpa.defvjp(_ring_tpa_fwd, _ring_tpa_bwd)


def ring_tpa(
    q: Float[Array, "b lq rq (h + dk)"],
    k: Float[Array, "b lk rk (h + dk)"],
    v: Float[Array, "b lk rk (h + dv)"],
    q_segment_ids: Int[Array, "b lq"],
    kv_segment_ids: Int[Array, "b lk"],
    axis_name: str,
    num_heads: int,
    sm_scale: float = 1.0,
    causal: bool = False,
    impl: str = "pallas",  # "xla", "pallas"
    debug: bool = False,
    interpret: bool = False,
) -> Float[Array, "b lq h dv"]:
    """Tensor Product Ring Attention - general.

    Args:
        q: single block of factored queries sharded along length dimension.
        k: single block of factored keys sharded along length dimension.
        v: single block of factored values sharded along length dimension.
        (q|kv)_segment_ids: single block of query and key/value segment ids.
        axis_name: name of device mesh axis along which sequences are sharded.
        num_heads: number of attention heads.
        sm_scale: optional softmax scale, defaults to 1/sqrt(dk) if unspecified.
        causal: if attention is causal.
    Returns:
        Attention output (sharded).
    """
    match impl:
        case "xla":
            raise NotImplementedError()
        case "pallas":
            block_sizes = tpa.BlockSizes()
            fwd_block_fn = jax.vmap(  # over batch
                functools.partial(
                    tpa._fwd,
                    num_heads=num_heads,
                    causal=causal,
                    sm_scale=sm_scale,
                    block_sizes=block_sizes,
                    nomat=False,
                    debug=debug,
                    interpret=interpret,
                ),
                in_axes=0,
                out_axes=0,
            )

            bwd_block_fn = jax.vmap(  # over batch
                functools.partial(
                    tpa._bwd,
                    num_heads=num_heads,
                    causal=causal,
                    sm_scale=sm_scale,
                    block_sizes=block_sizes,
                    nomat=False,
                    debug=debug,
                    interpret=interpret,
                ),
                in_axes=0,
                out_axes=0,
            )

        case _:
            raise ValueError(f"Unknown implementaiton: {impl}")

    return _ring_tpa(
        q,
        k,
        v,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        num_heads=num_heads,
        axis_name=axis_name,
        sm_scale=sm_scale,
        causal=causal,
        fwd_block_fn=fwd_block_fn,
        bwd_block_fn=bwd_block_fn,
    )


def ring_tp_self_attention(
    q: Float[jax.Array, "b l rq (h + dk)"],
    k: Float[Array, "b l rk (h + dk)"],
    v: Float[Array, "b l rk (h + dv)"],
    q_segment_ids: Int[Array, "b l"],
    kv_segment_ids: Int[Array, "b l"],
    mesh: jax.sharding.Mesh,
    pspec: jax.sharding.PartitionSpec,
    num_heads: int,
    sm_scale: float = 1.0,
    causal: bool = False,
    impl: str = "pallas",  # "xla", "pallas"
    debug: bool = False,
    interpret: bool = False,
) -> Float[Array, "b l h dv"]:
    """Tensor Product Ring Self-Attention.

    Args:
        q|k|v: query/key/value factored and sharded along length.
        (q|kv)_segment_ids: query and key/value segment_id tensors sharded along length.
        mesh: device mesh over which arrays are sharded
        pspec: partition spec describing the sharding of all arrays, example:
            PartitionSpec(None, "SP", None, None)
        ...
    Returns:
        ...
    """
    return shard_map(
        functools.partial(
            ring_tpa,
            axis_name=pspec[1],  # grab axis name
            num_heads=num_heads,
            causal=causal,
            sm_scale=sm_scale,
            impl=impl,
            debug=debug,
            interpret=interpret,
        ),
        mesh=mesh,
        in_specs=(
            pspec,  # q
            pspec,  # k
            pspec,  # v
            pspec,  # q_segment_ids
            pspec,  # kv_segment_ids
        ),
        out_specs=pspec,  # type: ignore
        check_rep=False,
    )(q, k, v, q_segment_ids, kv_segment_ids)
