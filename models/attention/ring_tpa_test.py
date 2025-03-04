"""Test for tensor-product ring attention."""

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
from . import ring_tpa

from models.attention.mha import mha

flags = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = flags + " --xla_force_host_platform_device_count=8"

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

    # easiest implementation is fully materialize q,k,v
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
@pytest.mark.parametrize("nomat", [False])
@pytest.mark.parametrize(
    "batch_size,lq,lk,h,dk,dv",
    [
        (1, 16, 16, 1, 4, 6),
        (2, 1024, 128, 4, 32, 8),
        (2, 128, 1024, 4, 32, 8),
    ],
)
def test_ring_tpa_fwd(
    batch_size: int,
    rank_q: int,
    rank_k: int,
    nomat: bool,
    lq: int,
    lk: int,
    h: int,
    dk: int,
    dv: int,
):
    key = jax.random.PRNGKey(0)

    keys = jax.random.split(key, 3)
    q = jax.random.uniform(keys[0], shape=(batch_size, lq, rank_q, (h + dk)))
    k = jax.random.uniform(keys[1], shape=(batch_size, lk, rank_k, (h + dk)))
    v = jax.random.uniform(keys[2], shape=(batch_size, lk, rank_k, (h + dv)))

    q_segment_ids = einops.repeat(jnp.arange(lq) // 42, "l -> b l", b=batch_size)
    kv_segment_ids = einops.repeat(jnp.arange(lk) // 42, "l -> b l", b=batch_size)

    out_ref = jax.jit(reference, static_argnames=["num_heads"])(
        q,
        k,
        v,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        num_heads=h,
    )
    assert out_ref.shape == (batch_size, lq, h, dv), out_ref.shape

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh(
        mesh_shape=(1, 8),
        devices=devices,
    )
    mesh = Mesh(device_mesh, axis_names=("dp", "sp"))
    sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))
    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)
    q_segment_ids = jax.device_put(q_segment_ids, sharding)
    kv_segment_ids = jax.device_put(kv_segment_ids, sharding)

    out = jax.jit(
        shard_map(
            functools.partial(
                ring_tpa.ring_tpa,
                axis_name="sp",
                num_heads=h,
                debug=False,
                interpret=True,
            ),
            mesh=mesh,
            in_specs=(
                PartitionSpec("dp", "sp", None, None),  # q
                PartitionSpec("dp", "sp", None, None),  # k
                PartitionSpec("dp", "sp", None, None),  # v
                PartitionSpec("dp", "sp"),  # q_segment_ids
                PartitionSpec("dp", "sp"),  # kv_segment_ids
            ),
            out_specs=PartitionSpec("dp", "sp", None, None),
            check_rep=False,
        )
    )(q, k, v, q_segment_ids, kv_segment_ids)

    assert out.shape == (batch_size, lq, h, dv), out_ref.shape
    np.testing.assert_allclose(out, out_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("rank_q", [1, 6])
@pytest.mark.parametrize("rank_k", [1, 4])
@pytest.mark.parametrize("nomat", [False, True])
@pytest.mark.parametrize(
    "batch_size,lq,lk,h,dim_k,dim_v",
    [
        (1, 8, 8, 1, 4, 6),
        (2, 1024, 128, 2, 3, 5),
        (2, 128, 1024, 2, 3, 5),
        (1, 1024, 1024, 2, 3, 5),
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
):
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)

    q = jax.random.uniform(keys[0], shape=(batch_size, lq, rank_q, (h + dim_k)))
    k = jax.random.uniform(keys[1], shape=(batch_size, lk, rank_k, (h + dim_k)))
    v = jax.random.uniform(keys[2], shape=(batch_size, lk, rank_k, (h + dim_v)))

    q_segment_ids = einops.repeat(jnp.arange(lq) // 42, "l -> b l", b=batch_size)
    kv_segment_ids = einops.repeat(jnp.arange(lk) // 42, "l -> b l", b=batch_size)

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh(
        mesh_shape=(1, 8),
        devices=devices,
    )
    mesh = Mesh(device_mesh, axis_names=("dp", "sp"))
    sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))
    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)
    q_segment_ids = jax.device_put(q_segment_ids, sharding)
    kv_segment_ids = jax.device_put(kv_segment_ids, sharding)

    def f_test(q, k, v) -> Float[Array, ""]:
        o = shard_map(
            functools.partial(
                ring_tpa.ring_tpa,
                axis_name="sp",
                num_heads=h,
                causal=False,
                impl="pallas",
                interpret=True,
            ),
            mesh=mesh,
            in_specs=(
                PartitionSpec("dp", "sp", None),  # q
                PartitionSpec("dp", "sp", None),  # k
                PartitionSpec("dp", "sp", None),  # v
                PartitionSpec("dp", "sp"),  # q_segment_ids
                PartitionSpec("dp", "sp"),  # kv_segment_ids
            ),
            out_specs=PartitionSpec("dp", "sp", None),
            check_rep=False,
        )(q, k, v, q_segment_ids, kv_segment_ids)
        return jnp.nansum(o)

    dq, dk, dv = jax.jit(jax.grad(f_test, argnums=(0, 1, 2)))(q, k, v)

    # computing expected outputs.
    def f_ref(q, k, v):
        o = reference(
            q,
            k,
            v,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            num_heads=h,
            causal=False,
        )
        return jnp.nansum(o)

    dq_, dk_, dv_ = jax.jit(jax.grad(f_ref, argnums=(0, 1, 2)))(q, k, v)

    np.testing.assert_allclose(dq, dq_, atol=1e-3, rtol=1e-2)
    np.testing.assert_allclose(dk, dk_, atol=1e-3, rtol=1e-2)
    np.testing.assert_allclose(dv, dv_, atol=1e-3, rtol=1e-2)
