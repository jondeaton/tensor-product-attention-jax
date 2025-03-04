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

    x = einops.einsum(q_, k_, "b lq h dk, b lk h dk -> b h lq lk")
    x *= sm_scale

    segment_mask = q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]
    x += jnp.where(segment_mask[:, None, :, :], 0, -jnp.inf)

    if causal:
        mask = jnp.tril(jnp.ones(shape=(lq, lk), dtype=bool))
        x += jnp.where(mask, 0, -jnp.inf)

    p = jax.nn.softmax(x, axis=-1)
    where_none = jnp.isnan(p).all(axis=-1)
    p = p.at[where_none].set(0)

    o = einops.einsum(p, v_, "b h lq lk, b lk h dv -> b lq h dv")

    where_none = einops.rearrange(where_none, "b h lq -> b lq h")
    return o.at[where_none].set(jnp.nan)


@pytest.mark.parametrize("rank_q", [1, 2, 6])
@pytest.mark.parametrize("rank_k", [1, 2, 4])
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

    devices = jax.devices()
    device_mesh = mesh_utils.create_device_mesh(
        mesh_shape=(1, 8),
        devices=devices,
    )
    mesh = Mesh(device_mesh, axis_names=("dp", "sp"))

    keys = jax.random.split(key, 3)
    q = jax.random.uniform(keys[0], shape=(batch_size, lq, rank_q, (h + dk)))
    k = jax.random.uniform(keys[1], shape=(batch_size, lk, rank_k, (h + dk)))
    v = jax.random.uniform(keys[2], shape=(batch_size, lk, rank_k, (h + dv)))

    q_segment_ids = einops.repeat(jnp.arange(lq) // 42, "l -> b l", b=batch_size)
    kv_segment_ids = einops.repeat(jnp.arange(lk) // 42, "l -> b l", b=batch_size)

    out_ref = reference(
        q,
        k,
        v,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        num_heads=h,
    )
    assert out_ref.shape == (batch_size, lq, h, dv), out_ref.shape

    sharding = NamedSharding(mesh, PartitionSpec(None, "sp"))
    q = jax.device_put(q, sharding)
    k = jax.device_put(k, sharding)
    v = jax.device_put(v, sharding)
    q_segment_ids = jax.device_put(q_segment_ids, sharding)
    kv_segment_ids = jax.device_put(kv_segment_ids, sharding)

    # out = jax.jit(
    out = shard_map(
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
        # )
    )(q, k, v, q_segment_ids, kv_segment_ids)

    assert out.shape == (batch_size, lq, h, dv), out_ref.shape
    np.testing.assert_allclose(out, out_ref, rtol=1e-3, atol=1e-3)
