"""Tests for MHA."""

import jax
import jax.numpy as jnp

from .mha import mha, _mha_fwd, _mha_bwd
import pytest


@pytest.mark.parametrize(
    "batch, lq, lk, h, dk, dv",
    [
        (2, 4, 6, 3, 8, 8),
        (1, 3, 5, 2, 4, 4),
    ],
)
@pytest.mark.parametrize("sm_scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("seed", [0, 1])
def test_gradients_match(batch, lq, lk, h, dk, dv, sm_scale: float, seed: int):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    q = jax.random.normal(keys[0], shape=(batch, lq, h, dk))
    k = jax.random.normal(keys[1], shape=(batch, lk, h, dk))
    v = jax.random.normal(keys[2], shape=(batch, lk, h, dv))

    bias = jax.random.normal(keys[3], shape=(batch, lq, lk))

    o, res = _mha_fwd(q, k, v, bias, sm_scale)
    do = jnp.ones_like(o)

    dq_manual, dk_manual, dv_manual, _, _ = _mha_bwd(res, do)

    def f(q, k, v, bias):
        o, _ = _mha_fwd(q, k, v, bias, sm_scale)
        return jnp.sum(o * do)

    dq_auto, dk_auto, dv_auto = jax.grad(f, argnums=(0, 1, 2))(q, k, v, bias)

    assert jnp.allclose(dq_manual, dq_auto, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(dk_manual, dk_auto, atol=1e-5, rtol=1e-5)
    assert jnp.allclose(dv_manual, dv_auto, atol=1e-5, rtol=1e-5)


# def test_mha_grads():
#     q = jnp.array(
#         [
#             [2, 1],
#             [1, 1],
#             [1, 1],
#         ],
#         dtype=float,
#     )[None, :, None, :]
#
#     k = jnp.array(
#         [
#             [1, 1],
#             [1, 1],
#             [1, 1],
#         ],
#         dtype=float,
#     )[None, :, None, :]
#     v = jnp.array(
#         [
#             [1, 1],
#             [1, 1],
#             [1, 1],
#         ],
#         dtype=float,
#     )[None, :, None, :]
#
#     bias = jnp.array(
#         [
#             [0, -jnp.inf, 0],
#             [0, -jnp.inf, 0],
#             [-jnp.inf, -jnp.inf, -jnp.inf],
#         ]
#     )[None, :, :]
#
#     o = mha(q, k, v, bias)
#     print("o:")
#     print(o)
#     print()
#
#     def f(q, k, v):
#         o = mha(q, k, v, bias=bias)
#         return jnp.nansum(o)
#
#     dq, dk, dv = jax.grad(f, argnums=(0, 1, 2))(q, k, v)
#     """
#     Expected:
#     o:
#     [[[[ 1.  1.]]
#
#     [[ 1.  1.]]
#
#     [[nan nan]]]]
#
#     dq:
#     [[[[0. 0.]]
#
#     [[0. 0.]]
#
#     [[0. 0.]]]]
#
#     dk:
#     [[[[0. 0.]]
#
#     [[0. 0.]]
#
#     [[0. 0.]]]]
#
#     dv:
#     [[[[1. 1.]]
#
#     [[0. 0.]]
#
#     [[1. 1.]]]]
#     """
