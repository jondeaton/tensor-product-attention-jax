"""Tests for MHA."""

import jax
import jax.numpy as jnp

from .mha import mha


def test_mha_grads():
    q = jnp.array(
        [
            [2, 1],
            [1, 1],
            [1, 1],
        ],
        dtype=float,
    )[None, :, None, :]

    k = jnp.array(
        [
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        dtype=float,
    )[None, :, None, :]
    v = jnp.array(
        [
            [1, 1],
            [1, 1],
            [1, 1],
        ],
        dtype=float,
    )[None, :, None, :]

    bias = jnp.array(
        [
            [0, -jnp.inf, 0],
            [0, -jnp.inf, 0],
            [-jnp.inf, -jnp.inf, -jnp.inf],
        ]
    )[None, :, :]

    o = mha(q, k, v, bias)
    print("o:")
    print(o)
    print()

    def f(q, k, v):
        o = mha(q, k, v, bias=bias)
        return jnp.nansum(o)

    dq, dk, dv = jax.grad(f, argnums=(0, 1, 2))(q, k, v)
    """
    Expected:
    o:
    [[[[ 1.  1.]]

    [[ 1.  1.]]

    [[nan nan]]]]

    dq:
    [[[[0. 0.]]

    [[0. 0.]]

    [[0. 0.]]]]

    dk:
    [[[[0. 0.]]

    [[0. 0.]]

    [[0. 0.]]]]

    dv:
    [[[[1. 1.]]

    [[0. 0.]]

    [[1. 1.]]]]
    """
