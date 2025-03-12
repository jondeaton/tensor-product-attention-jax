"""Multi-head attention."""

import jax
import jax.numpy as jnp
import einops

from jaxtyping import Float, Array


@jax.custom_vjp
def mha(
    q: Float[Array, "b lq h dk"],
    k: Float[Array, "b lk h dk"],
    v: Float[Array, "b lk h dv"],
    bias: Float[Array, "b lq lk"] | None = None,
    sm_scale: float = 1,
):
    """Multi-head attention."""
    o, _ = _mha_fwd(q, k, v, bias, sm_scale)
    return o


def _mha_fwd(
    q: Float[Array, "b lq h dk"],
    k: Float[Array, "b lk h dk"],
    v: Float[Array, "b lk h dv"],
    bias: Float[Array, "b lq lk"] | None,
    sm_scale: Float,
) -> tuple[Float[Array, "b lq h dv"], tuple[Array, ...]]:
    """Attention."""
    x = einops.einsum(q, k, "b lq h dk, b lk h dk -> b h lq lk")
    x = sm_scale * x
    if bias is not None:
        x += bias[:, None, :, :]
    p = jax.nn.softmax(x, axis=-1)
    o = einops.einsum(p, v, "b h lq lk, b lk h dv -> b lq h dv")
    return o, (p, v, q, k, sm_scale)


def _mha_bwd(
    res,
    do: Float[Array, "b lq h dv"],
) -> tuple[
    Float[Array, "b lq h dk"],
    Float[Array, "b lk h dk"],
    Float[Array, "b lk h dv"],
    None,
    None,
]:
    p, v, q, k, sm_scale = res

    # NOTE: handling these NANs corectly in the backwards pass is the entire reason why
    # we need this custom backwards pass implementaiton at all. Otherwise we end up with
    # nan gradients when the bias is -inf everywhere.
    p = jnp.where(jnp.isnan(p), 0, p)
    assert isinstance(p, jnp.ndarray)

    dv = einops.einsum(do, p, "b lq h dv, b h lq lk -> b lk h dv")
    dp = einops.einsum(do, v, "b lq h dv, b lk h dv -> b h lq lk")
    sum_dp_p = einops.einsum(dp, p, "b h lq lk, b h lq lk -> b h lq")
    dx = p * (dp - sum_dp_p[..., None])
    dx = sm_scale * dx
    dq = einops.einsum(dx, k, "b h lq lk, b lk h dk -> b lq h dk")
    dk = einops.einsum(dx, q, "b h lq lk, b lq h dk -> b lk h dk")
    return dq, dk, dv, None, None


mha.defvjp(_mha_fwd, _mha_bwd)
