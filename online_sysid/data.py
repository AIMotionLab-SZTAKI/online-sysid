from jax import numpy as jnp
import numpy as np
import jax

from typing import Optional


def init_online_buffer(m: int, N: int, nu: int, ny: int, nx: int, n: int=0):
    """
    Initializes the replay buffer for online system identification with fixed sizes so that efficient JIT compilation
    can be applied for the whole learning pipeline.

    Args:
        m (int): number of past segments to retain.
        N (int): training segment length per slot (steps used in the loss)
        nu (int): Dimension of the input variable.
        ny (int): Dimension of the output variable.
        nx (int): Dimension of the state variable.
        n (int, optional): encoder lag; if > 0 each slot also stores n context samples taken from the tail of the
            preceding segment for state estimation. Defaults to 0.
    
    Returns:
        buf (dict): Replay buffer.
    """
    buf = {
        "Y": jnp.zeros((m, N, ny)),
        "U": jnp.zeros((m, N, nu)),
        "x0": jnp.zeros((m, nx)),
        "ptr": jnp.int64(0),
        "filled": jnp.int64(0),
    }
    if n > 0:
        buf["ctx_Y"] = jnp.zeros((m, n, ny))
        buf["ctx_U"] = jnp.zeros((m, n, nu))
    return buf


def insert_segment(buf: dict, Y_new: np.ndarray, U_new: np.ndarray, x0_new: Optional[np.ndarray]=None):
    """
    Insert one new segment into the replay buffer using JIT-compatible operations.

    When n == 0 (no encoder and state initialization window):
        Y_new shaped (N, ny), U_new shaped (N, nu) stored in the replay buffer. If the buffer is fully populated, 
        the oldest batch is dicarded.

    When n > 0 (with encoder lag):
        First call: Y_new, U_new must be shaped (n+N, ny/nu). The first n data points are placed in the state
        initialization window (encoder context); the remaining N samples are stored separately.
        Subsequent calls: Y_new, U_new are shaped (N, ny/nu). State initialization is taken automatically from the
        previous batches.

    Args:
        buf (dict): Replay buffer that requires updating.
        Y_new (ndarray): Output measurements for the new batch, shaped (N, ny). If an encoder is used, for the first
            call, it should be shaped (n+N, ny).
        U_new (ndarray): Input measurements for the new batch, shaped (N, nu). If an encoder is used, for the first
            call, it should be shaped (n+N, nu).
        x0_new (ndarray, optional): Only used when full-state measurements are available. In that case, it provides the
            measured initial state corresponding to the new batch, shaped (nx,). If None, initial states are estimated.
            Defaults to None.
    
    Returns:
        result (dict): Updated replay buffer.
    """
    ptr = int(buf["ptr"])
    m = buf["Y"].shape[0]
    has_ctx = "ctx_Y" in buf

    if has_ctx:
        n = buf["ctx_Y"].shape[1]
        if int(buf["filled"]) == 0:
            ctx_Y = Y_new[:n]
            ctx_U = U_new[:n]
            seg_Y = Y_new[n:]
            seg_U = U_new[n:]
        else:
            ptr_prev = (ptr - 1 + m) % m
            ctx_Y = buf["Y"][ptr_prev, -n:]
            ctx_U = buf["U"][ptr_prev, -n:]
            seg_Y = Y_new
            seg_U = U_new
        ctx_Y_buf = jax.lax.dynamic_update_slice(buf["ctx_Y"], ctx_Y[None], (ptr, 0, 0))
        ctx_U_buf = jax.lax.dynamic_update_slice(buf["ctx_U"], ctx_U[None], (ptr, 0, 0))
    else:
        seg_Y = Y_new
        seg_U = U_new

    Y = jax.lax.dynamic_update_slice(buf["Y"], seg_Y[None], (ptr, 0, 0))
    U = jax.lax.dynamic_update_slice(buf["U"], seg_U[None], (ptr, 0, 0))

    if x0_new is not None:
        x0 = jax.lax.dynamic_update_slice(buf["x0"], x0_new[None], (ptr, 0))
    else:
        x0 = buf["x0"]

    result = {
        "Y": Y,
        "U": U,
        "x0": x0,
        "ptr": jnp.int64((ptr + 1) % m),
        "filled": jnp.minimum(buf["filled"] + 1, m),
    }
    if has_ctx:
        result["ctx_Y"] = ctx_Y_buf
        result["ctx_U"] = ctx_U_buf
    return result
