from __future__ import annotations

"""
broussard_model_jax.py

A self-contained JAX implementation of a Broussard et al. (2024)-style biophysical
calcium indicator model with multi-site sequential Ca2+ binding, endogenous buffers,
and extrusion, using stiff integration (backward Euler via Newton) and a fitting pipeline.

This version eliminates dataclass State from lax.cond and scan carry:
- New step_vec operates purely on packed jnp.ndarray state vectors and uses a JAX boolean predicate.
- simulate() uses step_vec directly and carries a packed vector in lax.scan.
- step() remains for API convenience and uses a Python if (no JAX cond) so it doesn't trace dataclasses.
- Scalar Params are 0D jnp.ndarray; State.c is a jnp array at init.

Design goals:
- Drop-in replacement for a Greenberg-like Sequential Binding Model (SBM) API.
- JAX-first, jit- and vmap-friendly pure functions with explicit state/params.
- Clear units, shapes, and docstrings.

Units and conventions:
- Concentrations: micromolar (uM)
- Time: seconds (s)
- Rates:
  - kon: 1/(uM*s)
  - koff: 1/s
  - k_ex: 1/s (linear extrusion)
  - Vmax: uM/s (Michaelis–Menten extrusion)
  - Km: uM
- Spikes: spikes[t] is spike count per bin; converted to influx rate u_influx = alpha_spike * spikes[t] (uM/s).
- Emission: F(t) = b0 + b1 * (phi · I), with phi dimensionless and I as occupancy concentrations (uM).

Numerics:
- Default method is "backward_euler" solved by Newton–Raphson with damping.
- A "forward_euler" step is provided for debugging.
- Internally uses lax.scan for time stepping in simulate(); carry is a packed state vector.
"""

from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.lax as lax
import jax.random as jr
import optax


# --------------------------
# Config and Params
# --------------------------

@dataclass
class Config:
    """
    Configuration for the Broussard biophysical model and solver.

    Attributes:
        n_sites: number of sequential Ca2+ binding sites on the indicator.
        n_buffers: number of endogenous buffers.
        extrusion: "linear" or "mm" (Michaelis–Menten).
        method: time stepping method ("backward_euler" or "forward_euler").
        newton_max_iters: maximum Newton iterations per step for backward Euler.
        newton_tol: infinity-norm tolerance on residual for convergence.
        newton_damping: damping factor applied to Newton updates (0<d<=1).
        clip_states: whether to clip physically impossible states each step.
    """
    n_sites: int = 4
    n_buffers: int = 2
    extrusion: str = "linear"  # "linear" or "mm"
    method: str = "backward_euler"  # "backward_euler" or "forward_euler"
    newton_max_iters: int = 15
    newton_tol: float = 1e-7
    newton_damping: float = 0.5
    clip_states: bool = True


# Register Params as a PyTree so Optax/JAX can handle it
@jax.tree_util.register_pytree_node_class
@dataclass
class Params:
    """
    Model parameters.

    Shapes depend on cfg.n_sites and cfg.n_buffers:

    Indicator kinetics (arrays length n_sites):
        kon: 1/(uM*s)
        koff: 1/s

    Brightness weights (array length n_sites+1):
        phi: dimensionless brightness per occupancy state (0..n_sites)

    Indicator concentration:
        I_tot: uM  (0D jnp.ndarray)

    Buffers (arrays length n_buffers):
        konB: 1/(uM*s)
        koffB: 1/s
        B_tot: uM

    Extrusion:
        k_ex: 1/s (for linear extrusion)  (0D jnp.ndarray)
        Vmax: uM/s (for Michaelis–Menten) (0D jnp.ndarray)
        Km: uM (for Michaelis–Menten)     (0D jnp.ndarray)

    Calcium baseline and spike coupling:
        c_rest: uM (resting free Ca2+)     (0D jnp.ndarray)
        alpha_spike: uM/s per spike (influx amplitude) (0D jnp.ndarray)

    Fluorescence readout:
        b0: offset  (0D jnp.ndarray)
        b1: scale   (0D jnp.ndarray)
    """
    # Indicator binding kinetics
    kon: jnp.ndarray   # shape [n_sites], 1/(uM*s)
    koff: jnp.ndarray  # shape [n_sites], 1/s

    # Brightness weights per occupancy state
    phi: jnp.ndarray   # shape [n_sites+1], dimensionless

    # Indicator total concentration (0D array)
    I_tot: jnp.ndarray

    # Buffers
    konB: jnp.ndarray  # shape [n_buffers], 1/(uM*s)
    koffB: jnp.ndarray # shape [n_buffers], 1/s
    B_tot: jnp.ndarray # shape [n_buffers], uM

    # Extrusion (0D arrays)
    k_ex: jnp.ndarray  # 1/s (linear)
    Vmax: jnp.ndarray  # uM/s (MM)
    Km: jnp.ndarray    # uM (MM)

    # Calcium baseline and spike coupling (0D arrays)
    c_rest: jnp.ndarray      # uM
    alpha_spike: jnp.ndarray # uM/s per spike

    # Fluorescence (0D arrays)
    b0: jnp.ndarray
    b1: jnp.ndarray

    # PyTree methods
    def tree_flatten(self):
        children = (
            self.kon, self.koff, self.phi, self.I_tot,
            self.konB, self.koffB, self.B_tot,
            self.k_ex, self.Vmax, self.Km,
            self.c_rest, self.alpha_spike,
            self.b0, self.b1
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def default_params(cfg: Config) -> Params:
    """
    Construct sensible default parameters for the given config.

    Defaults (float32):
      kon ~ [100, 80, 60, 40]/(uM*s)
      koff ~ [150, 120, 90, 60]/s
      phi linearly increasing 0..1 for states 0..n_sites
      I_tot = 20 uM
      Buffers:
        konB = [100, 50]/(uM*s)
        koffB = [50, 25]/s
        B_tot = [40, 20] uM
      Extrusion:
        k_ex = 20 1/s
        Vmax = 50 uM/s
        Km = 0.3 uM
      Calcium baseline and spike coupling:
        c_rest = 0.1 uM
        alpha_spike = 5.0 uM/s per spike
      Fluorescence:
        b0 = 0.0
        b1 = 1.0
    """
    n = cfg.n_sites
    nb = cfg.n_buffers
    f32 = jnp.float32

    # Indicator kinetics
    base_kon = jnp.array([100.0, 80.0, 60.0, 40.0], dtype=f32)
    base_koff = jnp.array([150.0, 120.0, 90.0, 60.0], dtype=f32)
    if n <= 4:
        kon = base_kon[:n]
        koff = base_koff[:n]
    else:
        extra = jnp.linspace(40.0, 20.0, n - 4, dtype=f32)
        kon = jnp.concatenate([base_kon, extra])
        extra_off = jnp.linspace(60.0, 30.0, n - 4, dtype=f32)
        koff = jnp.concatenate([base_koff, extra_off])

    # Brightness weights
    phi = jnp.linspace(0.0, 1.0, n + 1, dtype=f32)

    # Buffers
    base_konB = jnp.array([100.0, 50.0], dtype=f32)
    base_koffB = jnp.array([50.0, 25.0], dtype=f32)
    base_Btot = jnp.array([40.0, 20.0], dtype=f32)
    if nb <= 2:
        konB = base_konB[:nb]
        koffB = base_koffB[:nb]
        B_tot = base_Btot[:nb]
    else:
        add = nb - 2
        konB = jnp.concatenate([base_konB, jnp.full((add,), base_konB[-1], dtype=f32)])
        koffB = jnp.concatenate([base_koffB, jnp.full((add,), base_koffB[-1], dtype=f32)])
        B_tot = jnp.concatenate([base_Btot, jnp.full((add,), base_Btot[-1], dtype=f32)])

    return Params(
        kon=kon,
        koff=koff,
        phi=phi,
        I_tot=jnp.asarray(20.0, dtype=f32),
        konB=konB,
        koffB=koffB,
        B_tot=B_tot,
        k_ex=jnp.asarray(20.0, dtype=f32),
        Vmax=jnp.asarray(50.0, dtype=f32),
        Km=jnp.asarray(0.3, dtype=f32),
        c_rest=jnp.asarray(0.1, dtype=f32),
        alpha_spike=jnp.asarray(5.0, dtype=f32),
        b0=jnp.asarray(0.0, dtype=f32),
        b1=jnp.asarray(1.0, dtype=f32),
    )


# --------------------------
# State and initialization
# --------------------------

@dataclass
class State:
    """
    Dynamical state variables.

    Attributes:
        c: free Ca2+ concentration (uM), scalar jnp.ndarray (0D)
        I: indicator occupancy vector (uM), shape [n_sites+1], sum equals I_tot
        Bc: bound buffer concentrations per buffer (uM), shape [n_buffers]
    """
    c: jnp.ndarray
    I: jnp.ndarray
    Bc: jnp.ndarray


def init_state(params: Params, cfg: Config) -> State:
    """
    Initialize state at equilibrium rest:
      c = c_rest (as 0D jnp.ndarray)
      I = [I_tot, 0, ..., 0] (all indicator unbound)
      Bc = B_tot * c_rest / (c_rest + KdB) with KdB = koffB/konB

    Returns:
        State with shapes consistent with cfg.
    """
    f32 = jnp.float32
    c = jnp.asarray(params.c_rest, dtype=f32)
    I0 = jnp.zeros((cfg.n_sites + 1,), dtype=f32)
    I0 = I0.at[0].set(params.I_tot.astype(f32))
    KdB = params.koffB / params.konB
    Bc = params.B_tot * c / (c + KdB)
    return State(c=c, I=I0, Bc=Bc)


# --------------------------
# Helpers: pack/unpack state
# --------------------------

def pack_state(state: State) -> jnp.ndarray:
    """
    Pack State into a 1D vector: [c, I..., Bc...]

    Returns:
        x: shape [1 + (n_sites+1) + n_buffers]
    """
    c_vec = jnp.atleast_1d(state.c)
    return jnp.concatenate([c_vec, state.I, state.Bc])


def unpack_state(x: jnp.ndarray, cfg: Config) -> State:
    """
    Unpack a 1D vector into State, using cfg for sizes.

    Args:
        x: vector shape [1 + (n_sites+1) + n_buffers]

    Returns:
        State with c (0D array), I (n_sites+1), Bc (n_buffers)
    """
    nI = cfg.n_sites + 1
    nb = cfg.n_buffers
    c = jnp.asarray(x[0])
    I = x[1: 1 + nI]
    Bc = x[1 + nI: 1 + nI + nb]
    return State(c=c, I=I, Bc=Bc)


# --------------------------
# Model dynamics
# --------------------------

def indicator_fluxes(c: jnp.ndarray, I: jnp.ndarray, kon: jnp.ndarray, koff: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute indicator binding/unbinding fluxes for each site.

    Args:
        c: free Ca2+ (uM), 0D or scalar array
        I: occupancy vector [n_sites+1] (uM)
        kon: [n_sites], 1/(uM*s)
        koff: [n_sites], 1/s

    Returns:
        bind: [n_sites], rates from I_j -> I_{j+1} (uM/s)
        unbind: [n_sites], rates from I_{j+1} -> I_j (uM/s)
    """
    bind = kon * c * I[:-1]
    unbind = koff * I[1:]
    return bind, unbind


def rhs(state: State, params: Params, u_influx: jnp.ndarray, cfg: Config) -> State:
    """
    Right-hand side of ODEs (mass-action kinetics).

    Species:
      - Indicator occupancy I[0..n_sites]: sequential binding/unbinding
      - Buffers Bc[k]: endogenous buffers
      - Free Ca2+ c

    Fluxes (uM/s):
      - Indicator: J_I = sum(bind - unbind)
      - Buffers: per buffer k, dBc_k = konB_k * c * (B_tot_k - Bc_k) - koffB_k * Bc_k
        Aggregate J_B = sum_k dBc_k
      - Extrusion:
          linear: J_ex = k_ex * (c - c_rest)
          mm:     J_ex = Vmax * (c/(Km + c) - c_rest/(Km + c_rest))

    Ca2+ ODE:
      dc = u_influx - J_I - J_B - J_ex

    Args:
        state: current State
        params: model Parameters
        u_influx: Ca2+ influx rate (uM/s), typically alpha_spike * spikes_t
        cfg: Config

    Returns:
        State of derivatives: (dc, dI, dBc)
    """
    c = state.c
    I = state.I
    Bc = state.Bc

    # Indicator kinetics
    bind, unbind = indicator_fluxes(c, I, params.kon, params.koff)

    # dI for the sequential chain
    dI0 = -bind[0] + unbind[0]
    mid = bind[:-1] - unbind[:-1] - bind[1:] + unbind[1:]
    dIn = bind[-1] - unbind[-1]
    dI = jnp.concatenate([jnp.atleast_1d(dI0), mid, jnp.atleast_1d(dIn)])

    # Buffers
    B_free = params.B_tot - Bc
    bindB = params.konB * c * B_free
    unbindB = params.koffB * Bc
    dBc = bindB - unbindB
    J_B = jnp.sum(dBc)

    # Indicator Ca flux (1 Ca per binding step)
    J_I = jnp.sum(bind - unbind)

    # Extrusion with JAX boolean predicate
    pred_ex = jnp.asarray(cfg.extrusion == "linear", dtype=jnp.bool_)

    def linear_ex(_):
        return params.k_ex * (c - params.c_rest)

    def mm_ex(_):
        # Centered so J_ex(c_rest) = 0
        return params.Vmax * (c / (params.Km + c) - params.c_rest / (params.Km + params.c_rest))

    J_ex = lax.cond(pred_ex, linear_ex, mm_ex, operand=None)

    # Free calcium ODE
    dc = u_influx - J_I - J_B - J_ex

    return State(c=dc, I=dI, Bc=dBc)


# --------------------------
# Time stepping (State)
# --------------------------

def _clip_and_project(state: State, params: Params, cfg: Config) -> State:
    """
    Enforce physical constraints:
      - c >= 0
      - I >= 0 and sum(I) == I_tot (renormalize after clipping)
      - 0 <= Bc <= B_tot
    """
    c = jnp.maximum(state.c, 0.0)

    # Clip occupancy, then renormalize to I_tot
    I_pos = jnp.maximum(state.I, 0.0)
    sumI = jnp.sum(I_pos)
    # Construct target [I_tot, 0, ..., 0]
    I_target = jnp.concatenate([params.I_tot[None], jnp.zeros_like(I_pos[1:])])
    scale = params.I_tot / (sumI + 1e-16)
    I_scaled = I_pos * scale
    I = jnp.where(sumI > 1e-16, I_scaled, I_target)

    # Buffers within [0, B_tot]
    Bc = jnp.clip(state.Bc, 0.0, params.B_tot)

    return State(c=c, I=I, Bc=Bc)


def _newton_solve_backward_euler(
    x0: jnp.ndarray,
    dt: float,
    params: Params,
    u_influx: jnp.ndarray,
    cfg: Config,
) -> jnp.ndarray:
    """
    Solve for x_next with backward Euler:
      g(x_next) = x_next - x0 - dt * f(x_next) = 0

    Uses damped Newton iterations with automatic Jacobian via jax.jacrev.
    """
    def f_vec(x_vec: jnp.ndarray) -> jnp.ndarray:
        st = unpack_state(x_vec, cfg)
        derivs = rhs(st, params, u_influx, cfg)
        return pack_state(derivs)

    def g(x_vec: jnp.ndarray) -> jnp.ndarray:
        return x_vec - x0 - dt * f_vec(x_vec)

    # Newton iterations (masked early stop)
    def body_fun(carry):
        x, converged = carry
        gx = g(x)
        J = jax.jacrev(g)(x)
        dx = jnp.linalg.solve(J, -gx)
        x_new = x + cfg.newton_damping * dx
        res = jnp.max(jnp.abs(gx))
        converged_now = res < cfg.newton_tol
        converged_out = jnp.logical_or(converged, converged_now)
        # If already converged, keep x
        x_out = jnp.where(converged, x, x_new)
        return (x_out, converged_out)

    def scan_step(carry, _):
        return body_fun(carry), None

    carry0 = (x0, jnp.array(False))
    (xf, _), _ = lax.scan(scan_step, carry0, xs=None, length=cfg.newton_max_iters)
    return xf


def backward_euler_step(state: State, params: Params, u: jnp.ndarray, dt: float, cfg: Config) -> State:
    """
    One implicit backward Euler step using Newton's method.
    """
    x0 = pack_state(state)
    x_next = _newton_solve_backward_euler(x0, dt, params, u, cfg)
    st_next = unpack_state(x_next, cfg)
    if cfg.clip_states:
        st_next = _clip_and_project(st_next, params, cfg)
    return st_next


def forward_euler_step(state: State, params: Params, u: jnp.ndarray, dt: float, cfg: Config) -> State:
    """
    One explicit forward Euler step (debugging).
    """
    derivs = rhs(state, params, u, cfg)
    st_next = State(
        c=state.c + dt * derivs.c,
        I=state.I + dt * derivs.I,
        Bc=state.Bc + dt * derivs.Bc,
    )
    if cfg.clip_states:
        st_next = _clip_and_project(st_next, params, cfg)
    return st_next


# --------------------------
# Emission
# --------------------------

def emission(params: Params, state: State) -> jnp.ndarray:
    """
    Fluorescence emission model.

    F(t) = b0 + b1 * (phi · I)

    Returns:
        scalar fluorescence (0D jnp.ndarray)
    """
    return params.b0 + params.b1 * jnp.dot(params.phi, state.I)


# --------------------------
# Vectorized step and simulation
# --------------------------

def step_vec(
    x_vec: jnp.ndarray,
    params: Params,
    u: jnp.ndarray,
    dt: float,
    cfg: Config,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized single step that avoids carrying/returning dataclasses in JAX cond.

    Uses a JAX boolean predicate and returns a packed state vector and emission.
    """
    pred = jnp.asarray(cfg.method == "backward_euler", dtype=jnp.bool_)

    def do_backward(x_in: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        st_in = unpack_state(x_in, cfg)
        st_next = backward_euler_step(st_in, params, u, dt, cfg)
        y_t = emission(params, st_next)
        return pack_state(st_next), y_t

    def do_forward(x_in: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        st_in = unpack_state(x_in, cfg)
        st_next = forward_euler_step(st_in, params, u, dt, cfg)
        y_t = emission(params, st_next)
        return pack_state(st_next), y_t

    x_next, y_t = lax.cond(pred, do_backward, do_forward, x_vec)
    return x_next, y_t


def simulate(
    params: Params,
    cfg: Config,
    spikes: jnp.ndarray,
    dt: float,
    T: int,
    state0: Optional[State] = None,
) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
    """
    Simulate over T steps given a spike train.

    This implementation is PyTree-safe for JAX transforms:
    - Carries a packed jnp.ndarray state vector in lax.scan (no Python dataclass carry).
    - Returns the final State (unpacked), the packed state trajectory, and fluorescence.

    Args:
        params: Params
        cfg: Config
        spikes: [T] spike counts per bin
        dt: step (s)
        T: number of steps
        state0: optional initial State (defaults to rest)

    Returns:
        final_state: State at T
        all_states: [T, 1 + (n_sites+1) + n_buffers] packed state trajectory
        y_pred: [T] fluorescence trajectory
    """
    if state0 is None:
        state0 = init_state(params, cfg)

    x0 = pack_state(state0)

    def scan_fn(x_vec: jnp.ndarray, spk: jnp.ndarray):
        u = params.alpha_spike * spk  # uM/s per spike count
        x_next, y_t = step_vec(x_vec, params, u, dt, cfg)
        return x_next, (x_next, y_t)

    xT, (states_pack, y) = lax.scan(scan_fn, x0, spikes, length=T)
    final_state = unpack_state(xT, cfg)
    return final_state, states_pack, y


# --------------------------
# API convenience step (Python if)
# --------------------------

def step(
    state: State,
    params: Params,
    u: jnp.ndarray,
    dt: float,
    cfg: Config,
) -> Tuple[State, jnp.ndarray]:
    """
    API-compatible step that returns (State, emission) using a Python if.
    This avoids tracing a dataclass through lax.cond.
    """
    if cfg.method == "backward_euler":
        st_next = backward_euler_step(state, params, u, dt, cfg)
    else:
        st_next = forward_euler_step(state, params, u, dt, cfg)
    y_t = emission(params, st_next)
    return st_next, y_t


# --------------------------
# Loss and fitting
# --------------------------

def _l2_regularizer_params(params: Params, weight: float = 1e-6) -> jnp.ndarray:
    """
    Small L2 regularization on parameters to stabilize fitting.
    """
    sq = 0.0
    sq += jnp.sum(params.kon ** 2)
    sq += jnp.sum(params.koff ** 2)
    sq += jnp.sum(params.phi ** 2)
    sq += jnp.sum(params.I_tot ** 2)
    sq += jnp.sum(params.konB ** 2)
    sq += jnp.sum(params.koffB ** 2)
    sq += jnp.sum(params.B_tot ** 2)
    sq += jnp.sum(params.k_ex ** 2)
    sq += jnp.sum(params.Vmax ** 2)
    sq += jnp.sum(params.Km ** 2)
    sq += jnp.sum(params.c_rest ** 2)
    sq += jnp.sum(params.alpha_spike ** 2)
    sq += jnp.sum(params.b0 ** 2)
    sq += jnp.sum(params.b1 ** 2)
    return weight * sq


def mse_loss(
    params: Params,
    cfg: Config,
    spikes: jnp.ndarray,
    y_true: jnp.ndarray,
    dt: float,
    l2_weight: float = 1e-6,
) -> jnp.ndarray:
    """
    Mean squared error between predicted and observed fluorescence.

    Args:
        params: Params
        cfg: Config
        spikes: [T] spike counts
        y_true: [T] observed fluorescence
        dt: time step (s)
        l2_weight: small L2 regularization strength

    Returns:
        scalar loss (0D jnp.ndarray)
    """
    T = y_true.shape[0]
    _, _, y_pred = simulate(params, cfg, spikes, dt, T)
    mse = jnp.mean((y_pred - y_true) ** 2)
    reg = _l2_regularizer_params(params, weight=l2_weight)
    return mse + reg


def _make_bounds_like(
    params: Params,
    cfg: Config,
    clip: Optional[Dict[str, Tuple[float, float]]],
) -> Tuple[Params, Params]:
    """
    Build parameter-wise lower and upper bounds (Params-shaped) from a dict mapping
    field names to (low, high). Missing fields default to (-inf, +inf).
    """
    neg_inf = -jnp.inf
    pos_inf = jnp.inf

    def get(name: str, default: Tuple[float, float]) -> Tuple[float, float]:
        if clip is None:
            return default
        return clip.get(name, default)

    # Scalars (0D arrays)
    I_tot_lo, I_tot_hi = get("I_tot", (neg_inf, pos_inf))
    k_ex_lo, k_ex_hi = get("k_ex", (neg_inf, pos_inf))
    Vmax_lo, Vmax_hi = get("Vmax", (neg_inf, pos_inf))
    Km_lo, Km_hi = get("Km", (neg_inf, pos_inf))
    c_rest_lo, c_rest_hi = get("c_rest", (neg_inf, pos_inf))
    alpha_lo, alpha_hi = get("alpha_spike", (neg_inf, pos_inf))
    b0_lo, b0_hi = get("b0", (neg_inf, pos_inf))
    b1_lo, b1_hi = get("b1", (neg_inf, pos_inf))

    # Arrays: allow scalar bounds broadcasted to shapes
    kon_lo_s, kon_hi_s = get("kon", (neg_inf, pos_inf))
    koff_lo_s, koff_hi_s = get("koff", (neg_inf, pos_inf))
    phi_lo_s, phi_hi_s = get("phi", (neg_inf, pos_inf))
    konB_lo_s, konB_hi_s = get("konB", (neg_inf, pos_inf))
    koffB_lo_s, koffB_hi_s = get("koffB", (neg_inf, pos_inf))
    Btot_lo_s, Btot_hi_s = get("B_tot", (neg_inf, pos_inf))

    lo = Params(
        kon=jnp.full_like(params.kon, kon_lo_s),
        koff=jnp.full_like(params.koff, koff_lo_s),
        phi=jnp.full_like(params.phi, phi_lo_s),
        I_tot=jnp.asarray(I_tot_lo, dtype=params.I_tot.dtype),
        konB=jnp.full_like(params.konB, konB_lo_s),
        koffB=jnp.full_like(params.koffB, koffB_lo_s),
        B_tot=jnp.full_like(params.B_tot, Btot_lo_s),
        k_ex=jnp.asarray(k_ex_lo, dtype=params.k_ex.dtype),
        Vmax=jnp.asarray(Vmax_lo, dtype=params.Vmax.dtype),
        Km=jnp.asarray(Km_lo, dtype=params.Km.dtype),
        c_rest=jnp.asarray(c_rest_lo, dtype=params.c_rest.dtype),
        alpha_spike=jnp.asarray(alpha_lo, dtype=params.alpha_spike.dtype),
        b0=jnp.asarray(b0_lo, dtype=params.b0.dtype),
        b1=jnp.asarray(b1_lo, dtype=params.b1.dtype),
    )
    hi = Params(
        kon=jnp.full_like(params.kon, kon_hi_s),
        koff=jnp.full_like(params.koff, koff_hi_s),
        phi=jnp.full_like(params.phi, phi_hi_s),
        I_tot=jnp.asarray(I_tot_hi, dtype=params.I_tot.dtype),
        konB=jnp.full_like(params.konB, konB_hi_s),
        koffB=jnp.full_like(params.koffB, koffB_hi_s),
        B_tot=jnp.full_like(params.B_tot, Btot_hi_s),
        k_ex=jnp.asarray(k_ex_hi, dtype=params.k_ex.dtype),
        Vmax=jnp.asarray(Vmax_hi, dtype=params.Vmax.dtype),
        Km=jnp.asarray(Km_hi, dtype=params.Km.dtype),
        c_rest=jnp.asarray(c_rest_hi, dtype=params.c_rest.dtype),
        alpha_spike=jnp.asarray(alpha_hi, dtype=params.alpha_spike.dtype),
        b0=jnp.asarray(b0_hi, dtype=params.b0.dtype),
        b1=jnp.asarray(b1_hi, dtype=params.b1.dtype),
    )
    return lo, hi


def _clip_params_to_bounds(p: Params, lo: Params, hi: Params) -> Params:
    """
    Elementwise clip of Params to [lo, hi], broadcasting as needed.
    """
    return Params(
        kon=jnp.clip(p.kon, lo.kon, hi.kon),
        koff=jnp.clip(p.koff, lo.koff, hi.koff),
        phi=jnp.clip(p.phi, lo.phi, hi.phi),
        I_tot=jnp.clip(p.I_tot, lo.I_tot, hi.I_tot),
        konB=jnp.clip(p.konB, lo.konB, hi.konB),
        koffB=jnp.clip(p.koffB, lo.koffB, hi.koffB),
        B_tot=jnp.clip(p.B_tot, lo.B_tot, hi.B_tot),
        k_ex=jnp.clip(p.k_ex, lo.k_ex, hi.k_ex),
        Vmax=jnp.clip(p.Vmax, lo.Vmax, hi.Vmax),
        Km=jnp.clip(p.Km, lo.Km, hi.Km),
        c_rest=jnp.clip(p.c_rest, lo.c_rest, hi.c_rest),
        alpha_spike=jnp.clip(p.alpha_spike, lo.alpha_spike, hi.alpha_spike),
        b0=jnp.clip(p.b0, lo.b0, hi.b0),
        b1=jnp.clip(p.b1, lo.b1, hi.b1),
    )


def fit(
    params: Params,
    cfg: Config,
    spikes: jnp.ndarray,
    y_true: jnp.ndarray,
    dt: float,
    steps: int = 2000,
    lr: float = 1e-2,
    clip: Optional[Dict[str, Tuple[float, float]]] = None,
    l2_weight: float = 1e-6,
) -> Tuple[Params, jnp.ndarray]:
    """
    Fit parameters to fluorescence y_true given spikes using Adam.

    Args:
        params: initial Params
        cfg: Config
        spikes: [T] spike counts
        y_true: [T] fluorescence observations
        dt: time step (s)
        steps: number of optimizer steps
        lr: learning rate
        clip: optional dict of bounds per parameter name: {"kon": (lo, hi), ...}
              Scalars are broadcast to the corresponding arrays.
        l2_weight: small L2 regularization

    Returns:
        (fitted_params, loss_history[steps])
    """
    lo_bounds, hi_bounds = _make_bounds_like(params, cfg, clip)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def loss_fn(p: Params) -> jnp.ndarray:
        return mse_loss(p, cfg, spikes, y_true, dt, l2_weight=l2_weight)

    def train_step(carry, _):
        p, s = carry
        loss_val, grads = jax.value_and_grad(loss_fn)(p)
        updates, s_new = optimizer.update(grads, s, params=p)
        p_new = optax.apply_updates(p, updates)
        p_new = _clip_params_to_bounds(p_new, lo_bounds, hi_bounds)
        return (p_new, s_new), loss_val

    (p_final, _), loss_hist = lax.scan(train_step, (params, opt_state), xs=None, length=steps)
    return p_final, loss_hist


# --------------------------
# API compatibility shim
# --------------------------

class BroussardModel:
    """
    Greenberg-like SBM API shim.

    Methods:
        defaults(cfg) -> Params
        init_state(params, cfg) -> State
        step(state, params, input_t, dt, cfg) -> (state_next, y_t)
        simulate(params, cfg, spikes, dt, state0=None) -> (state_T, states_packed, y_pred)
        loss(params, cfg, spikes, y_true, dt) -> float
        fit(params, cfg, spikes, y_true, dt, ...) -> (params_fitted, history)
    """

    @staticmethod
    def defaults(cfg: Config) -> Params:
        return default_params(cfg)

    @staticmethod
    def init_state(params: Params, cfg: Config) -> State:
        return init_state(params, cfg)

    @staticmethod
    def step(state: State, params: Params, input_t: float, dt: float, cfg: Config) -> Tuple[State, jnp.ndarray]:
        return step(state, params, input_t, dt, cfg)

    @staticmethod
    def simulate(
        params: Params,
        cfg: Config,
        spikes: jnp.ndarray,
        dt: float,
        state0: Optional[State] = None,
    ) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
        T = spikes.shape[0]
        return simulate(params, cfg, spikes, dt, T, state0=state0)

    @staticmethod
    def loss(
        params: Params,
        cfg: Config,
        spikes: jnp.ndarray,
        y_true: jnp.ndarray,
        dt: float,
        l2_weight: float = 1e-6,
    ) -> jnp.ndarray:
        return mse_loss(params, cfg, spikes, y_true, dt, l2_weight=l2_weight)

    @staticmethod
    def fit(
        params: Params,
        cfg: Config,
        spikes: jnp.ndarray,
        y_true: jnp.ndarray,
        dt: float,
        steps: int = 2000,
        lr: float = 1e-2,
        clip: Optional[Dict[str, Tuple[float, float]]] = None,
        l2_weight: float = 1e-6,
    ) -> Tuple[Params, jnp.ndarray]:
        return fit(params, cfg, spikes, y_true, dt, steps=steps, lr=lr, clip=clip, l2_weight=l2_weight)


# --------------------------
# Testing utilities
# --------------------------

def synthetic_demo(T: int = 2000, dt: float = 0.01) -> None:
    """
    Quick synthetic test:
      - Generate Poisson spikes at 5 Hz
      - Simulate fluorescence with default params
      - Add small Gaussian noise (sigma=0.02)
      - Fit parameters (a subset bounded) and report losses

    Prints initial and final loss values.
    """
    key = jr.PRNGKey(0)

    cfg = Config(
        n_sites=4,
        n_buffers=2,
        extrusion="linear",
        method="backward_euler",
        newton_max_iters=15,
        newton_tol=1e-7,
        newton_damping=0.5,
        clip_states=True,
    )
    params0 = default_params(cfg)

    # Generate Poisson spikes at rate 5 Hz
    rate_hz = 5.0
    lam = rate_hz * dt
    spikes = jr.poisson(key, lam=lam, shape=(T,)).astype(jnp.float32)

    # Simulate "ground truth"
    _, _, y_gt = simulate(params0, cfg, spikes, dt, T)

    # Add Gaussian noise
    key, sub = jr.split(key)
    sigma = 0.02
    y_obs = y_gt + sigma * jr.normal(sub, shape=y_gt.shape)

    # Initial loss
    init_loss = mse_loss(params0, cfg, spikes, y_obs, dt)

    # Fit with reasonable bounds (keep positivity and sanity)
    clip = {
        "kon": (1e-2, 1e4),
        "koff": (1e-3, 1e4),
        "phi": (0.0, 2.0),
        "I_tot": (1e-3, 1e3),
        "konB": (1e-2, 1e4),
        "koffB": (1e-3, 1e4),
        "B_tot": (1e-3, 1e4),
        "k_ex": (1e-3, 1e3),
        "Vmax": (1e-3, 1e4),
        "Km": (1e-6, 1e2),
        "c_rest": (1e-6, 10.0),
        "alpha_spike": (1e-6, 1e3),
        "b0": (-10.0, 10.0),
        "b1": (1e-3, 1e3),
    }

    params_fit, hist = fit(params0, cfg, spikes, y_obs, dt, steps=500, lr=2e-2, clip=clip, l2_weight=1e-6)
    final_loss = hist[-1]

    # Report
    print("Synthetic demo")
    print(f"  T = {T}, dt = {dt}")
    print(f"  Initial loss: {float(init_loss):.6f}")
    print(f"  Final  loss:  {float(final_loss):.6f}")


# If run as a script, execute synthetic demo
if __name__ == "__main__":
    synthetic_demo()