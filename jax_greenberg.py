#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX fitting of a sequential-binding Ca2+ model (GCaMP-like) to fluorescence
using the provided ground-truth .mat loader (CAttached struct).

Fixes "Non-hashable static arguments" and "PjitFunction passed as arg" by:
  - Using a config dict (pytree) of JAX arrays/floats
  - NOT passing jitted callables as arguments to other jitted functions
  - Computing the RHS directly in the implicit step with cfg as a normal arg

Data loader expects 'CAttached' with fields:
  - fluo_time (s)
  - fluo_mean (dF/F)
  - events_AP (in 0.1 ms units)

Usage:
  python greenberg_sbm_jax_gt_fixed2.py --mat path/to/data.mat --rec-id 0 --outdir results --steps 1500 --n-steps 3

Requires: jax, jaxlib, numpy, scipy, matplotlib
"""

import os
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, lax
from functools import partial

# ---------------------------
# Data loader (exact function requested)
# ---------------------------

def load_ground_truth_mat(mat_path: str, recording_id: int = 0):
    """
    Expects 'CAttached' cell array with fields:
      - fluo_time (s)
      - fluo_mean (dF/F)
      - events_AP (in 0.1 ms units)
    """
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    CAttached = data["CAttached"]
    rec = CAttached[recording_id] if isinstance(CAttached, np.ndarray) else CAttached
    fluo_time = np.asarray(rec.fluo_time).ravel().astype(float)
    fluo_mean = np.asarray(rec.fluo_mean).ravel().astype(float)
    events_AP = np.asarray(rec.events_AP).ravel().astype(float)
    ap_times_s = events_AP / 1e4  # convert 0.1 ms units to seconds
    return fluo_time, fluo_mean, ap_times_s

# ---------------------------
# Utilities
# ---------------------------

def spike_times_to_train(t: np.ndarray, spike_times_s: np.ndarray) -> np.ndarray:
    """
    Convert spike times (seconds) to a per-sample spike train (counts per bin).
    """
    t = np.asarray(t, dtype=float)
    spike_times_s = np.asarray(spike_times_s, dtype=float).ravel()
    T = t.size
    train = np.zeros(T, dtype=float)
    if spike_times_s.size == 0:
        return train
    idx = np.clip(np.searchsorted(t, spike_times_s), 0, T - 1)
    np.add.at(train, idx, 1.0)
    return train

def default_config_tree(N_STEPS=3):
    """
    Return a JAX-friendly config dict (pytree) with arrays/floats only.
    """
    k_on = np.full(N_STEPS, 100.0, dtype=float)                 # 1/(uM*s)
    Kd_list = np.geomspace(0.3, 10.0, N_STEPS)                  # uM
    k_off = k_on * Kd_list                                      # 1/s
    beta = np.linspace(0.1, 1.0, N_STEPS + 1)                   # brightness

    cfg = {
        "N_STEPS": int(N_STEPS),
        "k_on": jnp.asarray(k_on),
        "k_off": jnp.asarray(k_off),
        "k_onB": jnp.array(100.0),   # 1/(uM*s)
        "Kd_B": jnp.array(10.0),     # uM
        "B_total": jnp.array(50.0),  # uM
        "G_total": jnp.array(10.0),  # uM
        "k_ex": jnp.array(2.0),      # 1/s (fit)
        "A_spk": jnp.array(0.5),     # uM·s per spike (fit)
        "tau_spk": jnp.array(0.05),  # s (fit)
        "beta": jnp.asarray(beta),   # [N+1]
        "b0": jnp.array(0.0),        # dF/F (fit)
        "b1": jnp.array(1.0),        # dF/F per brightness unit (fit)
        "ca0": jnp.array(0.1),       # uM (fit)
    }
    return cfg

def apply_params_to_cfg_tree(theta, cfg_tree):
    """
    theta = [logA, logTau, logKex, b0, logb1, ca0]
    """
    logA, logTau, logKex, b0, logb1, ca0 = theta
    new = dict(cfg_tree)
    new["A_spk"]   = jnp.exp(logA)
    new["tau_spk"] = jnp.exp(logTau)
    new["k_ex"]    = jnp.exp(logKex)
    new["b0"]      = b0
    new["b1"]      = jnp.exp(logb1)
    new["ca0"]     = jnp.clip(ca0, 1e-3, 1.0)
    return new

# ---------------------------
# JAX core: influx and ODE
# ---------------------------

def influx_series(train, t, A_spk, tau):
    """
    Causal discrete-time exponential drive with exact area A per spike.
    y[n] = exp(-dt/tau)*y[n-1] + A*(1-exp(-dt/tau))/dt * train[n]
    """
    t = jnp.asarray(t)
    train = jnp.asarray(train)
    dt = jnp.clip(t[1:] - t[:-1], 1e-9, jnp.inf)

    def step(y_prev, x):
        dt_n, u_n = x
        alpha = jnp.exp(-dt_n / tau)
        beta = A_spk * (1.0 - alpha) / dt_n
        y = alpha * y_prev + beta * u_n
        return y, y

    y0 = jnp.array(0.0)
    xs = (dt, train[1:])
    _, y_hist = lax.scan(step, y0, xs)
    J0 = train[0] * (A_spk / jnp.maximum(t[1] - t[0], 1e-9))
    J = jnp.concatenate([jnp.array([J0]), y_hist])
    return J

def indicator_equilibrium_distribution(ca, k_on, k_off, G_total):
    """
    S_j = G_total * Q_j / sum_k Q_k, with Q_0=1, Q_j = prod_{i=1..j} ca / Kd_i
    """
    Kd = k_off / k_on
    def body(Q_prev, kd):
        Q_j = Q_prev * (ca / kd)
        return Q_j, Q_j
    Q0 = jnp.array(1.0)
    _, Q_rest = lax.scan(body, Q0, Kd)
    Q = jnp.concatenate([jnp.array([Q0]), Q_rest])
    S = G_total * Q / jnp.sum(Q)
    return S  # (N+1,)

def rhs_sbm(y, Jn, cfg):
    """
    Right-hand side for SBM + buffer + extrusion.
    y = [ca, CaB, S0..SN]
    """
    ca  = y[0]
    CaB = y[1]
    S   = y[2:]                   # (N+1,)
    N   = S.shape[0] - 1

    k_on   = cfg["k_on"]
    k_off  = cfg["k_off"]
    k_onB  = cfg["k_onB"]
    Kd_B   = cfg["Kd_B"]
    B_total= cfg["B_total"]
    k_ex   = cfg["k_ex"]

    # Indicator flows
    fwd = k_on * jnp.clip(ca, 0.0) * S[:-1]   # j=0..N-1
    bwd = k_off * S[1:]                       # j=0..N-1

    # dS
    dS = jnp.zeros_like(S)
    dS = dS.at[0].set(-fwd[0] + bwd[0])
    def middle_body(j, dS_):
        return dS_.at[j].set(fwd[j-1] - bwd[j-1] - fwd[j] + bwd[j])
    dS = lax.fori_loop(1, N, middle_body, dS)
    dS = dS.at[N].set(fwd[N-1] - bwd[N-1])

    # Buffer kinetics
    koffB = k_onB * Kd_B
    dCaB = k_onB * jnp.clip(ca, 0.0) * (B_total - CaB) - koffB * CaB

    # Calcium balance
    net_bind_indicator = jnp.sum(fwd - bwd)  # consumes Ca when positive
    net_bind_buffer    = dCaB                # consumes Ca when positive
    dca = Jn - k_ex * ca - net_bind_indicator - net_bind_buffer

    return jnp.concatenate([jnp.array([dca]), jnp.array([dCaB]), dS])

@partial(jit, static_argnames=("newton_iters",))
def implicit_euler_step(y, dt, Jn, cfg, newton_iters: int = 8):
    """
    Backward-Euler step: solve z - y - dt * rhs_sbm(z, Jn, cfg) = 0 via Newton.
    cfg is a pytree of arrays/floats (dynamic arg). No function args are passed.
    """
    def F(z):
        return z - y - dt * rhs_sbm(z, Jn, cfg)

    def newton(z, _):
        Jf = jax.jacfwd(lambda z_: rhs_sbm(z_, Jn, cfg))(z)
        dim = z.shape[0]
        JG = jnp.eye(dim, dtype=z.dtype) - dt * Jf
        r = F(z)
        dx = jnp.linalg.solve(JG, -r)
        z_new = jnp.maximum(z + dx, 0.0)  # enforce non-negativity
        return z_new, None

    z0 = jnp.maximum(y + dt * rhs_sbm(y, Jn, cfg), 0.0)  # explicit Euler warm start
    z_final, _ = lax.scan(newton, z0, xs=None, length=newton_iters)
    return z_final

def simulate(cfg, t, train, newton_iters: int = 8):
    """
    Integrate SBM over t with spike train.
    Returns F_pred (T,), Y (T, 2+N+1), J (T,)
    """
    # Influx
    J = influx_series(train, t, cfg["A_spk"], cfg["tau_spk"])

    # Initial conditions
    ca0   = cfg["ca0"]
    S0    = indicator_equilibrium_distribution(ca0, cfg["k_on"], cfg["k_off"], cfg["G_total"])
    CaB0  = cfg["B_total"] * ca0 / (cfg["Kd_B"] + ca0)
    y0    = jnp.concatenate([jnp.array([ca0]), jnp.array([CaB0]), S0])

    # Time steps (non-uniform allowed)
    dt = jnp.concatenate([jnp.array([jnp.clip(t[1] - t[0], 1e-9, jnp.inf)]), t[1:] - t[:-1]])

    def step(y_prev, inputs):
        dt_n, Jn = inputs
        y_next = implicit_euler_step(y_prev, dt_n, Jn, cfg, newton_iters=newton_iters)
        return y_next, y_next

    _, Y_hist = lax.scan(step, y0, (dt, J))
    # Align trajectory to time stamps: prepend initial state
    Y_full = jnp.concatenate([y0[None, :], Y_hist[:-1, :]], axis=0)

    # Fluorescence readout
    S_traj = Y_full[:, 2:]  # (T, N+1)
    F_pred = cfg["b0"] + cfg["b1"] * (S_traj @ cfg["beta"])
    return F_pred, Y_full, J

# ---------------------------
# Loss and optimizer
# ---------------------------

def loss_with_cfg(theta, t, F, train, cfg_base, newton_iters: int):
    """
    Non-jitted wrapper to build cfg on-the-fly for grad.
    """
    cfg = apply_params_to_cfg_tree(theta, cfg_base)
    return _loss_core(theta, t, F, train, cfg, newton_iters)

@partial(jit, static_argnames=("newton_iters",))
def _loss_core(theta, t, F, train, cfg, newton_iters: int):
    F_pred, _, _ = simulate(cfg, t, train, newton_iters=newton_iters)
    resid = F_pred - F
    # Robust Huber
    delta = 1.4826 * jnp.median(jnp.abs(resid - jnp.median(resid)) + 1e-9)
    delta = jnp.maximum(delta, 1e-6)
    abs_r = jnp.abs(resid)
    huber = jnp.where(abs_r <= delta, 0.5 * (resid / delta)**2, abs_r / delta - 0.5)

    # Mild parameter regularization
    logA, logTau, logKex, b0, logb1, ca0 = theta
    pen = 1e-3 * (
        jnp.maximum(0.0, jnp.log(1e-3) - logTau)**2 + jnp.maximum(0.0, logTau - jnp.log(1.0))**2
        + jnp.maximum(0.0, 0.02 - ca0)**2 + jnp.maximum(0.0, ca0 - 0.5)**2
    )
    return jnp.mean(huber) + pen

@jit
def adam_init(theta):
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    t = jnp.array(0)
    return (theta, m, v, t)

@jit
def adam_step(state, grad, lr=5e-3, b1=0.9, b2=0.999, eps=1e-8):
    theta, m, v, t = state
    t = t + 1
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * (grad * grad)
    mhat = m / (1 - b1**t)
    vhat = v / (1 - b2**t)
    theta = theta - lr * mhat / (jnp.sqrt(vhat) + eps)
    return (theta, m, v, t)

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat', required=True, type=str, help='Path to .mat file')
    parser.add_argument('--rec-id', default=0, type=int, help='Recording index within CAttached')
    parser.add_argument('--outdir', default='results', type=str, help='Output directory')
    parser.add_argument('--n-steps', default=3, type=int, help='Sequential binding steps (N)')
    parser.add_argument('--steps', default=1500, type=int, help='Optimizer steps')
    parser.add_argument('--lr', default=5e-3, type=float, help='Adam learning rate')
    parser.add_argument('--newton-iters', default=8, type=int, help='Newton iterations per implicit step')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data (ground truth format)
    t_np, F_np, ap_times_s = load_ground_truth_mat(args.mat, recording_id=args.rec_id)
    train_np = spike_times_to_train(t_np, ap_times_s)

    # Base config tree
    cfg0 = default_config_tree(N_STEPS=args.n_steps)
    # Initialize readout to match dF/F scale
    cfg0["b0"] = jnp.array(float(np.percentile(F_np, 10)))
    cfg0["b1"] = jnp.array(max(float(np.std(F_np)), 1e-3))

    # Initial theta
    theta0 = jnp.array([
        jnp.log(cfg0["A_spk"]),    # logA
        jnp.log(cfg0["tau_spk"]),  # logTau
        jnp.log(cfg0["k_ex"]),     # logKex
        cfg0["b0"],                # b0 (dF/F baseline)
        jnp.log(cfg0["b1"]),       # logb1
        cfg0["ca0"],               # ca0 (uM)
    ], dtype=jnp.float32)

    # Data as JAX arrays
    t = jnp.asarray(t_np)
    F = jnp.asarray(F_np)
    train = jnp.asarray(train_np)

    # Optimize
    state = adam_init(theta0)
    best_theta = theta0
    best_loss = jnp.inf

    print("Starting optimization...")
    for k in range(1, args.steps + 1):
        val, grad = value_and_grad(lambda th: loss_with_cfg(th, t, F, train, cfg0, args.newton_iters))(state[0])
        state = adam_step(state, grad, lr=args.lr)
        if val < best_loss:
            best_loss = val
            best_theta = state[0]
        if (k % 100) == 0 or k == 1:
            print(f"Step {k:5d}  loss={float(val):.6f}")

    # Final simulation with best params
    cfg_fit = apply_params_to_cfg_tree(best_theta, cfg0)
    F_pred, Y_fit, J_fit = simulate(cfg_fit, t, train, newton_iters=args.newton_iters)

    # Save results
    np.savez(
        os.path.join(args.outdir, "fit_results_sbm_jax_groundtruth_fixed2.npz"),
        theta=np.array(best_theta),
        loss=float(best_loss),
        t=t_np,
        F=F_np,
        F_pred=np.array(F_pred),
        Y=np.array(Y_fit),
        J=np.array(J_fit),
        # Save config parts explicitly
        N_STEPS=np.array(cfg_fit["N_STEPS"]),
        k_on=np.array(cfg_fit["k_on"]),
        k_off=np.array(cfg_fit["k_off"]),
        k_onB=float(cfg_fit["k_onB"]),
        Kd_B=float(cfg_fit["Kd_B"]),
        B_total=float(cfg_fit["B_total"]),
        G_total=float(cfg_fit["G_total"]),
        beta=np.array(cfg_fit["beta"]),
        k_ex=float(cfg_fit["k_ex"]),
        A_spk=float(cfg_fit["A_spk"]),
        tau_spk=float(cfg_fit["tau_spk"]),
        b0=float(cfg_fit["b0"]),
        b1=float(cfg_fit["b1"]),
        ca0=float(cfg_fit["ca0"]),
    )

    # Plots
    plt.figure(figsize=(12, 4))
    plt.plot(t_np, F_np, 'k', lw=1, label='F (data)')
    plt.plot(t_np, np.array(F_pred), 'r', lw=1.25, label='F (SBM fit)')
    plt.xlabel('Time (s)')
    plt.ylabel('Fluorescence (dF/F)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'fit_fluorescence.png'), dpi=150)

    plt.figure(figsize=(12, 4))
    plt.plot(t_np, np.array(Y_fit)[:, 0], 'b', lw=1.0, label='[Ca] (uM)')
    plt.xlabel('Time (s)')
    plt.ylabel('Free Ca (uM)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'fit_free_ca.png'), dpi=150)

    beta = np.array(cfg_fit["beta"])
    S_traj = np.array(Y_fit)[:, 2:]
    contrib = S_traj * beta[None, :]
    plt.figure(figsize=(12, 4))
    for j in range(contrib.shape[1]):
        plt.plot(t_np, contrib[:, j], lw=1.0, label=f"beta*S{j}")
    plt.xlabel('Time (s)')
    plt.ylabel('Brightness-weighted state contrib')
    plt.legend(ncol=min(4, contrib.shape[1]))
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, 'state_contributions.png'), dpi=150)

    print("\nBest-fit parameters:")
    logA, logTau, logKex, b0n, logb1n, ca0 = [float(x) for x in best_theta]
    print(f"  A_spk     = {float(jnp.exp(logA)):.4g} uM·s")
    print(f"  tau_spk   = {float(jnp.exp(logTau)):.4g} s")
    print(f"  k_ex      = {float(jnp.exp(logKex)):.4g} 1/s")
    print(f"  b0        = {b0n:.4g} (dF/F)")
    print(f"  b1        = {float(jnp.exp(logb1n)):.4g} (dF/F per brightness unit)")
    print(f"  ca0       = {ca0:.4g} uM")
    print(f"\nSaved outputs to: {args.outdir}")

if __name__ == "__main__":
    main()