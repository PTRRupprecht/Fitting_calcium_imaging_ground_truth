#!/usr/bin/env python3
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

# ---------------------------
# 1) Data IO
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


def make_spike_train(fluo_time: np.ndarray, ap_times_s: np.ndarray):
    """
    Bin spike times to imaging samples using mid-point bin edges.
    Returns spike counts per frame and median dt.
    """
    dt = float(np.median(np.diff(fluo_time)))
    edges = np.concatenate([
        [fluo_time[0] - 0.5 * dt],
        0.5 * (fluo_time[1:] + fluo_time[:-1]),
        [fluo_time[-1] + 0.5 * dt],
    ])
    spike_counts, _ = np.histogram(ap_times_s, bins=edges)
    return spike_counts.astype(float), dt


# ---------------------------
# 2) Deneux-style forward model (JAX)
# ---------------------------
import jax
import jax.numpy as jnp

def exp_filter_scan(x, a):
    """
    Causal first-order IIR: y[n] = a * y[n-1] + x[n].
    Implements convolution with impulse response g[n] = a^n (n>=0).
    """
    def step(y_prev, x_t):
        y_t = a * y_prev + x_t
        return y_t, y_t
    _, y_series = jax.lax.scan(step, 0.0, x)
    return y_series

def hill_nl(c, kd, n):
    c = jnp.clip(c, 0.0, jnp.inf)
    n = jnp.maximum(n, 1e-3)
    kd = jnp.maximum(kd, 1e-9)
    c_n = c**n
    kd_n = kd**n
    return c_n / (c_n + kd_n)

def simulate(params, spike_train, dt, nonlinearity="hill"):
    """
    Parameterization:
      tau_rise = softplus(tau_rise_raw) + eps
      tau_decay = tau_rise + softplus(tau_gap_raw) + eps  (ensures tau_decay > tau_rise)
      amp, kd > 0 via softplus, n >= 0.5, f0 free
    Convolution:
      c = (exp-filter with a_d) - (exp-filter with a_r), then divide by peak of h to normalize.
    """
    # Constrained params
    tau_rise = jax.nn.softplus(params["tau_rise_raw"]) + 1e-4
    tau_gap  = jax.nn.softplus(params["tau_gap_raw"]) + 1e-4
    tau_decay = tau_rise + tau_gap

    amp = jax.nn.softplus(params["amp_raw"])
    kd  = jax.nn.softplus(params["kd_raw"])
    n   = jax.nn.softplus(params["n_raw"]) + 0.5
    f0  = params["f0_raw"]

    # Discrete decay factors
    a_r = jnp.exp(-dt / tau_rise)
    a_d = jnp.exp(-dt / tau_decay)

    # Two causal filters and difference -> convolution with (a_d^n - a_r^n)
    y_r = exp_filter_scan(spike_train, a_r)
    y_d = exp_filter_scan(spike_train, a_d)
    c_raw = y_d - y_r

    # Normalize kernel to unit peak (discrete-time peak near t* from continuous case)
    # t* = (tau_d * tau_r)/(tau_d - tau_r) * ln(tau_d/tau_r); discrete peak index n* ≈ round(t*/dt)
    t_star = (tau_decay * tau_rise) / (tau_decay - tau_rise) * jnp.log(tau_decay / tau_rise)
    n_star = jnp.maximum(0, jnp.floor(t_star / dt + 0.5)).astype(jnp.int32)
    h_peak = jnp.power(a_d, n_star) - jnp.power(a_r, n_star)
    h_peak = jnp.where(h_peak > 1e-8, h_peak, 1.0)  # guard against degenerate cases

    c = c_raw / h_peak

    # Nonlinearity and affine readout
    g = hill_nl(c, kd, n) if nonlinearity == "hill" else c
    F = f0 + amp * g
    return F

# ---------------------------
# 3) Loss, init, and training
# ---------------------------
def mse_loss(params, spike_train, dt, y_obs, nonlinearity="hill"):
    y_hat = simulate(params, spike_train, dt, nonlinearity)
    return jnp.mean((y_hat - y_obs) ** 2)


def inv_softplus(x):
    # Numerically stable inverse for initialization
    x = jnp.asarray(x)
    return jnp.log(jnp.expm1(x))


def init_params(rng):
    """
    Initialize around plausible jGCaMP8s values (tune as needed).
    """
    p = {
        "tau_rise_raw": inv_softplus(0.02),           # ~20 ms rise
        "tau_gap_raw": inv_softplus(0.6 - 0.02),      # decay - rise ~ 0.58 s
        "amp_raw": inv_softplus(0.2),                  # dF/F scale
        "kd_raw": inv_softplus(0.05),
        "n_raw": inv_softplus(1.5),                    # n ~2.0 after +0.5
        "f0_raw": jnp.array(0.0),
    }
    # Small jitter
    noise = 0.05 * jax.random.normal(rng, (6,))
    keys = list(p.keys())
    for i, k in enumerate(keys):
        if k.endswith("_raw"):
            p[k] = p[k] + noise[i]
    return p


def make_train_step(optimizer, nonlinearity: str):
    """
    Create a train_step that closes over the string 'nonlinearity',
    so we don't pass strings through jit.
    """
    @jax.jit
    def loss_and_grads(params, spike_train, dt, y_obs):
        return jax.value_and_grad(mse_loss)(params, spike_train, dt, y_obs, nonlinearity)

    def train_step(params, opt_state, spike_train, dt, y_obs):
        loss, grads = loss_and_grads(params, spike_train, dt, y_obs)
        updates, opt_state2 = optimizer.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss

    return train_step


# ---------------------------
# 4) Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JAX fitting of spike->fluorescence model")
    parser.add_argument("--mat", type=str, required=True, help="Path to .mat ground-truth file")
    parser.add_argument("--recording", type=int, default=0, help="Recording index (0-based)")
    parser.add_argument("--nl", type=str, default="hill", choices=["hill", "linear"], help="Nonlinearity")
    parser.add_argument("--steps", type=int, default=4000, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate")
    args = parser.parse_args()

    # Load data
    fluo_time, fluo_mean, ap_times_s = load_ground_truth_mat(args.mat, args.recording)
    spike_train_np, dt_f = make_spike_train(fluo_time, ap_times_s)

    # Move to JAX types
    y_obs = jnp.asarray(fluo_mean, dtype=jnp.float32)
    spike_train = jnp.asarray(spike_train_np, dtype=jnp.float32)
    dt = jnp.asarray(dt_f, dtype=jnp.float32)

    # Optimizer and params
    optimizer = optax.adam(args.lr)
    params = init_params(jax.random.PRNGKey(0))
    opt_state = optimizer.init(params)

    # Build the step function WITHOUT passing strings through jit
    train_step = make_train_step(optimizer, args.nl)

    # Train
    print(f"Training with nonlinearity='{args.nl}', steps={args.steps}, lr={args.lr}")
    for step in range(1, args.steps + 1):
        params, opt_state, loss = train_step(params, opt_state, spike_train, dt, y_obs)
        if step % max(100, args.steps // 40) == 0 or step == 1:
            print(f"step {step:5d}  loss {float(loss):.6f}")

    # Predict with fitted params
    y_fit = simulate(params, spike_train, dt, args.nl)

    # Map raw -> constrained for reporting
    def to_constrained(p):
        tau_rise = float(jax.nn.softplus(p["tau_rise_raw"]) + 1e-4)
        tau_gap = float(jax.nn.softplus(p["tau_gap_raw"]) + 1e-4)
        tau_decay = tau_rise + tau_gap
        amp = float(jax.nn.softplus(p["amp_raw"]))
        kd = float(jax.nn.softplus(p["kd_raw"]))
        n = float(jax.nn.softplus(p["n_raw"]) + 0.5)
        f0 = float(p["f0_raw"])
        return {"tau_rise": tau_rise, "tau_decay": tau_decay, "amp": amp, "kd": kd, "n": n, "f0": f0}

    fitted = to_constrained(params)
    print("Fitted parameters:", fitted)

    # Metrics
    def corr(a, b):
        a = a - jnp.mean(a); b = b - jnp.mean(b)
        return float(jnp.sum(a*b) / (jnp.sqrt(jnp.sum(a*a)*jnp.sum(b*b)) + 1e-12))
    rmse = float(jnp.sqrt(jnp.mean((y_fit - y_obs) ** 2)))
    cc = corr(y_fit, y_obs)
    print(f"RMSE: {rmse:.6f}  Corr: {cc:.4f}")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(fluo_time, np.asarray(y_obs), "k", lw=1, label="Observed")
    plt.plot(fluo_time, np.asarray(y_fit), "C1", lw=1.5, label="JAX fit")
    # Spike ticks (thinned)
    if len(ap_times_s) > 0:
        ymin = float(np.min(fluo_mean)); ymax = float(np.max(fluo_mean))
        yr = ymax - ymin if ymax > ymin else 1.0
        y0, y1 = ymin - 0.02*yr, ymin - 0.06*yr
        step = max(1, len(ap_times_s)//400)
        for t in ap_times_s[::step]:
            plt.plot([t, t], [y1, y0], "r", alpha=0.25, lw=0.6)
    plt.xlabel("Time (s)"); plt.ylabel("dF/F")
    plt.title(f"Deneux-style fit in JAX — RMSE {rmse:.4f}, Corr {cc:.3f}, NL={args.nl}")
    plt.legend(); plt.tight_layout(); plt.show()