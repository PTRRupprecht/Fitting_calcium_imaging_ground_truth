#!/usr/bin/env python3
"""
Interactive GUI for the Broussard et al. (2024)-style biophysical calcium indicator model.

- Loads a ground-truth MATLAB file (default: CAttached_jGCaMP8s_472182_2_mini.mat).
- Bins spikes to the imaging frames.
- Simulates the Broussard model (multi-site sequential Ca2+ binding + buffers + extrusion).
- Lets you adjust parameters with sliders (no gradient descent). Updates the fit live.
- Shows RMSE and Pearson correlation in the title.
- Optional: auto-scale (b0, b1) to best align predicted fluorescence to observed (closed-form).

Notes
- Default integrator: forward Euler (fast and robust for interactive use).
- You can switch to backward Euler (implicit) if desired. It's slower but more stable for stiff cases.
- The implementation keeps the core biophysical structure; UI consolidates parameters to stay usable.

Dependencies: numpy, scipy (for loadmat), matplotlib
"""

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox


# ---------------------------
# Data IO and preprocessing
# ---------------------------

def load_ground_truth_mat(mat_path: str, recording_id: int = 0):
    """
    Load a single-neuron ground-truth recording from mini .mat file.
    Expects 'CAttached' cell array with fields:
      - fluo_time (s), fluo_mean (dF/F), events_AP (in 0.1 ms units)
    """
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    CAttached = data["CAttached"]
    if isinstance(CAttached, np.ndarray):
        rec = CAttached[recording_id]
    else:
        rec = CAttached  # single recording case
    fluo_time = np.asarray(rec.fluo_time).ravel().astype(float)
    fluo_mean = np.asarray(rec.fluo_mean).ravel().astype(float)
    events_AP = np.asarray(rec.events_AP).ravel().astype(float) if hasattr(rec, "events_AP") else np.array([])
    ap_times_s = events_AP / 1e4  # convert 0.1 ms units to seconds
    return fluo_time, fluo_mean, ap_times_s


def make_spike_train(fluo_time: np.ndarray, ap_times_s: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Bin spike times to the imaging sample times using mid-point bin edges.
    Returns spike counts per frame and the median dt.
    """
    if len(fluo_time) < 2:
        raise ValueError("fluo_time must have at least 2 samples.")
    dt = float(np.median(np.diff(fluo_time)))
    edges = np.concatenate([
        [fluo_time[0] - 0.5 * dt],
        0.5 * (fluo_time[1:] + fluo_time[:-1]),
        [fluo_time[-1] + 0.5 * dt]
    ])
    spike_counts, _ = np.histogram(ap_times_s, bins=edges)
    return spike_counts.astype(float), dt


# ---------------------------
# Broussard-style model (NumPy)
# ---------------------------

@dataclass
class Config:
    n_sites: int = 4
    n_buffers: int = 2
    extrusion: str = "linear"   # "linear" or "mm"
    method: str = "forward"     # "forward" or "backward"
    newton_max_iters: int = 12
    newton_tol: float = 1e-7
    newton_damping: float = 0.5
    clip_states: bool = True


@dataclass
class Params:
    # Indicator kinetics (vectors length n_sites)
    kon: np.ndarray      # 1/(uM*s)
    koff: np.ndarray     # 1/s
    # Brightness per occupancy state (length n_sites+1)
    phi: np.ndarray      # dimensionless
    # Indicator total concentration
    I_tot: float         # uM

    # Buffers (vectors length n_buffers)
    konB: np.ndarray     # 1/(uM*s)
    koffB: np.ndarray    # 1/s
    B_tot: np.ndarray    # uM

    # Extrusion
    k_ex: float          # 1/s (linear)
    Vmax: float          # uM/s (MM)
    Km: float            # uM (MM)

    # Baseline and spike coupling
    c_rest: float        # uM
    alpha_spike: float   # uM/s per spike

    # Fluorescence readout
    b0: float
    b1: float


def default_params(cfg: Config) -> Params:
    # Base defaults similar to prior examples
    n, nb = cfg.n_sites, cfg.n_buffers
    base_kon = np.array([100.0, 80.0, 60.0, 40.0], dtype=float)
    base_koff = np.array([150.0, 120.0, 90.0, 60.0], dtype=float)
    kon = (base_kon[:n] if n <= 4 else np.concatenate([base_kon, np.linspace(40.0, 20.0, n - 4)])).astype(float)
    koff = (base_koff[:n] if n <= 4 else np.concatenate([base_koff, np.linspace(60.0, 30.0, n - 4)])).astype(float)

    phi = np.linspace(0.0, 1.0, n + 1).astype(float)

    base_konB = np.array([100.0, 50.0], dtype=float)
    base_koffB = np.array([50.0, 25.0], dtype=float)
    base_Btot = np.array([40.0, 20.0], dtype=float)
    if nb <= 2:
        konB = base_konB[:nb]
        koffB = base_koffB[:nb]
        B_tot = base_Btot[:nb]
    else:
        add = nb - 2
        konB = np.concatenate([base_konB, np.full(add, base_konB[-1])])
        koffB = np.concatenate([base_koffB, np.full(add, base_koffB[-1])])
        B_tot = np.concatenate([base_Btot, np.full(add, base_Btot[-1])])

    return Params(
        kon=kon, koff=koff, phi=phi, I_tot=20.0,
        konB=konB, koffB=koffB, B_tot=B_tot,
        k_ex=20.0, Vmax=50.0, Km=0.3,
        c_rest=0.1, alpha_spike=5.0,
        b0=0.0, b1=1.0
    )


@dataclass
class State:
    c: float          # free Ca2+ (uM)
    I: np.ndarray     # occupancies [n_sites+1] (uM), sum = I_tot
    Bc: np.ndarray    # bound buffers [n_buffers] (uM)


def init_state(params: Params, cfg: Config) -> State:
    c = float(params.c_rest)
    I = np.zeros(cfg.n_sites + 1, dtype=float)
    I[0] = params.I_tot
    KdB = params.koffB / params.konB
    Bc = params.B_tot * c / (c + KdB)
    return State(c=c, I=I, Bc=Bc.copy())


def indicator_fluxes(c: float, I: np.ndarray, kon: np.ndarray, koff: np.ndarray):
    bind = kon * c * I[:-1]     # I_j -> I_{j+1}
    unbind = koff * I[1:]       # I_{j+1} -> I_j
    return bind, unbind


def rhs(state: State, params: Params, u_influx: float, cfg: Config) -> State:
    c, I, Bc = state.c, state.I, state.Bc

    # Indicator kinetics
    bind, unbind = indicator_fluxes(c, I, params.kon, params.koff)
    dI0 = -bind[0] + unbind[0]
    mid = bind[:-1] - unbind[:-1] - bind[1:] + unbind[1:]
    dIn = bind[-1] - unbind[-1]
    dI = np.concatenate([[dI0], mid, [dIn]])

    # Buffers
    B_free = params.B_tot - Bc
    bindB = params.konB * c * B_free
    unbindB = params.koffB * Bc
    dBc = bindB - unbindB
    J_B = float(np.sum(dBc))

    # Indicator Ca flux
    J_I = float(np.sum(bind - unbind))

    # Extrusion
    if cfg.extrusion == "linear":
        J_ex = params.k_ex * (c - params.c_rest)
    else:
        # Centered MM so J_ex(c_rest)=0
        J_ex = params.Vmax * (c / (params.Km + c) - params.c_rest / (params.Km + params.c_rest))

    # Free calcium ODE
    dc = u_influx - J_I - J_B - J_ex
    return State(c=dc, I=dI, Bc=dBc)


def clip_and_project(state: State, params: Params, cfg: Config) -> State:
    c = max(state.c, 0.0)
    I = np.maximum(state.I, 0.0)
    sumI = float(np.sum(I))
    if sumI <= 1e-16:
        I = np.zeros_like(I)
        I[0] = params.I_tot
    else:
        I *= params.I_tot / sumI
    Bc = np.clip(state.Bc, 0.0, params.B_tot)
    return State(c=c, I=I, Bc=Bc)


def forward_euler_step(state: State, params: Params, u: float, dt: float, cfg: Config) -> State:
    derivs = rhs(state, params, u, cfg)
    st = State(
        c=state.c + dt * derivs.c,
        I=state.I + dt * derivs.I,
        Bc=state.Bc + dt * derivs.Bc,
    )
    return clip_and_project(st, params, cfg) if cfg.clip_states else st


def _finite_diff_jacobian(g_func, x, eps=1e-6):
    """
    Finite-difference Jacobian of g: R^D -> R^D at x.
    Uses forward differences.
    """
    D = x.size
    gx = g_func(x)
    J = np.zeros((D, D), dtype=float)
    for i in range(D):
        dx = np.zeros_like(x)
        step = eps * (1.0 + abs(x[i]))
        dx[i] = step
        gxp = g_func(x + dx)
        J[:, i] = (gxp - gx) / step
    return J, gx


def pack_state(state: State) -> np.ndarray:
    return np.concatenate([[state.c], state.I, state.Bc])


def unpack_state(x: np.ndarray, cfg: Config) -> State:
    nI = cfg.n_sites + 1
    nb = cfg.n_buffers
    c = float(x[0])
    I = x[1:1 + nI]
    Bc = x[1 + nI:1 + nI + nb]
    return State(c=c, I=I, Bc=Bc)


def backward_euler_step(state: State, params: Params, u: float, dt: float, cfg: Config) -> State:
    """
    Implicit backward Euler: solve x_next - x - dt * f(x_next) = 0 via damped Newton.
    Uses finite-difference Jacobian (small system, interactive use).
    """
    x0 = pack_state(state)

    def f_vec(x_next: np.ndarray) -> np.ndarray:
        stn = unpack_state(x_next, cfg)
        derivs = rhs(stn, params, u, cfg)
        return pack_state(derivs)

    def g(x_next: np.ndarray) -> np.ndarray:
        return x_next - x0 - dt * f_vec(x_next)

    x = x0.copy()
    for _ in range(cfg.newton_max_iters):
        J, gx = _finite_diff_jacobian(g, x, eps=1e-7)
        try:
            dx = np.linalg.solve(J, -gx)
        except np.linalg.LinAlgError:
            # Fallback: damped gradient step
            dx = -gx * 0.1
        x_new = x + cfg.newton_damping * dx
        if np.max(np.abs(gx)) < cfg.newton_tol:
            x = x_new
            break
        x = x_new

    st_next = unpack_state(x, cfg)
    return clip_and_project(st_next, params, cfg) if cfg.clip_states else st_next


def step(state: State, params: Params, u: float, dt: float, cfg: Config) -> Tuple[State, float]:
    if cfg.method == "backward":
        st_next = backward_euler_step(state, params, u, dt, cfg)
    else:
        st_next = forward_euler_step(state, params, u, dt, cfg)
    # Emission
    y_t = params.b0 + params.b1 * float(np.dot(params.phi, st_next.I))
    return st_next, y_t


def simulate(params: Params, cfg: Config, spikes: np.ndarray, dt: float) -> np.ndarray:
    """
    Simulate fluorescence for a given spike train.
    """
    T = spikes.size
    st = init_state(params, cfg)
    y = np.zeros(T, dtype=float)
    for t in range(T):
        u = params.alpha_spike * spikes[t]  # uM/s
        st, y[t] = step(st, params, u, dt, cfg)
    return y


# ---------------------------
# GUI
# ---------------------------

class BroussardGUI:
    def __init__(self, mat_path=None, recording_id=0):
        self.cfg = Config(n_sites=4, n_buffers=2, extrusion="linear", method="forward")
        self.params = default_params(self.cfg)

        # Base profiles for site-wise kinetics and brightness (scaled by sliders)
        self.kon_base = self.params.kon.copy()   # e.g., [100, 80, 60, 40]
        self.koff_base = self.params.koff.copy() # e.g., [150, 120, 90, 60]
        self.phi_base = self.params.phi.copy()   # linear 0..1

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(13, 7))
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.35, top=0.92)
        self.ax.set_title("Broussard biophysical model — interactive parameters (no gradient descent)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("dF/F")

        # Data placeholders
        self.fluo_time = np.array([])
        self.fluo_mean = np.array([])
        self.ap_times_s = np.array([])
        self.spike_train = np.array([])
        self.dt = 0.0

        # Plots
        (self.obs_line,) = self.ax.plot([], [], "k", lw=1.0, label="Observed dF/F")
        (self.fit_line,) = self.ax.plot([], [], color="tab:orange", lw=1.5, label="Broussard model")
        self.spike_lc = None
        self.ax.legend(loc="upper right")

        self._init_controls()

        # Auto-load default or provided file
        default_candidate = "CAttached_jGCaMP8s_472182_2_mini.mat"
        path_to_load = None
        if mat_path and os.path.isfile(mat_path):
            path_to_load = mat_path
        elif os.path.isfile(default_candidate):
            path_to_load = default_candidate

        if path_to_load:
            self.path_box.set_val(path_to_load)
            self._load_from_boxes()
        else:
            self.ax.text(0.02, 0.95, "Enter path to .mat and click Load", transform=self.ax.transAxes,
                         fontsize=10, va="top", ha="left", color="gray")

    def _init_controls(self):
        # Path and recording controls
        ax_path = self.fig.add_axes([0.08, 0.28, 0.56, 0.04])
        self.path_box = TextBox(ax_path, "MAT file: ", initial="", color="w", hovercolor="#f0f0f0")
        ax_rec = self.fig.add_axes([0.67, 0.28, 0.06, 0.04])
        self.rec_box = TextBox(ax_rec, "Rec#", initial="0", color="w", hovercolor="#f0f0f0")
        ax_load = self.fig.add_axes([0.75, 0.28, 0.08, 0.04])
        self.load_btn = Button(ax_load, "Load", color="#e0ffe0", hovercolor="#c0ffc0")
        self.load_btn.on_clicked(lambda evt: self._load_from_boxes())

        # Extrusion and integrator radios
        ax_ex = self.fig.add_axes([0.86, 0.72, 0.10, 0.12])
        self.ex_radio = RadioButtons(ax_ex, ("linear", "mm"))
        self.ex_radio.on_clicked(self._on_extrusion_changed)

        ax_m = self.fig.add_axes([0.86, 0.58, 0.10, 0.12])
        self.m_radio = RadioButtons(ax_m, ("forward", "backward"))
        self.m_radio.on_clicked(self._on_method_changed)

        # Sliders layout
        y0 = 0.22; h = 0.03; gap = 0.006
        xL = 0.08; w = 0.24
        xC = 0.36
        xR = 0.64

        def add_slider(label, axpos, vmin, vmax, vinit):
            ax = self.fig.add_axes(axpos)
            s = Slider(ax, label, vmin, vmax, valinit=vinit)
            s.on_changed(self._on_slider)
            return s

        # Left column — indicator kinetics and brightness
        self.s_kon_scale  = add_slider("kon scale",  [xL, y0, w, h], 0.01, 0.2, 0.03)
        self.s_koff_scale = add_slider("koff scale", [xL, y0-(h+gap), w, h], 0.01, 2.0, 0.4)
        self.s_I_tot      = add_slider("I_tot (uM)", [xL, y0-2*(h+gap), w, h], 0.1, 100.0, self.params.I_tot)
        self.s_phi_scale  = add_slider("phi scale",  [xL, y0-3*(h+gap), w, h], 0.1, 2.0, 1.7)
        self.s_phi_bias   = add_slider("phi bias",   [xL, y0-4*(h+gap), w, h], -0.5, 0.5, 0.0)

        # Center column — buffers (two buffers)
        self.s_konB1  = add_slider("konB1",   [xC, y0, w, h], 0.01, 10, self.params.konB[0] if self.cfg.n_buffers>=1 else 0.1)
        self.s_koffB1 = add_slider("koffB1",  [xC, y0-(h+gap), w, h], 0.001, 10.0, self.params.koffB[0] if self.cfg.n_buffers>=1 else 0.5)
        self.s_Btot1  = add_slider("B_tot1",  [xC, y0-2*(h+gap), w, h], 0.0, 100.0, self.params.B_tot[0] if self.cfg.n_buffers>=1 else 30.0)

        self.s_konB2  = add_slider("konB2",   [xC, y0-3*(h+gap), w, h], 0.01, 10.0, self.params.konB[1] if self.cfg.n_buffers>=2 else 2.0)
        self.s_koffB2 = add_slider("koffB2",  [xC, y0-4*(h+gap), w, h], 0.001, 10.0, self.params.koffB[1] if self.cfg.n_buffers>=2 else 1.0)
        self.s_Btot2  = add_slider("B_tot2",  [xC, y0-5*(h+gap), w, h], 0.0, 10.0, self.params.B_tot[1] if self.cfg.n_buffers>=2 else 1.0)

        # Right column — extrusion, baseline, coupling, readout
        self.s_k_ex   = add_slider("k_ex (1/s)", [xR, y0, w, h], 1e-2, 50.0, 20.0)
        self.s_Vmax   = add_slider("Vmax (uM/s)", [xR, y0-(h+gap), w, h], 1e-3, 500.0, 205.0)
        self.s_Km     = add_slider("Km (uM)", [xR, y0-2*(h+gap), w, h], 1e-5, 10.0, self.params.Km)

        self.s_c_rest = add_slider("c_rest (uM)", [xR, y0-3*(h+gap), w, h], 1e-5, 5.0, self.params.c_rest)
        self.s_alpha  = add_slider("alpha_spike", [xR, y0-4*(h+gap), w, h], 100.0, 1e4, 200.0)

        self.s_b0     = add_slider("b0", [xR, y0-5*(h+gap), w, h], -10.0, 10.0, 0.0)
        self.s_b1     = add_slider("b1", [xR, y0-6*(h+gap), w, h], 1e-3, 1e0, 0.0)

        # Buttons
        ax_reset = self.fig.add_axes([0.08, 0.02, 0.10, 0.05])
        self.reset_btn = Button(ax_reset, "Reset", color="#fff0e0", hovercolor="#ffd0b0")
        self.reset_btn.on_clicked(self._reset_params)

        ax_autoscale = self.fig.add_axes([0.20, 0.02, 0.18, 0.05])
        self.autoscale_btn = Button(ax_autoscale, "Auto-scale b0,b1", color="#f0f0ff", hovercolor="#d0d0ff")
        self.autoscale_btn.on_clicked(self._auto_scale_b0b1)

    # ---- callbacks ----
    def _on_extrusion_changed(self, label):
        self.cfg.extrusion = str(label).strip()
        self._update_prediction()

    def _on_method_changed(self, label):
        self.cfg.method = str(label).strip()
        self._update_prediction()

    def _on_slider(self, val):
        self._update_prediction()

    def _reset_params(self, evt):
        # Reset scales
        self.s_kon_scale.reset()
        self.s_koff_scale.reset()
        self.s_I_tot.reset()
        self.s_phi_scale.reset()
        self.s_phi_bias.reset()

        # Buffers
        self.s_konB1.reset(); self.s_koffB1.reset(); self.s_Btot1.reset()
        self.s_konB2.reset(); self.s_koffB2.reset(); self.s_Btot2.reset()

        # Extrusion and readout
        self.s_k_ex.reset(); self.s_Vmax.reset(); self.s_Km.reset()
        self.s_c_rest.reset(); self.s_alpha.reset()
        self.s_b0.reset(); self.s_b1.reset()

        # Radios
        for i, lab in enumerate(self.ex_radio.labels):
            if lab.get_text() == "linear":
                self.ex_radio.set_active(i); break
        for i, lab in enumerate(self.m_radio.labels):
            if lab.get_text() == "forward":
                self.m_radio.set_active(i); break

        self._update_prediction()

    def _auto_scale_b0b1(self, evt):
        """
        Solve for (b0, b1) minimizing ||b0 + b1*y_pred - y_obs||^2 in closed form.
        """
        if self.fluo_time.size == 0 or self.spike_train.size == 0:
            return
        p = self._current_params()
        y_pred = simulate(p, self.cfg, self.spike_train, self.dt)
        y_obs = self.fluo_mean
        X = np.vstack([np.ones_like(y_pred), y_pred]).T
        try:
            coef, _, _, _ = np.linalg.lstsq(X, y_obs, rcond=None)
            self.s_b0.set_val(float(coef[0]))
            self.s_b1.set_val(float(coef[1]))
        except np.linalg.LinAlgError:
            pass

    def _browse_file(self, evt):
        # Optional: not wired by default (Tk not guaranteed)
        pass

    def _load_from_boxes(self):
        path = self.path_box.text.strip().strip('"')
        try:
            rec_id = int(self.rec_box.text.strip())
        except Exception:
            rec_id = 0
            self.rec_box.set_val("0")
        if not os.path.isfile(path):
            print(f"[WARN] File not found: {path}")
            return
        try:
            fluo_time, fluo_mean, ap_times_s = load_ground_truth_mat(path, rec_id)
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            return

        spike_train, dt = make_spike_train(fluo_time, ap_times_s)
        self.fluo_time = fluo_time
        self.fluo_mean = fluo_mean
        self.ap_times_s = ap_times_s
        self.spike_train = spike_train
        self.dt = dt

        # Plot observed
        self.obs_line.set_data(self.fluo_time, self.fluo_mean)
        y_min = float(np.nanmin(self.fluo_mean)) if self.fluo_mean.size else -0.1
        y_max = float(np.nanmax(self.fluo_mean)) if self.fluo_mean.size else 0.1
        yr = y_max - y_min if y_max > y_min else 1.0
        self.ax.set_ylim(y_min - 0.1 * yr, y_max + 0.1 * yr)
        self.ax.set_xlim(self.fluo_time[0], self.fluo_time[-1])

        # Spikes overlay as vertical ticks near bottom
        if self.spike_lc is not None:
            self.spike_lc.remove()
            self.spike_lc = None
        if self.ap_times_s.size > 0:
            y0 = y_min - 0.05 * yr
            y1 = y_min - 0.02 * yr
            segs = [((t, y0), (t, y1)) for t in self.ap_times_s]
            self.spike_lc = LineCollection(segs, colors="r", linewidths=0.6, alpha=0.7)
            self.ax.add_collection(self.spike_lc)

        self._update_prediction()

    def _current_params(self) -> Params:
        # Build parameter vector from sliders, scaling base arrays
        kon = self.kon_base * float(self.s_kon_scale.val)
        koff = self.koff_base * float(self.s_koff_scale.val)
        phi = np.clip(self.phi_base * float(self.s_phi_scale.val) + float(self.s_phi_bias.val), 0.0, 2.0)

        # Buffers
        konB = np.array([float(self.s_konB1.val), float(self.s_konB2.val)])[:self.cfg.n_buffers]
        koffB = np.array([float(self.s_koffB1.val), float(self.s_koffB2.val)])[:self.cfg.n_buffers]
        B_tot = np.array([float(self.s_Btot1.val), float(self.s_Btot2.val)])[:self.cfg.n_buffers]

        # Extrusion and other scalars
        return Params(
            kon=kon, koff=koff, phi=phi, I_tot=float(self.s_I_tot.val),
            konB=konB, koffB=koffB, B_tot=B_tot,
            k_ex=float(self.s_k_ex.val), Vmax=float(self.s_Vmax.val), Km=float(self.s_Km.val),
            c_rest=float(self.s_c_rest.val), alpha_spike=float(self.s_alpha.val),
            b0=float(self.s_b0.val), b1=float(self.s_b1.val)
        )

    def _update_prediction(self):
        if self.fluo_time.size == 0 or self.spike_train.size == 0:
            self.fit_line.set_data([], [])
            self.fig.canvas.draw_idle()
            return

        params = self._current_params()

        # Keep phi length consistent and non-negative
        if params.phi.size != (self.cfg.n_sites + 1):
            params.phi = np.linspace(0.0, 1.0, self.cfg.n_sites + 1)

        # Simulate
        y_fit = simulate(params, self.cfg, self.spike_train, self.dt)
        self.fit_line.set_data(self.fluo_time, y_fit)

        # Metrics
        y_obs = self.fluo_mean
        if y_obs.size == y_fit.size:
            rmse = float(np.sqrt(np.mean((y_fit - y_obs) ** 2)))
            a = y_fit - np.mean(y_fit); b = y_obs - np.mean(y_obs)
            denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12
            corr = float(np.sum(a * b) / denom)
            self.ax.set_title(
                f"Broussard model — RMSE: {rmse:.4f}, Corr: {corr:.3f}  |  Extrusion: {self.cfg.extrusion}  |  Integrator: {self.cfg.method}"
            )
        else:
            self.ax.set_title(
                f"Broussard model  |  Extrusion: {self.cfg.extrusion}  |  Integrator: {self.cfg.method}"
            )

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive Broussard biophysical model GUI")
    parser.add_argument("mat_file", nargs="?", default=None, help="Path to .mat file")
    parser.add_argument("--recording", type=int, default=0, help="Recording index (0-based)")
    args = parser.parse_args()

    # Default file if not provided
    mat_path = args.mat_file
    if mat_path is None:
        default_candidate = "CAttached_jGCaMP8s_472182_2_mini.mat"
        if os.path.isfile(default_candidate):
            mat_path = default_candidate

    app = BroussardGUI(mat_path=mat_path, recording_id=args.recording)
    app.show()


if __name__ == "__main__":
    main()