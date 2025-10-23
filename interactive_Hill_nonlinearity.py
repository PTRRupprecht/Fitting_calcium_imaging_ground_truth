#!/usr/bin/env python3
import argparse
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from scipy.signal import fftconvolve

# Try to enable file dialog (optional)
try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except Exception:
    HAS_TK = False

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


def make_spike_train(fluo_time: np.ndarray, ap_times_s: np.ndarray):
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
# Deneux-style forward model
# ---------------------------
def bi_exp_kernel(t: np.ndarray, tau_rise: float, tau_decay: float) -> np.ndarray:
    """
    h(t) = (exp(-t/tau_decay) - exp(-t/tau_rise)) for t>=0, 0 else; normalized to unit peak.
    """
    t = np.asarray(t)
    h = np.zeros_like(t, dtype=float)
    tau_rise = max(tau_rise, 1e-6)
    tau_decay = max(tau_decay, tau_rise + 1e-6)
    mask = t >= 0
    h[mask] = np.exp(-t[mask] / tau_decay) - np.exp(-t[mask] / tau_rise)
    peak = h.max() if h.size else 0.0
    if peak > 0:
        h = h / peak
    return h


def hill_nl(c: np.ndarray, kd: float, n: float) -> np.ndarray:
    """
    Hill nonlinearity: g(c) = c^n / (c^n + kd^n). Parameters constrained for stability.
    """
    c = np.clip(np.asarray(c, dtype=float), 0.0, None)
    n = max(float(n), 1e-3)
    kd = max(float(kd), 1e-9)
    c_n = np.power(c, n)
    kd_n = kd ** n
    return c_n / (c_n + kd_n)


def simulate_fluorescence(params: dict, spike_train: np.ndarray, dt: float, nonlinearity: str = "hill") -> np.ndarray:
    """
    Deterministic simulation (no noise in GUI for stability):
      - spike_train -> convolve with bi-exp kernel -> optional Hill nonlinearity -> scale+offset
    params keys: tau_rise, tau_decay, amp, kd, n, f0
    """
    T = len(spike_train)
    tau_rise = float(params["tau_rise"])
    tau_decay = float(params["tau_decay"])
    amp = float(params["amp"])
    kd = float(params["kd"])
    n = float(params["n"])
    f0 = float(params["f0"])

    # Kernel length: 5*tau_decay
    kT = int(np.ceil(max(1.0, 5.0 * tau_decay / dt)))
    tker = np.arange(kT) * dt
    h = bi_exp_kernel(tker, tau_rise, tau_decay)

    # Convolution (fast FFT)
    c = fftconvolve(spike_train, h, mode="full")[:T]

    if nonlinearity == "hill":
        g = hill_nl(c, kd=kd, n=n)
    elif nonlinearity == "linear":
        g = c
    else:
        g = hill_nl(c, kd=kd, n=n)

    F = f0 + amp * g
    return F


# ---------------------------
# GUI
# ---------------------------
class CalciumGUI:
    def __init__(self, mat_path=None, recording_id=0):
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.35, top=0.92)

        self.ax.set_title("Interactive spike-to-fluorescence model (Deneux-style)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("dF/F")

        # Defaults
        self.fluo_time = np.array([])
        self.fluo_mean = np.array([])
        self.ap_times_s = np.array([])
        self.spike_train = np.array([])
        self.dt = 0.0
        self.nonlinearity = "hill"

        # Reasonable default parameter ranges for jGCaMP8s; tweak as needed
        self.param_bounds = {
            "tau_rise": (0.005, 0.060),   # seconds
            "tau_decay": (0.10, 4.00),    # seconds
            "amp": (0.01, 100.00),          # dF/F scale
            "log10_kd": (-4.0, 4),      # kd on log10 scale => kd in [1e-4, XX]
            "n": (0.5, 5.0),              # Hill exponent
            "f0": (-0.1, 0.1),            # baseline dF/F
        }
        self.params = {
            "tau_rise": 0.02,
            "tau_decay": 0.6,
            "amp": 15.0,
            "kd": 0.7,
            "n": 2.0,
            "f0": 0.0,
        }

        # Plot elements
        (self.obs_line,) = self.ax.plot([], [], "k", lw=1.0, label="Observed dF/F")
        (self.fit_line,) = self.ax.plot([], [], color="tab:orange", lw=1.5, label="Model prediction")
        self.spike_lc = None
        self.ax.legend(loc="upper right")

        # Controls
        self._init_controls()

        # Load initial file if provided
        if mat_path is not None and os.path.isfile(mat_path):
            self.path_box.set_val(mat_path)
            self._load_from_boxes()
        else:
            self.ax.text(0.02, 0.95, "Enter path to .mat and click Load", transform=self.ax.transAxes,
                         fontsize=10, va="top", ha="left", color="gray")

    def _init_controls(self):
        # Text boxes for path and recording idx
        ax_path = self.fig.add_axes([0.08, 0.28, 0.56, 0.04])
        self.path_box = TextBox(ax_path, "MAT file: ", initial="", color="w", hovercolor="#f0f0f0")

        ax_rec = self.fig.add_axes([0.67, 0.28, 0.06, 0.04])
        self.rec_box = TextBox(ax_rec, "Rec#", initial="0", color="w", hovercolor="#f0f0f0")

        ax_load = self.fig.add_axes([0.75, 0.28, 0.08, 0.04])
        self.load_btn = Button(ax_load, "Load", color="#e0ffe0", hovercolor="#c0ffc0")
        self.load_btn.on_clicked(lambda evt: self._load_from_boxes())

        if HAS_TK:
            ax_browse = self.fig.add_axes([0.85, 0.28, 0.08, 0.04])
            self.browse_btn = Button(ax_browse, "Browse", color="#e0f0ff", hovercolor="#c0e0ff")
            self.browse_btn.on_clicked(self._browse_file)

        # Radio buttons for nonlinearity
        ax_radio = self.fig.add_axes([0.86, 0.72, 0.10, 0.15])
        self.radio = RadioButtons(ax_radio, ("hill", "linear"))
        self.radio.on_clicked(self._on_nl_changed)

        # Sliders
        y0 = 0.22
        h = 0.03
        gap = 0.005
        x0 = 0.08
        w = 0.35
        x1 = 0.55

        def add_slider(label, axpos, valmin, valmax, valinit):
            ax = self.fig.add_axes(axpos)
            s = Slider(ax, label, valmin, valmax, valinit=valinit)
            s.on_changed(self._on_slider_changed)
            return s

        # Left column sliders
        self.s_tau_rise = add_slider("tau_rise (s)", [x0, y0, w, h],
                                     *self.param_bounds["tau_rise"], self.params["tau_rise"])
        self.s_tau_decay = add_slider("tau_decay (s)", [x0, y0 - (h+gap), w, h],
                                      *self.param_bounds["tau_decay"], self.params["tau_decay"])
        self.s_amp = add_slider("amp (dF/F)", [x0, y0 - 2*(h+gap), w, h],
                                *self.param_bounds["amp"], self.params["amp"])

        # Right column sliders
        log_kd_init = np.log10(max(self.params["kd"], 1e-9))
        self.s_log10_kd = add_slider("log10(kd)", [x1, y0, w, h],
                                     *self.param_bounds["log10_kd"], log_kd_init)
        self.s_n = add_slider("Hill n", [x1, y0 - (h+gap), w, h],
                              *self.param_bounds["n"], self.params["n"])
        self.s_f0 = add_slider("f0 (dF/F)", [x1, y0 - 2*(h+gap), w, h],
                               *self.param_bounds["f0"], self.params["f0"])

        # Reset and export buttons
        ax_reset = self.fig.add_axes([0.08, 0.02, 0.10, 0.05])
        self.reset_btn = Button(ax_reset, "Reset", color="#fff0e0", hovercolor="#ffd0b0")
        self.reset_btn.on_clicked(self._reset_params)

        ax_export = self.fig.add_axes([0.20, 0.02, 0.18, 0.05])
        self.export_btn = Button(ax_export, "Export params", color="#f0f0ff", hovercolor="#d0d0ff")
        self.export_btn.on_clicked(self._export_params)

    # ---- callbacks ----
    def _browse_file(self, evt):
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(filetypes=[("MAT files", "*.mat"), ("All files", "*.*")])
        root.destroy()
        if path:
            self.path_box.set_val(path)
            self._load_from_boxes()

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

        # Update plot data
        self.obs_line.set_data(self.fluo_time, self.fluo_mean)
        # Reset y-limits with small margin
        y_min = float(np.nanmin(self.fluo_mean)) if self.fluo_mean.size else -0.1
        y_max = float(np.nanmax(self.fluo_mean)) if self.fluo_mean.size else 0.1
        yr = y_max - y_min if y_max > y_min else 1.0
        self.ax.set_ylim(y_min - 0.1*yr, y_max + 0.1*yr)
        self.ax.set_xlim(self.fluo_time[0], self.fluo_time[-1])

        # Draw spike markers as a LineCollection for efficiency
        if self.spike_lc is not None:
            self.spike_lc.remove()
            self.spike_lc = None
        if self.ap_times_s.size > 0:
            # Put spikes near bottom
            y0 = y_min - 0.05 * yr
            y1 = y_min - 0.02 * yr
            segs = [((t, y0), (t, y1)) for t in self.ap_times_s]
            self.spike_lc = LineCollection(segs, colors="r", linewidths=0.6, alpha=0.7)
            self.ax.add_collection(self.spike_lc)

        # Update prediction
        self._update_prediction()

    def _on_nl_changed(self, label):
        self.nonlinearity = str(label).strip()
        self._update_prediction()

    def _on_slider_changed(self, val):
        self._update_prediction()

    def _reset_params(self, evt):
        self.s_tau_rise.reset()
        self.s_tau_decay.reset()
        self.s_amp.reset()
        self.s_log10_kd.reset()
        self.s_n.reset()
        self.s_f0.reset()
        self.nonlinearity = "hill"
        # Update radio buttons visually
        for i, label in enumerate(self.radio.labels):
            if label.get_text() == "hill":
                self.radio.set_active(i)
                break
        self._update_prediction()

    def _export_params(self, evt):
        p = self._current_params()
        print("[PARAMS]", p)

    def _current_params(self):
        kd = 10.0 ** self.s_log10_kd.val
        return {
            "tau_rise": float(self.s_tau_rise.val),
            "tau_decay": float(self.s_tau_decay.val),
            "amp": float(self.s_amp.val),
            "kd": float(kd),
            "n": float(self.s_n.val),
            "f0": float(self.s_f0.val),
        }

    def _update_prediction(self):
        if self.fluo_time.size == 0 or self.spike_train.size == 0:
            self.fit_line.set_data([], [])
            self.fig.canvas.draw_idle()
            return

        p = self._current_params()

        # Enforce tau_rise < tau_decay for stability
        if p["tau_decay"] <= p["tau_rise"]:
            p["tau_decay"] = p["tau_rise"] + 1e-4
            self.s_tau_decay.set_val(p["tau_decay"])

        y_fit = simulate_fluorescence(p, self.spike_train, self.dt, nonlinearity=self.nonlinearity)
        self.fit_line.set_data(self.fluo_time, y_fit)

        # Simple metrics in title
        y_obs = self.fluo_mean
        if y_obs.size == y_fit.size:
            rmse = float(np.sqrt(np.mean((y_fit - y_obs) ** 2)))
            # correlation
            a = y_fit - np.mean(y_fit)
            b = y_obs - np.mean(y_obs)
            denom = np.sqrt(np.sum(a*a) * np.sum(b*b)) + 1e-12
            corr = float(np.sum(a*b) / denom)
            self.ax.set_title(f"Interactive model — RMSE: {rmse:.4f}, Corr: {corr:.3f}  |  NL: {self.nonlinearity}")
        else:
            self.ax.set_title(f"Interactive model  |  NL: {self.nonlinearity}")

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive calcium model GUI")
    parser.add_argument("mat_file", nargs="?", default=None, help="Path to .mat file")
    parser.add_argument("--recording", type=int, default=0, help="Recording index (0-based)")
    args = parser.parse_args()

    app = CalciumGUI(mat_path=args.mat_file, recording_id=args.recording)
    app.show()


if __name__ == "__main__":
    main()