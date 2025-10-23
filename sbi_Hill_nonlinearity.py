# %% Optimized Setup
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from typing import Tuple, Dict
from functools import lru_cache

# sbi
from sbi.inference import SNPE
from sbi.utils import BoxUniform

# ---------------------------
# 1) Data loading (unchanged)
# ---------------------------

def load_ground_truth_mat(mat_path: str, recording_id: int = 0):
    """Load a single-neuron ground-truth recording from .mat files."""
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    CAttached = data["CAttached"]

    if isinstance(CAttached, np.ndarray):
        rec = CAttached[recording_id]
    else:
        rec = CAttached

    fluo_time = np.asarray(rec.fluo_time).ravel()
    fluo_mean = np.asarray(rec.fluo_mean).ravel()
    events_AP = np.asarray(rec.events_AP).ravel()
    ap_times_s = events_AP / 1e4

    return fluo_time, fluo_mean, ap_times_s


def make_spike_train(fluo_time: np.ndarray, ap_times_s: np.ndarray) -> Tuple[np.ndarray, float]:
    """Bin spike times to imaging time base."""
    dt = np.median(np.diff(fluo_time))
    if dt <= 0:
        raise ValueError("Non-increasing or invalid fluo_time.")

    edges = np.concatenate([
        [fluo_time[0] - 0.5 * dt],
        0.5 * (fluo_time[1:] + fluo_time[:-1]),
        [fluo_time[-1] + 0.5 * dt]
    ])

    spike_counts, _ = np.histogram(ap_times_s, bins=edges)
    spike_train = spike_counts.astype(float)

    return spike_train, float(dt)


# ---------------------------
# 2) OPTIMIZED Generative model
# ---------------------------

def bi_exp_kernel_torch(t: Tensor, tau_rise: Tensor, tau_decay: Tensor) -> Tensor:
    """
    Vectorized bi-exponential kernel for batched computation.
    t: [T] or scalar
    tau_rise, tau_decay: [N] batch dimension
    Returns: [N, T]
    """
    tau_rise = tau_rise.clamp(min=1e-6)
    tau_decay = tau_decay.clamp(min=tau_rise + 1e-6)
    
    # Reshape for broadcasting: tau [N, 1], t [1, T]
    if t.dim() == 0:
        t = t.unsqueeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0)  # [1, T]
    
    tau_rise = tau_rise.unsqueeze(-1)  # [N, 1]
    tau_decay = tau_decay.unsqueeze(-1)  # [N, 1]
    
    h = torch.exp(-t / tau_decay) - torch.exp(-t / tau_rise)  # [N, T]
    
    # Normalize each kernel
    peak = h.max(dim=-1, keepdim=True)[0].clamp(min=1e-12)
    h = h / peak
    
    return h


def hill_nl_torch(c: Tensor, kd: Tensor, n: Tensor) -> Tensor:
    """
    Vectorized Hill saturation.
    c: [N, T]
    kd, n: [N] or [N, 1]
    Returns: [N, T]
    """
    c = c.clamp(min=0.0)
    n = n.clamp(min=1e-3)
    kd = kd.clamp(min=1e-9)
    
    if kd.dim() == 1:
        kd = kd.unsqueeze(-1)
    if n.dim() == 1:
        n = n.unsqueeze(-1)
    
    c_n = torch.pow(c, n)
    kd_n = torch.pow(kd, n)
    
    return c_n / (c_n + kd_n)


class BatchedSimulator:
    """Optimized batched simulator with kernel caching."""
    
    def __init__(self, spike_train: np.ndarray, dt: float, device: str = "cpu", 
                 nonlinearity: str = "hill", max_tau_decay: float = 2.0):
        self.spike_train = torch.tensor(spike_train, dtype=torch.float32, device=device)
        self.dt = dt
        self.device = device
        self.nonlinearity = nonlinearity
        self.T = len(spike_train)
        
        # Pre-compute kernel time base (long enough for largest tau_decay)
        self.kT = int(np.ceil(5.0 * max_tau_decay / dt))
        self.tker = torch.arange(self.kT, dtype=torch.float32, device=device) * dt
        
    def __call__(self, theta: Tensor) -> Tensor:
        """
        Batched simulation.
        theta: [N, 7] where columns are [tau_rise, tau_decay, amp, kd, n, f0, sigma]
        Returns: [N, T] fluorescence traces
        """
        N = theta.shape[0]
        
        # Extract parameters
        tau_rise = theta[:, 0]    # [N]
        tau_decay = theta[:, 1]   # [N]
        amp = theta[:, 2]         # [N]
        kd = theta[:, 3]          # [N]
        n = theta[:, 4]           # [N]
        f0 = theta[:, 5]          # [N]
        sigma = theta[:, 6]       # [N]
        
        # Build kernels for all parameter sets at once [N, kT]
        h = bi_exp_kernel_torch(self.tker, tau_rise, tau_decay)  # [N, kT]
        
        # Convolve spike train with each kernel using FFT (fastest for long signals)
        # spike_train is [T], h is [N, kT]
        # Pad to avoid circular convolution artifacts
        conv_len = self.T + self.kT - 1
        fft_len = 2 ** int(np.ceil(np.log2(conv_len)))  # Next power of 2 for efficiency
        
        # FFT of spike train [fft_len]
        spike_fft = torch.fft.rfft(self.spike_train, n=fft_len)
        
        # FFT of kernels [N, fft_len//2 + 1]
        h_padded = torch.nn.functional.pad(h, (0, fft_len - self.kT))
        h_fft = torch.fft.rfft(h_padded, dim=1)
        
        # Multiply in frequency domain and transform back [N, fft_len]
        c_fft = h_fft * spike_fft.unsqueeze(0)
        c_full = torch.fft.irfft(c_fft, n=fft_len)
        
        # Extract valid portion [N, T]
        c = c_full[:, :self.T]
        
        # Apply nonlinearity
        if self.nonlinearity == "hill":
            g = hill_nl_torch(c, kd, n)  # [N, T]
        else:
            g = c
        
        # Fluorescence
        amp = amp.unsqueeze(-1)  # [N, 1]
        f0 = f0.unsqueeze(-1)    # [N, 1]
        F = f0 + amp * g         # [N, T]
        
        # Add noise
        if (sigma > 0).any():
            sigma = sigma.unsqueeze(-1)  # [N, 1]
            noise = torch.randn_like(F) * sigma
            F = F + noise
        
        return F


# ---------------------------
# 3) OPTIMIZED SBI workflow
# ---------------------------

def build_sbi_prior_focused(device: str = "cpu", tau_rise_range=(0.005, 0.008),
                           tau_decay_range=(0.2, 1.0), amp_range=(5.0, 30.0)):
    """Build tighter priors if you have good starting points."""
    low = torch.tensor([
        tau_rise_range[0],   # tau_rise
        tau_decay_range[0],  # tau_decay
        amp_range[0],        # amp
        5.0,                 # kd
        1.0,                 # n
        -0.1,                # f0
        0.03,                # sigma
    ], device=device)

    high = torch.tensor([
        tau_rise_range[1],   # tau_rise
        tau_decay_range[1],  # tau_decay
        amp_range[1],        # amp
        20.0,                # kd
        2.0,                 # n
        0.1,                 # f0
        0.3,                 # sigma
    ], device=device)

    return BoxUniform(low=low, high=high)


def run_sbi_fit_optimized(
    mat_path: str,
    recording_id: int = 0,
    nonlinearity: str = "hill",
    num_simulations: int = 2000,  # Reduced from 4000
    device: str = "cpu",
    seed: int = 123,
    use_rejection_sampling: bool = False,  # Faster than MCMC
    batch_size: int = 500,  # Simulate in batches
):
    """Optimized SBI fitting with batched simulation."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Load data
    fluo_time, fluo_mean, ap_times_s = load_ground_truth_mat(mat_path, recording_id=recording_id)
    spike_train, dt = make_spike_train(fluo_time, ap_times_s)
    y_obs = fluo_mean.copy()

    # Quick visualization
    plt.figure(figsize=(10, 4))
    plt.plot(fluo_time, y_obs, 'k', lw=1)
    st = np.where(spike_train > 0)[0]
    for k in st[::max(1, len(st)//200)]:
        plt.axvline(fluo_time[k], color='r', alpha=0.2, lw=0.5)
    plt.title("Observed dF/F with spike markers")
    plt.xlabel("Time (s)")
    plt.ylabel("dF/F")
    plt.tight_layout()
    plt.show()

    # Build prior and batched simulator
    prior = build_sbi_prior_focused(device=device)
    simulator = BatchedSimulator(spike_train, dt, device=device, nonlinearity=nonlinearity)
    y_obs_t = torch.tensor(y_obs, dtype=torch.float32, device=device)

    # Generate training data in batches
    print(f"Running {num_simulations} simulations in batches of {batch_size}...")
    all_thetas = []
    all_xs = []
    
    num_batches = (num_simulations + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch_n = min(batch_size, num_simulations - i * batch_size)
        thetas_batch = prior.sample((batch_n,)).to(device)
        xs_batch = simulator(thetas_batch)
        all_thetas.append(thetas_batch)
        all_xs.append(xs_batch)
        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/{num_batches} complete")
    
    thetas = torch.cat(all_thetas, dim=0)
    xs = torch.cat(all_xs, dim=0)
    
    print("Training posterior...")
    inference = SNPE(prior=prior, device=device)
    density_estimator = inference.append_simulations(thetas, xs).train()
    
    # Use rejection sampling (faster) or MCMC
    if use_rejection_sampling:
        posterior = inference.build_posterior(density_estimator, sample_with="rejection")
        num_samples = 2000
        print(f"Sampling {num_samples} from posterior with rejection sampling...")
        thetas_post = posterior.sample((num_samples,), x=y_obs_t)
    else:
        posterior = inference.build_posterior(density_estimator, sample_with="mcmc")
        print("Sampling from posterior with MCMC (slower)...")
        thetas_post = posterior.sample((2000,), x=y_obs_t, 
                                      mcmc_parameters={"num_chains": 2, "warmup_steps": 200, "thin": 3})
    
    post_mean = thetas_post.mean(0)

    # Posterior predictive check
    y_fit = simulator(post_mean.unsqueeze(0)).squeeze(0).cpu().numpy()

    # Metrics
    def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    def corr(a, b): 
        a = a - np.mean(a)
        b = b - np.mean(b)
        denom = np.sqrt(np.sum(a*a)*np.sum(b*b)) + 1e-12
        return float(np.sum(a*b) / denom)

    print("\nPosterior mean parameters (tau_rise, tau_decay, amp, kd, n, f0, sigma):")
    print(post_mean.detach().cpu().numpy())
    print(f"\nRMSE (fit vs obs): {rmse(y_fit, y_obs):.4f} dF/F")
    print(f"Pearson corr:     {corr(y_fit, y_obs):.4f}")

    # Plot fit vs observed
    plt.figure(figsize=(10, 4))
    plt.plot(fluo_time, y_obs, 'k', lw=1, label="Observed")
    plt.plot(fluo_time, y_fit, 'C1', lw=1.5, label="Posterior-mean sim")
    plt.xlabel("Time (s)")
    plt.ylabel("dF/F")
    plt.legend()
    plt.title("Posterior predictive check")
    plt.tight_layout()
    plt.show()

    results = {
        "posterior": posterior,
        "posterior_samples": thetas_post,
        "posterior_mean": post_mean,
        "y_fit": y_fit,
        "y_obs": y_obs,
        "fluo_time": fluo_time,
        "spike_train": spike_train,
        "dt": dt,
    }
    return results


# ---------------------------
# 5) Example run
# ---------------------------
if __name__ == "__main__":
    mat_file = "CAttached_jGCaMP8s_472182_2_mini.mat"
    
    # For GPU: device = "cuda" if torch.cuda.is_available() else "cpu"
    results = run_sbi_fit_optimized(
        mat_path=mat_file,
        recording_id=0,
        nonlinearity="hill",
        num_simulations=2000,  # Reduced from 4000
        device="cpu",
        seed=123,
        use_rejection_sampling=False,  # Much faster than MCMC
        batch_size=500,
    )