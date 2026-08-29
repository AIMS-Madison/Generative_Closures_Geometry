"""Shared runtime, sampling, plotting, and metric utilities."""

import os
import random

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset


################################
##### Data Preprosessing #######
################################
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def resolve_device(device=None):
    """Resolve ``None`` or ``'auto'`` to the best available PyTorch device."""
    if device is None or str(device).lower() == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resolved = torch.device(device)
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return resolved


################################
########### Sampling ###########
################################
def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n, time_min, time_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = time_min ** (1 / rho)
    max_inv_rho = time_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas)


################################
####### Energy Spectrum ########
################################
def moving_average(data, window_size):
    """ Simple moving average """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def energy_spectrum(phi, lx=1, ly=1, smooth=True):
    # Assuming phi is of shape (time_steps, nx, ny)
    nx, ny = phi.shape[1], phi.shape[2]
    nt = nx * ny

    phi_h = np.fft.fftn(phi, axes=(1, 2)) / nt  # Fourier transform along spatial dimensions

    energy_h = 0.5 * (phi_h * np.conj(phi_h)).real  # Spectral energy density

    k0x = 2.0 * np.pi / lx
    k0y = 2.0 * np.pi / ly
    knorm = (k0x + k0y) / 3.0

    kxmax = nx // 2
    kymax = ny // 2

    wave_numbers = knorm * np.arange(0, nx)

    energy_spectrum = np.zeros(len(wave_numbers))

    for kx in range(nx):
        rkx = kx if kx <= kxmax else kx - nx
        for ky in range(ny):
            rky = ky if ky <= kymax else ky - ny
            rk = np.sqrt(rkx ** 2 + rky ** 2)
            k = int(np.round(rk))
            if k < len(wave_numbers):
                energy_spectrum[k] += np.sum(energy_h[:, kx, ky])

    energy_spectrum /= knorm

    if smooth:
        smoothed_spectrum = moving_average(energy_spectrum, 5)  # Smooth the spectrum
        smoothed_spectrum = np.append(smoothed_spectrum, np.zeros(4))  # Append zeros to match original length after convolution
        smoothed_spectrum[:4] = np.sum(energy_h[:, :4, :4].real, axis=(0, 1, 2)) / (knorm * phi.shape[0])  # First 4 values corrected
        energy_spectrum = smoothed_spectrum

    knyquist = knorm * min(nx, ny) / 2

    return knyquist, wave_numbers, energy_spectrum

################################
########### Plotting ###########
################################
def plot_heatmaps(original, sample, num_samples=5):
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))

    for i in range(num_samples):
        # Plot original
        sns.heatmap(original[i], ax=axes[0, i], cmap='coolwarm', cbar=True, square=True)
        axes[0, i].set_title(f'Original {i + 1}')
        axes[0, i].axis('off')

        # Plot reconstructed
        sns.heatmap(sample[i], ax=axes[1, i], cmap='coolwarm', cbar=True, square=True)
        axes[1, i].set_title(f'Generated {i + 1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

class ErrorMetrics:
    def __init__(self):
        pass
    def mse(self, data1, data2):
        """Mean Squared Error (MSE)"""
        return torch.mean((data1 - data2) ** 2)

    def frobenius(self, data1, data2):
        """Normalized Frobenius Norm Error"""
        error_fro = torch.linalg.matrix_norm(data1 - data2, ord='fro', dim=(1, 2))
        norm_ref = torch.linalg.matrix_norm(data1, ord='fro', dim=(1, 2))
        eps = torch.finfo(norm_ref.dtype).eps
        return torch.mean(error_fro / norm_ref.clamp_min(eps))

class H5ClosureDataset(Dataset):
    """
    Loads paired vorticity (omega) and closure H from an H5 file.
    Expects datasets 'train_vorticity_64' and 'train_nonlinear_64'.
    """
    def __init__(self, filename, key_omega, key_H, max_samples=None, transform=None):
        super().__init__()
        with h5py.File(filename, 'r') as f:
            missing = [key for key in (key_omega, key_H) if key not in f]
            if missing:
                raise KeyError(f"Missing HDF5 dataset(s) {missing} in {filename}")
            v = f[key_omega]
            h = f[key_H]
            n = min(len(v), len(h))
            if max_samples is not None:
                n = min(n, max_samples)
            data_o = v[:n]
            data_h = h[:n]
        # store as (N,H,W) float tensors
        self.omega = torch.from_numpy(data_o).float()
        self.H     = torch.from_numpy(data_h).float()
        self.transform = transform

    def __len__(self):
        return len(self.omega)

    def __getitem__(self, idx):
        o = self.omega[idx]
        h = self.H[idx]
        if self.transform:
            o = self.transform(o)
            h = self.transform(h)

        return o, h

# Define tick positions and labels
def create_ticks_labels(size, step=20):
    ticks = np.arange(0, size, step * size / 64)
    tick_labels = [str(int(tick)) for tick in ticks]
    return ticks, tick_labels
