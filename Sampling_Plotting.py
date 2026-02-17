import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from functools import partial

from utils import *
from Models.Generative_Closures.Generative_Models import *
from Models.Generative_Closures.Interpolant import *

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import matplotlib.pyplot as plt
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

### Configure NumPy & PyTorch
np.set_printoptions(
    suppress=False,
    formatter={'float': lambda x: f"{x:.3e}"}
)
torch.set_printoptions(
    sci_mode=True,
    precision=3
)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')

# Load the data
file_name = 'c:\\UWMadisonResearch\\Comparisons\\Data\\test_diffusion_nonlinear.h5'

N = 100
num_ensembles = 1
with h5py.File(file_name, 'r') as file:
    test_nonlinear = torch.tensor(file['test_nonlinear_64'][:N], device=device)
    test_vorticity = torch.tensor(file['test_vorticity_64'][:N], device=device)

metrics = ErrorMetrics()


def sample_closure_field(
    method: str,
    model: torch.nn.Module,
    omega: torch.Tensor,
    num_ensembles: int = 1,
    sample_steps: int = 100,
    sigma_coef: float = 1.0,
    time_min: float = 1e-3,
    time_max: float = 1.0,
    time_schedule: str = 'uniform',   # 'uniform' or 'karras'
    device: str = 'cuda',
):
    """
    Unified sampler for diffusion and interpolant-based closures.

    Returns:
        samples: Tensor[B, num_ensembles, T, H, W] of all intermediate fields.
        total_distance: Tensor[B, num_ensembles] accumulated path length.
    """
    B, H, W = omega.shape
    omega = omega.to(device)
    model = model.to(device).eval()

    # Build time grids
    if time_schedule == 'uniform':
        ts = torch.linspace(time_max, time_min, sample_steps + 1, device=device)
        ts_forward = torch.linspace(time_min, time_max, sample_steps + 1, device=device)
    elif time_schedule == 'karras':
        ts = get_sigmas_karras(sample_steps, time_min, time_max, device=device)
        ts_forward = list(reversed(ts))
    else:
        raise ValueError(f"Unknown time_schedule: {time_schedule}")

    dt = ts[:-1] - ts[1:]
    dt_forward = torch.tensor(ts_forward[1:]) - torch.tensor(ts_forward[:-1])

    # Tile conditioning
    omega_rep = omega.unsqueeze(1).expand(-1, num_ensembles, -1, -1).reshape(-1, H, W)
    batch = B * num_ensembles

    # Initialize based on method
    if method == 'diffusion':
        from functools import partial
        marginal_std = partial(marginal_prob_std, sigma=sigma_coef, device_=device)
        diff_coeff = partial(diffusion_coeff, sigma=sigma_coef, device_=device)

        t0 = ts[0].repeat(batch)
        x = torch.randn_like(omega_rep) * marginal_std(t0)[:, None, None]

    elif method in ('interpolant_knownsde', 'interpolant_gaussiansde', 'interpolant_gaussianode'):
        if method == 'interpolant_knownsde':
            x = omega_rep.clone()
        else:
            x = torch.randn_like(omega_rep)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Prepare storage
    states = [x.clone()]
    total_distance = torch.zeros(batch, device=device)
    prev = x.clone()

    with torch.no_grad():
        for i in range(sample_steps):
            if method == 'diffusion':
                t_backward = ts[i]
                t_batch = t_backward.repeat(batch)
                g = diff_coeff(t_batch)
                grad = model(t_batch, x, omega_rep)
                dt_i = dt[i]
                mean = x + (g**2)[:, None, None] * grad * dt_i
                if i < sample_steps - 1:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(dt_i) * g[:, None, None] * noise
                else:
                    x = mean

            else:
                t_forward = ts_forward[i]
                t_batch = t_forward.repeat(batch)
                dt_i = dt_forward[i]
                drift = model(t_batch, x, omega_rep)
                x = x + drift * dt_i

            # Track distance and states
            total_distance += (x - prev).flatten(1).norm(dim=1)
            prev.copy_(x)
            states.append(x.clone())

    # Stack and reshape
    traj = torch.stack(states, dim=1)                   # [batch, T+1, H, W]
    traj = traj.view(B, num_ensembles, sample_steps + 1, H, W)
    total_distance = total_distance.view(B, num_ensembles)

    return traj, total_distance


set_seed(42)
sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)

modes = 6
width = 40
pcdm = FNO2d_Diffusion(marginal_prob_std_fn, modes, modes, width, padding = 0, embed_dim = 512, length = 1).to(device)

model_name = 'c:\\UWMadisonResearch\\Comparisons\\Models\\Score_Diffusion\\P-CDM.pth'
pcdm.load_state_dict(torch.load(model_name))

# Sample closure fields using the P-CDM model
pcdm_uniform_samples, pcdm_uniform_distance = sample_closure_field(
    method='diffusion',
    model=pcdm,
    omega=test_vorticity,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=sigma,
    time_min=0,
    time_max=1.0,
    time_schedule='uniform',
    device=device
)

pcdm_karras_samples, pcdm_karras_distance = sample_closure_field(
    method='diffusion',
    model=pcdm,
    omega=test_vorticity,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=sigma,
    time_min=0.001,
    time_max=0.4,
    time_schedule='karras',
    device=device
)


# SI with Gaussian SDE
modes = 6
width = 40
SISDE_Gussian = FNO2d_Orig(modes, modes, width, padding = 0, embed_dim = 256, length = 1).to(device)

model_name = 'c:\\UWMadisonResearch\\Comparisons\\Models\\Stochastic_Interpolant\\Stochastic_Interpolant_gaussianbase_sde.pth'
SISDE_Gussian.load_state_dict(torch.load(model_name))

psi_gaussian_uniform_samples, psi_gaussian_uniform_distance = sample_closure_field(
    method='interpolant_gaussiansde',
    model=SISDE_Gussian,
    omega=test_vorticity,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0,
    time_max=1,
    time_schedule='uniform',
    device=device
)

# SI with Known SDE
SISDE_Known = FNO2d_Orig(modes, modes, width, padding = 0, embed_dim = 256, length = 1).to(device)

model_name = 'c:\\UWMadisonResearch\\Comparisons\\Models\\Stochastic_Interpolant\\Stochastic_Interpolant_knownbase.pth'
SISDE_Known.load_state_dict(torch.load(model_name))

psi_known_uniform_samples, psi_known_uniform_distance = sample_closure_field(
    method='interpolant_knownsde',
    model=SISDE_Known,
    omega=test_vorticity,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0,
    time_max=1,
    time_schedule='uniform',
    device=device
)

# SI with Gaussian ODE
SIODE_Gaussian = FNO2d_Orig(modes, modes, width, padding = 0, embed_dim = 256, length = 1).to(device)

model_name = 'c:\\UWMadisonResearch\\Comparisons\\Models\\Flow_Matching\\FM.pth'
SIODE_Gaussian.load_state_dict(torch.load(model_name))

psi_gaussian_ode_samples, psi_gaussian_ode_distance = sample_closure_field(
    method='interpolant_gaussianode',
    model=SIODE_Gaussian,
    omega=test_vorticity,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0,
    time_max=1,
    time_schedule='uniform',
    device=device
)





### Plot and save
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

set_seed(13)

starting_field = psi_gaussian_ode_samples[:, 0, 0].cpu()
gt_field = test_nonlinear[:, :, :].cpu()
generated_field = psi_gaussian_ode_samples[:, 0, -1].cpu()
err_field = np.abs(gt_field - generated_field)

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(4, 5, figsize=(20, 16), constrained_layout=True)
fs = 28
plt.rcParams.update({'font.size': fs})


ticks_1, tick_labels_1 = create_ticks_labels(gt_field.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(generated_field.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(err_field.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, gt_field.shape[0], (1,)).item() for _ in range(5)]

# Define color scale parameters
max_val = 0.7
min_val = -0.8
err_max = 0.15
err_min = 0
cbar_ticks = np.linspace(min_val, max_val, 6)
cbar_ticks_err = np.linspace(err_min, err_max, 6)
cbar_ticks_contour = np.linspace(err_min, err_max, 6)

# Plot heatmaps and contour plots
for i, idx in enumerate(indices):
    j = i % 5  # Column index

    # --- Row 1: Starting Heatmap ---
    starting = starting_field[idx, ...].cpu().numpy()
    cbar_ticks_starting = np.linspace(-3.5, 3.5, 6)
    sns.heatmap(
        starting,
        ax=axs[0, j],
        cmap='rocket',
        cbar=(j == 4),  # Show colorbar only on the last column
        vmax=3.5,
        vmin=-3.5,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks_starting}
    )
    axs[0, j].set_title(r"\text{Initial }" + str(j + 1))
    axs[0, j].set_xticks(ticks_1)
    axs[0, j].set_yticks(ticks_1)
    axs[0, j].set_xticklabels(tick_labels_1, rotation=0)
    axs[0, j].set_yticklabels(tick_labels_1, rotation=0)
    axs[0, j].invert_yaxis()

    # --- Row 2: Generated Heatmap ---
    generated = generated_field[idx, ...].cpu().numpy()
    sns.heatmap(
        generated,
        ax=axs[1, j],
        cmap='rocket',
        cbar=(j == 4),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )

    axs[1, j].set_title(r"\text{Generated }" + str(j + 1))
    axs[1, j].set_xticks(ticks_2)
    axs[1, j].set_yticks(ticks_2)
    axs[1, j].set_xticklabels(tick_labels_2, rotation=0)
    axs[1, j].set_yticklabels(tick_labels_2, rotation=0)
    axs[1, j].invert_yaxis()

    # --- Row 3: Truth Heatmap ---
    truth = gt_field[idx, ...].cpu().numpy()
    sns.heatmap(
        truth,
        ax=axs[2, j],
        cmap='rocket',
        cbar=(j == 4),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )

    axs[2, j].set_title(r"\text{Truth }" + str(j + 1))
    axs[2, j].set_xticks(ticks_2)
    axs[2, j].set_yticks(ticks_2)
    axs[2, j].set_xticklabels(tick_labels_2, rotation=0)
    axs[2, j].set_yticklabels(tick_labels_2, rotation=0)
    axs[2, j].invert_yaxis()


    # --- Row 4: Error Heatmap ---
    error = err_field[idx, ...].cpu().numpy()
    ax_contour = axs[3, j]
    # Define the grid coordinates
    S = error.shape[0]
    x = np.arange(S)
    y = np.arange(S)
    X, Y = np.meshgrid(x, y)

    # Create filled contour plot using matplotlib
    contour = ax_contour.contourf(
        X, Y, error,
        levels=cbar_ticks_contour,  # Six levels to match cbar_ticks_err
        cmap='rocket',
        vmin=err_min,
        vmax=err_max
    )

    # Add colorbar only on the last column
    if j == 4:
        cbar_contour = fig.colorbar(
            contour,
            ax=ax_contour,
            format='%.2f'
        )

    ax_contour.set_title(r"\text{Error Contour }" + str(j + 1))
    ax_contour.set_xticks(ticks_3)
    ax_contour.set_yticks(ticks_3)
    ax_contour.set_xticklabels(tick_labels_3, rotation=0)
    ax_contour.set_yticklabels(tick_labels_3, rotation=0)

# Adjust tick parameters for all axes
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=fs)


# Adjust layout and save the plot
plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
# plt.show()
plt.savefig(
    'Plots/FM_Samples.png',
    dpi=300,
    bbox_inches='tight'
)


















uniform_steps_backward = torch.linspace(1, 0, 11, device=device)[::3]  # 11 steps from 1 to 0
uniform_steps_forward = torch.linspace(0, 1, 11, device=device)[::3]
karras_steps_backward = get_sigmas_karras(10, 0.0001, 0.4, device=device)[::3]

# List your methods and their titles
methods = [
    ('P-DM (Uniform)', pcdm_uniform_samples, uniform_steps_backward),
    ('P-DM (Adaptive)', pcdm_karras_samples, karras_steps_backward),
    ('P-FM', psi_gaussian_ode_samples, uniform_steps_forward),
    ('P-SI (Gaussian)', psi_gaussian_uniform_samples, uniform_steps_forward),
    ('P-SI (Empirical)', psi_known_uniform_samples, uniform_steps_forward),
]

# Ground‐truth field
truth = test_nonlinear[0].cpu().numpy()

# User-adjustable parameters
error_vmin, error_vmax = 0.0, 0.8
title_fontsize = 32
tick_fontsize = 22
label_fontsize = 32

n_methods = len(methods)

# Create figure
fig = plt.figure(figsize=(28, 4 * n_methods))

# Manual positioning parameters
img_width = 0.11929  # Width of each image
img_height = 0.156  # Height of each image
cbar_width = 0.005  # Width of colorbar
gap_x = 0.02  # Horizontal gap between images
gap_y = 0.03  # Vertical gap between rows
left_margin = 0.05
x_pos_prev = left_margin
bottom_margin = 0.05

for i, (label, samples, steps) in enumerate(methods):
    # Convert tensor to numpy if needed
    traj = samples[0, 0] if isinstance(samples, np.ndarray) else samples[0, 0].cpu().numpy()

    # Calculate vertical position for this row
    x_pos_prev = left_margin
    y_pos = bottom_margin + (n_methods - 1 - i) * (img_height + gap_y)

    # Plot intermediate states
    for t in range(6):
        # Calculate horizontal position
        if t ==1:
            gap_x = 0.001  # Adjust gap for first column
        else:
            gap_x = 0.001
        if t == 0:
            x_pos = x_pos_prev
        else:
            x_pos = x_pos_prev + img_width + gap_x
            x_pos_prev = x_pos

        # Create axis with exact position
        ax = fig.add_axes([x_pos, y_pos, img_width, img_height])
        data = traj[::2][t]

        if t == 0:
            # First column
            print(label)
            ax.set_ylabel(label, fontsize=label_fontsize, rotation=90, labelpad=20, va='center')
            vmin0, vmax0 = data.min(), data.max()
            vmin0_round = np.round(vmin0 * 2) / 2  # round to nearest .0 or .5
            vmax0_round = np.round(vmax0 * 2) / 2
            im = ax.imshow(data, cmap='rocket', vmin=vmin0_round, vmax=vmax0_round, aspect='equal')
            ticks0 = np.linspace(vmin0_round, vmax0_round, 5)

            # # Add colorbar manually positioned
            # cax = fig.add_axes([x_pos + img_width + 0.002, y_pos, cbar_width, img_height])
            # cb = fig.colorbar(im, cax=cax, ticks=ticks0)
            # cb.ax.set_yticklabels([f"{tick:.2f}" for tick in ticks0], fontsize=tick_fontsize)

            ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            # Middle columns
            im = ax.imshow(data, cmap='rocket', vmin=data.min(), vmax=data.max(), aspect='equal')
            ax.axis('off')

        # Title with scientific τ
        tau = steps[t] if isinstance(steps, np.ndarray) else steps[t].item()
        ax.set_title(r'$\tau={:.4f}$'.format(tau), fontsize=title_fontsize, pad=10)

    # # Error plot in last column
    # # x_pos_err = left_margin + 6 * (img_width + gap_x)
    # x_pos_err = x_pos_prev + img_width + gap_x
    # ax_err = fig.add_axes([x_pos_err, y_pos, img_width, img_height])

    # error_ticks = np.linspace(error_vmin, error_vmax, 5)
    # err = abs(traj[-1] - (truth if isinstance(truth, np.ndarray) else truth.cpu().numpy()))

    # # Use imshow instead of contourf for consistent sizing
    # im_err = ax_err.imshow(err, cmap='rocket', vmin=error_vmin, vmax=error_vmax, aspect='equal')

    # # Optional: add contour lines on top
    # cs = ax_err.contour(err, levels=error_ticks[1:-1], colors='white', linewidths=0.5, alpha=0.5)

    # ax_err.axis('off')
    # ax_err.set_title('Error', fontsize=title_fontsize, pad=10)

    # # Error colorbar manually positioned
    # cax_err = fig.add_axes([x_pos_err + img_width + 0.002, y_pos, cbar_width, img_height])
    # cb_err = fig.colorbar(im_err, cax=cax_err, ticks=error_ticks)
    # cb_err.ax.set_yticklabels([f"{t:.2f}" for t in error_ticks], fontsize=tick_fontsize)

plt.savefig('c:\\UWMadisonResearch\\Comparisons\\Plots\\closure_field_sampling.png', dpi=300, bbox_inches='tight')












# resolution invariance sampling and plotting
file_name = 'Data/test_diffusion_nonlinear_resolution_invariant.h5'

with h5py.File(file_name, 'r') as file:
    test_nonlinear_64 = torch.tensor(file['test_nonlinear_64'], device=device).permute(3, 0, 1, 2).reshape(-1, 64, 64)[:N]
    test_vorticity_64 = torch.tensor(file['test_vorticity_64'], device=device).permute(3, 0, 1, 2).reshape(-1, 64, 64)[:N]
    test_nonlinear_128 = torch.tensor(file['test_nonlinear_128'], device=device).permute(3, 0, 1, 2).reshape(-1, 128, 128)[:N]
    test_vorticity_128 = torch.tensor(file['test_vorticity_128'], device=device).permute(3, 0, 1, 2).reshape(-1, 128, 128)[:N]
    test_vorticity_256 = torch.tensor(file['test_vorticity_256'], device=device).permute(3, 0, 1, 2).reshape(-1, 256, 256)[:N]
    test_nonlinear_256 = torch.tensor(file['test_nonlinear_256'], device=device).permute(3, 0, 1, 2).reshape(-1, 256, 256)[:N]


set_seed(42)
sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)

modes = 6
width = 40
pcdm = FNO2d_Diffusion(marginal_prob_std_fn, modes, modes, width, padding = 0, embed_dim = 512, length = 1).to(device)

model_name = 'Models/Score_Diffusion/P-CDM.pth'
pcdm.load_state_dict(torch.load(model_name))

pcdm_64, _ = sample_closure_field(
    method='diffusion',
    model=pcdm,
    omega=test_vorticity_64,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=sigma,
    time_min=0.001,
    time_max=0.4,
    time_schedule='karras',
    device=device
)
pcdm_64 = pcdm_64[:, 0, -1, :, :]

pcdm_128, _ = sample_closure_field(
    method='diffusion',
    model=pcdm,
    omega=test_vorticity_128,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=sigma,
    time_min=0.001,
    time_max=0.4,
    time_schedule='karras',
    device=device
)
pcdm_128 = pcdm_128[:, 0, -1, :, :]

pcdm_256, _ = sample_closure_field(
    method='diffusion',
    model=pcdm,
    omega=test_vorticity_256,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=sigma,
    time_min=0.001,
    time_max=0.4,
    time_schedule='karras',
    device=device
)
pcdm_256 = pcdm_256[:, 0, -1, :, :]

# metrics print
mse_64 = metrics.mse(test_nonlinear_64, pcdm_64)
mse_128 = metrics.mse(test_nonlinear_128, pcdm_128)
mse_256 = metrics.mse(test_nonlinear_256, pcdm_256)
print(f"P-CDM MSE at 64x64: {mse_64.item():.3e}")
print(f"P-CDM MSE at 128x128: {mse_128.item():.3e}")
print(f"P-CDM MSE at 256x256: {mse_256.item():.3e}")
re_64 = metrics.frobenius(test_nonlinear_64, pcdm_64)
re_128 = metrics.frobenius(test_nonlinear_128, pcdm_128)
re_256 = metrics.frobenius(test_nonlinear_256, pcdm_256)
print(f"P-CDM RE at 64x64: {re_64.item():.3e}")
print(f"P-CDM RE at 128x128: {re_128.item():.3e}")
print(f"P-CDM RE at 256x256: {re_256.item():.3e}")

# SI with Gaussian SDE
modes = 6
width = 40
SISDE_Gussian = FNO2d_Orig(modes, modes, width, padding = 0, embed_dim = 256, length = 1).to(device)

model_name = 'Models/Stochastic_Interpolant/Stochastic_Interpolant_gaussianbase_sde.pth'
SISDE_Gussian.load_state_dict(torch.load(model_name))

psi_gaussian_64, _ = sample_closure_field(
    method='interpolant_gaussiansde',
    model=SISDE_Gussian,
    omega=test_vorticity_64,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0.001,
    time_max=1,
    time_schedule='uniform',
    device=device
)
psi_gaussian_64 = psi_gaussian_64[:, 0, -1, :, :]

psi_gaussian_128, _ = sample_closure_field(
    method='interpolant_gaussiansde',
    model=SISDE_Gussian,
    omega=test_vorticity_128,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0.001,
    time_max=1,
    time_schedule='uniform',
    device=device
)
psi_gaussian_128 = psi_gaussian_128[:, 0, -1, :, :]

psi_gaussian_256, _ = sample_closure_field(
    method='interpolant_gaussiansde',
    model=SISDE_Gussian,
    omega=test_vorticity_256,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0.001,
    time_max=1,
    time_schedule='uniform',
    device=device
)
psi_gaussian_256 = psi_gaussian_256[:, 0, -1, :, :]

# metrics print
mse_64 = metrics.mse(test_nonlinear_64, psi_gaussian_64)
mse_128 = metrics.mse(test_nonlinear_128, psi_gaussian_128)
mse_256 = metrics.mse(test_nonlinear_256, psi_gaussian_256)
re_64 = metrics.frobenius(test_nonlinear_64, psi_gaussian_64)
re_128 = metrics.frobenius(test_nonlinear_128, psi_gaussian_128)
re_256 = metrics.frobenius(test_nonlinear_256, psi_gaussian_256)
print(f"SI-Gaussian MSE at 64x64: {mse_64.item():.3e}")
print(f"SI-Gaussian MSE at 128x128: {mse_128.item():.3e}")
print(f"SI-Gaussian MSE at 256x256: {mse_256.item():.3e}")
print(f"SI-Gaussian RE at 64x64: {re_64.item():.3e}")
print(f"SI-Gaussian RE at 128x128: {re_128.item():.3e}")
print(f"SI-Gaussian RE at 256x256: {re_256.item():.3e}")


# SI with Known SDE
SISDE_Known = FNO2d_Orig(modes, modes, width, padding = 0, embed_dim = 256, length = 1).to(device)

model_name = 'Models/Stochastic_Interpolant/Stochastic_Interpolant_knownbase.pth'
SISDE_Known.load_state_dict(torch.load(model_name))

psi_known_64, _ = sample_closure_field(
    method='interpolant_knownsde',
    model=SISDE_Known,
    omega=test_vorticity_64,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0.001,
    time_max=1,
    time_schedule='uniform',
    device=device
)
psi_known_64 = psi_known_64[:, 0, -1, :, :]

psi_known_128, _ = sample_closure_field(
    method='interpolant_knownsde',
    model=SISDE_Known,
    omega=test_vorticity_128,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0.001,
    time_max=1,
    time_schedule='uniform',
    device=device
)
psi_known_128 = psi_known_128[:, 0, -1, :, :]

psi_known_256, _ = sample_closure_field(
    method='interpolant_knownsde',
    model=SISDE_Known,
    omega=test_vorticity_256,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0.001,
    time_max=1,
    time_schedule='uniform',
    device=device
)
psi_known_256 = psi_known_256[:, 0, -1, :, :]

# metrics print
mse_64 = metrics.mse(test_nonlinear_64, psi_known_64)
mse_128 = metrics.mse(test_nonlinear_128, psi_known_128)
mse_256 = metrics.mse(test_nonlinear_256, psi_known_256)
re_64 = metrics.frobenius(test_nonlinear_64, psi_known_64)
re_128 = metrics.frobenius(test_nonlinear_128, psi_known_128)
re_256 = metrics.frobenius(test_nonlinear_256, psi_known_256)
print(f"SI-Known MSE at 64x64: {mse_64.item():.3e}")
print(f"SI-Known MSE at 128x128: {mse_128.item():.3e}")
print(f"SI-Known MSE at 256x256: {mse_256.item():.3e}")
print(f"SI-Known RE at 64x64: {re_64.item():.3e}")
print(f"SI-Known RE at 128x128: {re_128.item():.3e}")
print(f"SI-Known RE at 256x256: {re_256.item():.3e}")


# SI with Gaussian ODE
SIODE_Gaussian = FNO2d_Orig(modes, modes, width, padding = 0, embed_dim = 256, length = 1).to(device)

model_name = 'Models/Stochastic_Interpolant/Stochastic_Interpolant_gaussianbase_ode.pth'
SIODE_Gaussian.load_state_dict(torch.load(model_name))

psi_gaussian_ode_64, _ = sample_closure_field(
    method='interpolant_gaussianode',
    model=SIODE_Gaussian,
    omega=test_vorticity_64,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0.001,
    time_max=1,
    time_schedule='uniform',
    device=device
)

psi_gaussian_ode_64 = psi_gaussian_ode_64[:, 0, -1, :, :]

psi_gaussian_ode_128, _ = sample_closure_field(
    method='interpolant_gaussianode',
    model=SIODE_Gaussian,
    omega=test_vorticity_128,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0.001,
    time_max=1,
    time_schedule='uniform',
    device=device
)
psi_gaussian_ode_128 = psi_gaussian_ode_128[:, 0, -1, :, :]

psi_gaussian_ode_256, _ = sample_closure_field(
    method='interpolant_gaussianode',
    model=SIODE_Gaussian,
    omega=test_vorticity_256,
    num_ensembles=num_ensembles,
    sample_steps=10,
    sigma_coef=1.0,
    time_min=0.001,
    time_max=1,
    time_schedule='uniform',
    device=device
)
psi_gaussian_ode_256 = psi_gaussian_ode_256[:, 0, -1, :, :]

# metrics print
mse_64 = metrics.mse(test_nonlinear_64, psi_gaussian_ode_64)
mse_128 = metrics.mse(test_nonlinear_128, psi_gaussian_ode_128)
mse_256 = metrics.mse(test_nonlinear_256, psi_gaussian_ode_256)
re_64 = metrics.frobenius(test_nonlinear_64, psi_gaussian_ode_64)
re_128 = metrics.frobenius(test_nonlinear_128, psi_gaussian_ode_128)
re_256 = metrics.frobenius(test_nonlinear_256, psi_gaussian_ode_256)
print(f"SI-Known MSE at 64x64: {mse_64.item():.3e}")
print(f"SI-Known MSE at 128x128: {mse_128.item():.3e}")
print(f"SI-Known MSE at 256x256: {mse_256.item():.3e}")
print(f"SI-Known RE at 64x64: {re_64.item():.3e}")
print(f"SI-Known RE at 128x128: {re_128.item():.3e}")
print(f"SI-Known RE at 256x256: {re_256.item():.3e}")


generated_64_fields = [pcdm_64, psi_gaussian_64, psi_known_64, psi_gaussian_ode_64]
generated_128_fields = [pcdm_128, psi_gaussian_128, psi_known_128, psi_gaussian_ode_128]
generated_256_fields = [pcdm_256, psi_gaussian_256, psi_known_256, psi_gaussian_ode_256]
gts = [test_nonlinear_64, test_nonlinear_128, test_nonlinear_256]

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

    return {'k': wave_numbers, 'E': energy_spectrum}
# Compute energy spectra for each resolution
res_fields = [(generated_64_fields, gts[0]),
              (generated_128_fields, gts[1]),
              (generated_256_fields, gts[2])]

fig, axs = plt.subplots(1, 3, figsize=(27, 8), sharey=True)

# === 设置 ===
fs = 32  # 全局字号
model_labels = ['P-CDM', 'P-SI (Gaussian)', 'P-SI (Empirical)', 'P-FM']
linestyles = ['-', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]
res_labels = [r'$64 \times 64$', r'$128 \times 128$', r'$256 \times 256$']
ymin, ymax = 1e-21, 1e0  # y轴范围

# === 收集 legend 句柄 ===
legend_handles = []
legend_labels = []

for i, (models, gt) in enumerate(res_fields):
    ax = axs[i]
    gt_spec = energy_spectrum(gt.cpu())
    k = gt_spec['k']
    E_gt = gt_spec['E']
    line_gt, = ax.plot(k, E_gt, label='Ground Truth', linewidth=3, linestyle='--')

    if i == 0:
        legend_handles.append(line_gt)
        legend_labels.append('Ground Truth')

    for model_idx, model_field in enumerate(models):
        spec = energy_spectrum(model_field.cpu())
        E = spec['E']
        line_model, = ax.plot(spec['k'], E, label=model_labels[model_idx], linewidth=3, linestyle=linestyles[model_idx])
        if i == 0:
            legend_handles.append(line_model)
            legend_labels.append(model_labels[model_idx])

    ax.set_title(f'Energy Spectrum of $H$: {res_labels[i]}', fontsize=fs)
    ax.set_xlabel('Wavenumber $k$', fontsize=fs)
    if i == 0:
        ax.set_ylabel('Energy $E(k)$', fontsize=fs)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(ymin, ymax)
    ax.grid(False)

    # ✅ 加粗坐标轴 ticks
    ax.tick_params(labelsize=fs, width=1, length=8)

    # ✅ 添加黑色边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

# ✅ 全局 legend，添加黑框
legend = fig.legend(
    legend_handles,
    legend_labels,
    loc='upper center',
    ncol=5,
    fontsize=fs,
    frameon=True,
    edgecolor='black'
)

legend.get_frame().set_boxstyle('Square')
legend.get_frame().set_linewidth(1)

plt.tight_layout(rect=[0, 0, 1, 0.8])  # 为 legend 留出空间
plt.savefig(
    'Plots/energy_resolution.png',
    dpi=300,
    bbox_inches='tight'
)