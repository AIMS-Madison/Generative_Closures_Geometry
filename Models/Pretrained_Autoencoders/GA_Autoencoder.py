import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import h5py
import numpy as np
import torch
import torch.nn as nn
import phate
from torch.utils.data import DataLoader, TensorDataset
import warnings
from scipy.sparse import SparseEfficiencyWarning
from Models.Pretrained_Autoencoders.AE import VariationalAutoEncoder
from project_paths import project_path, resolve_output_path
from utils import ErrorMetrics

# Suppress sparse warnings
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

# ------------------------------
# 1) Precompute manifold distances via PHATE
# ------------------------------
def compute_and_store_phate_diff_potential_distances(h5_path, knn, t_diffusion, out_path, block_size=100):
    """
    Compute diffusion potential-based distances from PHATE and save to disk as a memory-mapped .npy.
    """
    # Load full data
    with h5py.File(h5_path, 'r') as f:
        X = f['train_vorticity_64'][:]   # shape (N, 64, 64)
    N = X.shape[0]
    X_flat = X.reshape(N, -1)          # shape (N, 4096)

    # Compute PHATE (Diffusion Potential only)
    phate_op = phate.PHATE(knn=knn, t=t_diffusion, verbose=False)
    _ = phate_op.fit(X_flat)
    D_pot = phate_op.diff_potential     # shape (N, d), d typically = N or slightly smaller

    # Prepare memory-mapped distance matrix
    D = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float32, shape=(N, N))

    # Compute blockwise pairwise distances in diffusion potential space
    for i in range(0, N, block_size):
        print('Computing diffusion potential-based distances for block {}'.format(i))
        i2 = min(N, i + block_size)
        for j in range(i, N, block_size):
            j2 = min(N, j + block_size)
            diff = D_pot[i:i2, None, :] - D_pot[None, j:j2, :]
            dist_block = np.linalg.norm(diff, axis=2).astype(np.float32)
            D[i:i2, j:j2] = dist_block
            if i != j:
                D[j:j2, i:i2] = dist_block.T  # fill symmetric block

    del D
    print(f"Saved PHATE diffusion potential distances ({N}×{N}) to {out_path}")
    
# ------------------------------
# 2) Geometry-aware distance matching loss
# ------------------------------
def geometry_loss(z, idxs, D, zeta):
    """
    z: tensor (B, ...) latent activations
    idxs: tensor (B,) original dataset indices
    D: numpy memmap (N×N) of manifold distances
    zeta: float decay
    """
    B = z.size(0)
    # flatten latent codes
    z_flat = z.view(B, -1)
    # pairwise latent distances
    d_z = torch.cdist(z_flat, z_flat, p=2)

    # batch manifold distances
    idxs_np = idxs.cpu().numpy().clip(0, D.shape[0] - 1)
    d_x = torch.from_numpy(D[idxs_np[:, None], idxs_np]).to(z.device)

    # mask upper triangle
    mask = torch.triu(torch.ones(B, B, dtype=torch.bool, device=z.device), diagonal=1)

    # weighted squared error
    se = (d_z - d_x).pow(2)
    w  = torch.exp(-zeta * d_x)
    return (w[mask] * se[mask]).mean()

# ------------------------------
# 3) Training routine
# ------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
h5_path = project_path('Data_Generation', 'train_diffusion_nonlinear.h5')
dist_path = resolve_output_path('Data_Generation/manifold_dists_train_vorticity.npy')
save_path = resolve_output_path('Trained_Models/AE/Vorticity/AE_Vorticity_GA.pth')

# Hyperparameters
batch_size      = 100
test_batch_size = 10
lr              = 1e-3
gamma_loc       = 1e-3
zeta            = 1.0
knn             = 10
t_diffusion     = 10
num_epochs      = 1000
patience        = 1000

metrics = ErrorMetrics()

# Load data
with h5py.File(h5_path, 'r') as f:
    train_vort = torch.tensor(f['train_vorticity_64'][:], dtype=torch.float32)
    test_vort  = torch.tensor(f['train_vorticity_64'][:100], dtype=torch.float32)

# Precompute distances if needed
if not os.path.exists(dist_path):
    compute_and_store_phate_diff_potential_distances(h5_path, knn, t_diffusion, dist_path)

# Load memmapped distances
D = np.load(dist_path, mmap_mode='r')

# Datasets and loaders (with indices)
train_ds = TensorDataset(train_vort, torch.arange(len(train_vort)))
test_ds  = TensorDataset(test_vort,  torch.arange(len(test_vort)))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=test_batch_size, shuffle=False)

# Model & optimizer
model           = VariationalAutoEncoder().to(device)
optimizer       = torch.optim.Adam(model.parameters(), lr=lr)
scheduler       = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=30)
recon_criterion = nn.MSELoss()

best_val_loss = float('inf')
counter       = 0

for epoch in range(1, num_epochs+1):
    model.train()
    train_sum = 0.0

    for inputs, idxs in train_loader:
        inputs, idxs = inputs.to(device), idxs.to(device)
        z      = model.encode(inputs)
        recon  = model.decode(z)
        recon_loss = recon_criterion(recon, inputs)

        # flattened_latent_x = z.view(z.shape[0], -1)
        # latent_mean = flattened_latent_x.mean(dim=0)
        # latent_var = flattened_latent_x.var(dim=0, unbiased=True)
        # kl_divergence = 0.5 * (latent_var + latent_mean ** 2 - 1 - torch.log(
        #     latent_var + 1e-8))
        # var_loss = kl_divergence.mean() * 1e-3

        geo_loss   = geometry_loss(z, idxs, D, zeta) * gamma_loc
        loss       = recon_loss + geo_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_sum += loss.item() * inputs.size(0)

    avg_train = train_sum / len(train_loader.dataset)

    # validation
    model.eval()
    val_sum = 0.0
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            recon  = model.decode(model.encode(inputs))
            val_sum += recon_criterion(recon, inputs).item() * inputs.size(0)
    avg_val = val_sum / len(test_loader.dataset)

    fro_err_val = metrics.frobenius(inputs, recon).item()
    print(f"Epoch {epoch}/{num_epochs}  train_loss={avg_train:.4e}  val_loss={avg_val:.4e}, fro_err={fro_err_val:.4e}")
    scheduler.step(avg_val)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        counter = 0
        torch.save(model.state_dict(), save_path)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping.")
            break

print("Best validation loss:", best_val_loss)
