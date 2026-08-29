import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import warnings
import numpy as np
import torch
import h5py
from torch.optim import Adam
from functools import partial
from tqdm import trange
from utils import *
from Models.Generative_Closures.Generative_Models import (
    FNO2d_Diffusion,
    diffusion_coeff,
    loss_fn,
    marginal_prob_std,
)
from Models.Pretrained_Autoencoders.AE import VariationalAutoEncoder
from project_paths import project_path, resolve_output_path

np.set_printoptions(suppress=False, formatter={'float': '{:.2e}'.format})
torch.set_printoptions(sci_mode=True)
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')


def load_data():
    train_name = project_path('Data_Generation', 'train_diffusion_nonlinear.h5')

    print(f"Loading training data from {train_name}")
    with h5py.File(train_name, 'r') as file:
        train_nonlinear = torch.tensor(file['train_nonlinear_64'][:10000], device=device)
        train_vorticity = torch.tensor(file['train_vorticity_64'][:10000], device=device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_nonlinear, train_vorticity, torch.arange(len(train_nonlinear))),
        batch_size=100, shuffle=True
    )
    return train_loader


train_loader = load_data()

sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 4
width = 20
padding = 0
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

AEG_model = VariationalAutoEncoder().to(device)
AEW_model = VariationalAutoEncoder().to(device)
diffusion_model = FNO2d_Diffusion(marginal_prob_std_fn, modes, modes, width, padding, embed_dim=256, length=1).to(device)


optimizer = Adam(list(diffusion_model.parameters()) + list(AEW_model.parameters()) + list(AEG_model.parameters()),
                 lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

metrics = ErrorMetrics()

def train():
    loss_history = []
    tqdm_epoch = trange(epochs, desc="Training")
    for epoch in tqdm_epoch:
        diffusion_model.train()
        AEW_model.train()
        AEG_model.train()
        total_loss = 0.0
        num_items = 0

        for x, w, idx in train_loader:
            x, w, idx = x.to(device), w.to(device), idx.to(device)
            optimizer.zero_grad()

            latent_x = AEG_model.encode(x)
            recon_x = AEG_model.decode(latent_x)
            fro_x = metrics.frobenius(x, recon_x)


            flattened_latent_x = latent_x.view(latent_x.shape[0], -1)
            latent_mean = flattened_latent_x.mean(dim=0)
            latent_var = flattened_latent_x.var(dim=0, unbiased=True)
            kl_divergence = 0.5 * (latent_var + latent_mean ** 2 - 1 - torch.log(latent_var + 1e-8))
            var_loss = kl_divergence.mean() * 0.1
            
            latent_w = AEW_model.encode(w)
            recon_w = AEW_model.decode(latent_w)
            fro_w = metrics.frobenius(w, recon_w)
            recon_loss_x = torch.nn.MSELoss()(recon_x, x) * 100
            recon_loss_w = torch.nn.MSELoss()(recon_w, w)


            score_loss, _, _ = loss_fn(diffusion_model, latent_x, latent_w, None, marginal_prob_std_fn, sparse=False)

            loss = score_loss + recon_loss_x + recon_loss_w + var_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        scheduler.step()
        avg_loss = total_loss / num_items
        loss_history.append(avg_loss)
        tqdm_epoch.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.5f}, Fro Nonlinear: {fro_x:.5f}, Fro Vorticity: {fro_w:.5f}")

    diffusion_model_save = resolve_output_path(
        'Trained_Models/DM/Latent_DM/Joint_DM.pth'
    )
    torch.save(diffusion_model.state_dict(), diffusion_model_save)

    AEG_model_save = resolve_output_path(
        'Trained_Models/AE/Nonlinear/Joint_AE_Nonlinear_DM.pth'
    )
    torch.save(AEG_model.state_dict(), AEG_model_save)

    AEW_model_save = resolve_output_path(
        'Trained_Models/AE/Vorticity/Joint_AE_Vorticity_DM.pth'
    )
    torch.save(AEW_model.state_dict(), AEW_model_save)


if __name__ == '__main__':
    set_seed(42)
    train()
