import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
from torch.optim import Adam
from utils import *
from Models.Generative_Closures.Interpolant import *
from Models.Generative_Closures.Generative_Models import FNO2d_Orig
from Models.Pretrained_Autoencoders.AE import VariationalAutoEncoder
from project_paths import project_path

warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader

def train_joint_si(
    h5file='Data_Generation/train_diffusion_nonlinear.h5',
    cond_key='train_vorticity_64',
    targ_key='train_nonlinear_64',
    max_samples=10000,
    batch_size=100,
    sigma_coef=1.0,
    lr=1e-3,
    epochs=1000,
    scheduler_step=100,
    scheduler_gamma=0.5,
    device='auto',
    save_dir='Trained_Models/SI/Latent_SI/Joint_training',
):
    device = resolve_device(device)
    h5file = Path(h5file)
    save_dir = Path(save_dir)
    if not h5file.is_absolute():
        h5file = project_path(h5file)
    if not save_dir.is_absolute():
        save_dir = project_path(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    full_dataset = H5ClosureDataset(h5file, cond_key, targ_key, max_samples=max_samples)
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    AEH_model = VariationalAutoEncoder().to(device)
    AEW_model = VariationalAutoEncoder().to(device)
    SI_model = FNO2d_Orig(modes1=4, modes2=4, width=20, padding=0, embed_dim=256, length=1).to(device)
    
    optimizer = Adam(list(SI_model.parameters()) + list(AEW_model.parameters()) + list(AEH_model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    interp = Interpolant(device)
    metrics = ErrorMetrics()
    mse = torch.nn.MSELoss()

    for ep in range(1, epochs + 1):
        AEH_model.train()
        AEW_model.train()
        SI_model.train()
        total_loss = 0.0
        for w, x in train_loader:
            x, w = x.to(device), w.to(device)
            optimizer.zero_grad()

            # Autoencoder encode/decode
            latent_x = AEH_model.encode(x)
            recon_x = AEH_model.decode(latent_x)
            latent_w = AEW_model.encode(w)
            recon_w = AEW_model.decode(latent_w)

            # loss - reconstruction
            recon_loss_x = mse(recon_x, x) * 1000
            recon_loss_w = mse(recon_w, w)

            # loss - KL on latent_x
            flat_latent = latent_x.view(latent_x.size(0), -1)
            mean = flat_latent.mean(dim=0)
            var = flat_latent.var(dim=0, unbiased=True)
            kl_loss = 0.5 * (var + mean**2 - 1 - torch.log(var + 1e-8)).mean() * 0.1

            # SI loss in latent space
            B = latent_x.size(0)
            t = torch.rand(B, device=device)
            xt, R = interp.sample_knownbase(latent_x, latent_w, t)
            pred = SI_model(t, xt, latent_w)
            si_loss = mse(pred, R)

            loss = recon_loss_x + recon_loss_w + kl_loss + si_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * B

            fro_x = metrics.frobenius(x, recon_x)
            fro_w = metrics.frobenius(w, recon_w)

        scheduler.step()
        avg_loss = total_loss / len(full_dataset)

        print(f"[JointSI] Ep{ep}/{epochs} Loss={avg_loss:.3e} ReconX={recon_loss_x.item():.3e} "
                f"ReconW={recon_loss_w.item():.3e} KL={kl_loss.item():.3e} SI={si_loss.item():.3e} "
                f"FroX={fro_x:.3e} FroW={fro_w:.3e}")

    torch.save(SI_model.state_dict(), os.path.join(save_dir, 'Joint_SI_KnownSDE.pth'))
    torch.save(AEH_model.state_dict(), os.path.join(save_dir, 'Joint_AE_Nonlinear_SI_KnownSDE.pth'))
    torch.save(AEW_model.state_dict(), os.path.join(save_dir, 'Joint_AE_Vorticity_SI_KnownSDE.pth'))

    # Save final loss statistics.
    with open(os.path.join(save_dir, 'loss_stats.txt'), 'a') as f:
        f.write(f"{ep} {avg_loss:.3e} {recon_loss_x.item():.3e} {recon_loss_w.item():.3e} "
                f"{kl_loss.item():.3e} {si_loss.item():.3e} {fro_x:.3e} {fro_w:.3e}\n")

    print("Joint training complete.")
    return SI_model, AEH_model, AEW_model, interp

if __name__ == '__main__':
    set_seed(35)
    train_joint_si()
