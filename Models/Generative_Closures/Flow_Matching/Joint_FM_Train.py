import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import warnings, os
import torch
from torch.optim import Adam
from utils import *
from Generative_Models import FNO2d_Orig
from Models.Pretrained_Autoencoders.AE import VariationalAutoEncoder
from Interpolant import *
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader

def train_joint_fm(
        h5file = r"Data_Generation//train_diffusion_nonlinear.h5",
        cond_key='train_vorticity_64',
        targ_key='train_nonlinear_64',
        max_samples=10000,
        batch_size=100,
        sigma_coef=1.0,
        lr=1e-3,
        epochs=1000,
        scheduler_step=100,
        scheduler_gamma=0.5,
        device='cuda',
        save_dir=r"Flow_Matching//Joint_FM",
        eps: float = 1e-12,  
):
    os.makedirs(save_dir, exist_ok=True)

    full_dataset = H5ClosureDataset(h5file, cond_key, targ_key, max_samples=max_samples)
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    AEH_model = VariationalAutoEncoder().to(device)
    AEW_model = VariationalAutoEncoder().to(device)
    FM_model = FNO2d_Orig(modes1=4, modes2=4, width=20, padding=0, embed_dim=256, length=1).to(device)

    optimizer = Adam(list(FM_model.parameters()) + list(AEW_model.parameters()) + list(AEH_model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    interp = Interpolant(device)
    metrics = ErrorMetrics()
    mse = torch.nn.MSELoss()

    for ep in range(1, epochs + 1):
        AEH_model.train()
        AEW_model.train()
        FM_model.train()
        total_loss = 0.0

        # 累积归一化 fm 损失：按样本数加权求 epoch 平均
        total_norm_fm = 0.0
        n_seen = 0

        # 为了保持你原有的打印方式，这里也保留“最后一个 batch 的各分项损失”
        recon_loss_x = recon_loss_w = kl_loss = fm_loss = norm_fm_loss = torch.tensor(0.0, device=device)
        fro_x = fro_w = 0.0

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
            kl_loss = 0.5 * (var + mean ** 2 - 1 - torch.log(var + 1e-8)).mean() * 0.1

            # SI loss in latent space
            B = latent_x.size(0)
            t = torch.rand(B, device=device)
            xt, R = interp.sample_gaussianbase_ode(latent_x, t)
            pred = FM_model(t, xt, latent_w)
            fm_loss = mse(pred, R)

            R_flat = R.view(B, -1)
            Evv_batch = (R_flat.pow(2).sum(dim=1)).mean()    
            norm_fm_loss = fm_loss / (Evv_batch + eps)

            loss = recon_loss_x + recon_loss_w + kl_loss + fm_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_norm_fm += norm_fm_loss.item() * B
            n_seen += B

            fro_x = metrics.frobenius(x, recon_x)
            fro_w = metrics.frobenius(w, recon_w)

        scheduler.step()
        avg_loss = total_loss / len(full_dataset)
        avg_norm_fm = total_norm_fm / n_seen if n_seen > 0 else 0.0

        print(
            f"[JointSI] Ep{ep}/{epochs} "
            f"Loss={avg_loss:.3e} ReconX={recon_loss_x.item():.3e} "
            f"ReconW={recon_loss_w.item():.3e} KL={kl_loss.item():.3e} "
            f"SI={fm_loss.item():.3e} SI_norm_avg={avg_norm_fm:.3e} "
            f"FroX={fro_x:.3e} FroW={fro_w:.3e}"
        )

        with open(os.path.join(save_dir, 'loss_stats.txt'), 'a') as f:
            f.write(
                f"{ep} {avg_loss:.3e} {recon_loss_x.item():.3e} {recon_loss_w.item():.3e} "
                f"{kl_loss.item():.3e} {fm_loss.item():.3e} {fro_x:.3e} {fro_w:.3e} {avg_norm_fm:.6e}\n"
            )

    torch.save(FM_model.state_dict(), os.path.join(save_dir, 'Joint_FM.pth'))
    torch.save(AEH_model.state_dict(), os.path.join(save_dir, 'Joint_AE_Nonlinear_FM.pth'))
    torch.save(AEW_model.state_dict(), os.path.join(save_dir, 'Joint_AE_Vorticity_FM.pth'))

    print("Joint training complete.")
    return FM_model, AEH_model, AEW_model, interp


if __name__ == '__main__':
    set_seed(35)
    train_joint_fm()
