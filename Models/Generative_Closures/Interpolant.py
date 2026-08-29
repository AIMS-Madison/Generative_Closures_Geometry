import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import math


class Interpolant:
    def __init__(self, device='cuda'):
        self.device = device

    def sample_knownbase(self, target, condition, t):
        """
        Two‐sided forward interpolation:
          X_t = (1 - t)*condition + t*target + (1-t)*W_t,
          where W_t = sqrt(t)*noise.

        Returns:
          zt = X_t,
          R  = ∂_t X_t = -condition + target - W_t
        """
        B = t.size(0)
        t_ = t.view(B, 1, 1).to(self.device) * (1-1e-6) + 1e-6

        a, b      = 1 - t_,    t_
        a_dot     = -torch.ones_like(t_)
        b_dot     =  torch.ones_like(t_)

        gamma     = 1 - t_
        gamma_dot = -torch.ones_like(t_)

        noise = torch.randn_like(target, device=self.device)
        W_t   = torch.sqrt(t_) * noise

        zt = a * condition + b * target + gamma * W_t

        R  = a_dot * condition + b_dot * target + gamma_dot * W_t

        return zt, R

    def sample_gaussianbase_sde(self, target, t):
        """
        One‐sided interpolation with SDE sampling:
          X_t = (1 - t)*Z + t*target +  (1-t)*W_t,  W_t = sqrt(t)*noise

        Returns:
          zt = X_t,
          R  = ∂_t X_t = -Z + target - W_t
        """
        B = t.size(0)
        t_ = t.view(B, 1, 1).to(self.device)

        a, b      = 1 - t_,    t_
        a_dot     = -torch.ones_like(t_)
        b_dot     =  torch.ones_like(t_)

        gamma     = 1 - t_
        gamma_dot = -torch.ones_like(t_)

        noise = torch.randn_like(target, device=self.device)
        W_t   = torch.sqrt(t_) * noise

        noise_start = torch.randn_like(target, device=self.device)

        zt = a * noise_start + b * target + gamma * W_t

        R  = a_dot * noise_start + b_dot * target + gamma_dot * W_t

        return zt, R

    def sample_gaussianbase_ode(self, target, t):
        """
        One‐sided interpolation with ode sampling:
          X_t = (1 - t)*Z + t*target

        Returns:
          zt = X_t,
          R  = ∂_t X_t = -Z + target
        """
        B = t.size(0)
        t_ = t.view(B, 1, 1).to(self.device)

        a, b = 1 - t_, t_
        a_dot = -torch.ones_like(t_)
        b_dot = torch.ones_like(t_)

        noise_start = torch.randn_like(target, device=self.device)

        zt = a * noise_start + b * target

        R = a_dot * noise_start + b_dot * target

        return zt, R


def train_fno_interpolant(
    cond_data,
    targ_data,
    model: nn.Module,
    mode: str = 'gaussianbase_sde',  # 'knownbase', 'gaussianbase_sde', 'gaussianbase_ode'
    batch_size: int = 32,
    lr: float = 2e-4,
    epochs: int = 50,
    device: str = 'cuda',
    save_path: str = 'fno_si.pth',
    save_loss: str = 'loss_stats.txt',
    eps: float = 1e-12,
):
    full_ds = TensorDataset(targ_data, cond_data)
    train_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)

    interp = Interpolant(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_norm_loss = 0.0

        for target, condition in train_loader:
            target = target.to(device).float()
            condition = condition.to(device).float()
            B = target.size(0)

            t = torch.rand(B, device=device)

            if mode == 'knownbase':
                x1 = target
                x0 = condition
                interpolated_states, velocity = interp.sample_knownbase(x1, x0, t)
                pred = model(t, interpolated_states, x0)

            elif mode == 'gaussianbase_sde':
                x1 = target
                x0 = condition
                interpolated_states, velocity = interp.sample_gaussianbase_sde(x1, t)
                pred = model(t, interpolated_states, x0)

            elif mode == 'gaussianbase_ode':
                x1 = target
                x0 = condition
                interpolated_states, velocity = interp.sample_gaussianbase_ode(x1, t)
                pred = model(t, interpolated_states, x0)

            loss = mse(pred, velocity)

            v_flat = velocity.reshape(B, -1)
            v_power = (v_flat.pow(2).sum(dim=1)).mean()  # E||v||^2
            norm_loss = loss / (v_power + eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_norm_loss += norm_loss.item() * B

        scheduler.step()
        avg_loss = total_loss / len(full_ds)
        avg_norm_loss = total_norm_loss / len(full_ds)

        print(f"[LinSI:{mode}] Ep{ep}/{epochs} loss={avg_loss:.3e} norm_loss={avg_norm_loss:.3e}")

        with open(save_loss, 'a') as f:
            f.write(f"{ep} {avg_loss:.6e} {avg_norm_loss:.6e}\n")

    torch.save(model.state_dict(), save_path)
    return model



def sample_closure_latent(
    drift_model: torch.nn.Module,
    z_omega: torch.Tensor,
    num_steps: int = 500,
    num_ensembles: int = 10,
    sigma_coef: float = 1.0,
    device: str = 'cuda',
    use_follmer: bool = False,
    start_gaussian: bool = False,
) -> torch.Tensor:
    """
    Unified sampler for both original SI (use_follmer=False)
    and Föllmer‐optimal SI (use_follmer=True).
    """
    drift_model.eval()
    omega = z_omega.to(device)
    B, n, _ = omega.shape

    omega_rep = omega.unsqueeze(1).repeat(1, num_ensembles, 1, 1)
    omega_rep = omega_rep.view(-1, n, n)  # (B*num_ensembles, n, n)

    if start_gaussian:
        zt = torch.randn_like(omega_rep, device=device)
    else:
        zt = omega_rep.clone()

    ts = torch.linspace(0.0, 1.0, num_steps, device=device)
    dt = 1.0 / (num_steps - 1)

    with torch.no_grad():
        for i, t in enumerate(ts):
            t_batch   = t.repeat(B * num_ensembles)

            drift = drift_model(t_batch, zt, omega_rep)

            sigma_t = sigma_coef * (1.0 - t)

            if use_follmer:
                alpha_t   = 1.0 - t
                beta_t    = t * t
                alpha_dot = -1.0
                beta_dot  = 2.0 * t
                sigma_dot = -sigma_coef

                g_t = math.sqrt(abs(
                    2 * t * sigma_t * (beta_dot * sigma_t - beta_t * sigma_dot)
                    - sigma_t**2
                ))

                # Aₛ = 1 / [s σₛ (β̇ₛσₛ - βₛσ̇ₛ)]
                denom = (t * sigma_t * (beta_dot * sigma_t - beta_t * sigma_dot))
                A_t = 1.0 / (denom + 1e-12)

                # cₛ(x,x₀) = β̇ₛ·x + (βₛα̇ₛ - αₛβ̇ₛ)·x₀
                c_t = beta_dot * zt + (beta_t * alpha_dot - alpha_t * beta_dot) * omega_rep

                score = A_t * (beta_t * drift - c_t)

                drift = drift + 0.5 * (g_t**2 - sigma_t**2) * score
                noise_scale = g_t

            else:
                noise_scale = sigma_t

            if i < num_steps - 1:
                noise = torch.randn_like(zt, device=device)
                zt = zt + drift * dt + noise_scale * math.sqrt(dt) * noise
            else:
                zt = zt + drift * dt

    return zt.view(B, num_ensembles, n, n)
