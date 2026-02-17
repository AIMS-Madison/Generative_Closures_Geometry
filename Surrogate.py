from functools import partial
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from utils import *
from Models.Generative_Closures.Generative_Models import *
from Models.Generative_Closures.Interpolant import *
from Models.Pretrained_Autoencoders.AE import *

from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda')
# Load the data
num_ensembles = 1000

onedrive_path = 'C:\\Users\\dongx\\OneDriveUWM'
Surrogate_file_path = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Data", "surrogate_3050_v2.h5")
with h5py.File(Surrogate_file_path, 'r') as file:
    sol = torch.tensor(file['sol'][:], device=device)
    nonlinear = torch.tensor(file['nonlinear'][:], device=device)

sol_start = sol[..., 0].repeat(num_ensembles, 1, 1)
metrics = ErrorMetrics()


# Viscosity parameter
nu = 1e-3

# Spatial Resolution
s = 64
# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s + 1, device=device)
t = t[0:-1]

X, Y = torch.meshgrid(t, t)
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

def navier_stokes_2d_nonlinear(a, w0, f, visc, sampler, AEW, AEG, num_ensembles, closure = False, delta_t=1e-4, record_steps=1, eval_steps=10):
    # Grid size - must be power of 2
    N1, N2 = w0.size()[-2], w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N1 / 2.0)

    # Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)
    # Forcing to Fourier space
    f_h = torch.fft.rfft2(f)
    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                     torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N1, 1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)

    # Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    # Physical wavenumbers
    kx_2d = 2.0 * torch.pi * k_x / a[0]
    ky_2d = 2.0 * torch.pi * k_y / a[1]

    # Negative Laplacian in Fourier space
    lap = kx_2d ** 2 + ky_2d ** 2
    lap[0, 0] = 1.0

    sol = torch.zeros(*w0.size(), 5, device=w0.device)
    sol_t = torch.zeros(5, device=w0.device)

    t = 0.0

    start_time = time.time()
    for i in tqdm(range(record_steps)):
        w = torch.fft.irfft2(w_h, s=(N1, N2))

        if closure == True:
            if i % eval_steps == 0:
                if AEW != None:
                    with torch.no_grad():
                        w_cond = AEW.encode(w)
                else:
                    w_cond = w
                nonlinear_sample_ensemble = sampler(omega=w_cond)
                nonlinear_sample = torch.mean(nonlinear_sample_ensemble, dim=0, keepdim=True).repeat(num_ensembles, 1, 1)
                if AEG != None:
                    with torch.no_grad():
                        nonlinear_sample = AEG.decode(nonlinear_sample)
            else:
                nonlinear_sample = nonlinear_sample + torch.randn_like(nonlinear_sample) * 0.00005

            # convection term
            nonlinear_h = torch.fft.rfft2(nonlinear_sample)

            w_h = ((w_h  + delta_t * f_h + delta_t * nonlinear_h
                            - 0.5 * delta_t * visc * lap * w_h)
                           / (1.0 + 0.5 * delta_t * visc * lap))

        if closure == False:
            w_h = ((w_h  + delta_t * f_h
                            - 0.5 * delta_t * visc * lap * w_h)
                           / (1.0 + 0.5 * delta_t * visc * lap))
        if i == 0:
            sol[..., 0] = w
            sol_t[0] = t
        if (i+1) % 5000 == 0:
            j = int((i+1) / 5000)
            sol[..., j] = w
            sol_t[j] = t
        t += delta_t
    end_time = time.time()

    execution_time = end_time - start_time
    return sol, sol_t, execution_time

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
    Unified closure sampler for 'diffusion', 'onesided', and 'twosided' methods.
    - 'diffusion'  : standard score-based SDE sampler (unchanged).
    - 'onesided'   : one-sided interpolation (as before).
    - 'twosided'   : two-sided bridge sampler, using a single drift network b(x,t; x1)
                     and no separate score.
                     gamma(t) = t*(1-t), so gamma'(t) = 1 - 2*t.

    Returns:
        Tensor of shape (B, num_ensembles, H, W).  If return_starting_field=True,
    """
    B, H, W = omega.shape
    omega = omega.to(device)
    model = model.to(device)
    model.eval()

    # 1) Build the time grid ts = [time_max, ..., time_min]
    if time_schedule == 'uniform':
        ts = torch.linspace(time_max, time_min, sample_steps+1, device=device)
        ts_forward = torch.linspace(time_min, time_max, sample_steps+1, device=device)
    elif time_schedule == 'karras':
        ts = get_sigmas_karras(sample_steps, time_min, time_max, device=device)
        ts_forward = reversed(ts)
    else:
        raise ValueError(f"Unknown time_schedule: {time_schedule}")
    dt = ts[:-1] - ts[1:]  # step sizes for Euler integration
    dt_forward = ts_forward[1:] - ts_forward[:-1]

    # 2) Tile conditioning omega to match ensemble dimension
    # omega_rep = omega.unsqueeze(1).repeat(1, num_ensembles, 1, 1).view(-1, H, W)
    omega_rep = omega
    if method == 'diffusion':
        from functools import partial
        marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma_coef, device_=device)
        diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma_coef, device_=device)

        # 3) Initialize x at t = ts[0] ~ N(0, [marginal std]^2)
        t0 = ts[0].repeat( num_ensembles)
        x = torch.randn_like(omega_rep) * marginal_prob_std_fn(t0)[:, None, None]


        with torch.no_grad():
            for i in range(sample_steps):
                t_i = ts[i].repeat( num_ensembles)
                step_size = dt[i]

                g = diffusion_coeff_fn(t_i)        # shape [B*num_ensembles]
                grad = model(t_i, x, omega_rep)    # score: [B*num_ensembles, H, W]

                # Euler–Maruyama update:
                mean = x + (g**2)[:, None, None] * grad * step_size
                if i < sample_steps - 1:
                    noise = torch.randn_like(x)
                    x = mean + math.sqrt(step_size) * g[:, None, None] * noise
                else:
                    x = mean # no extra noise on final step

        # x = x.view(B, num_ensembles, H, W)
        return x

    elif method == 'interpolant_knownsde':
        # 3) Initialize X_t at t=ts[0]=1 as x1 = omega_rep
        Xt = omega_rep.clone()

        with torch.no_grad():
            for i in range(sample_steps):
                t = ts_forward[i]
                step_size = dt_forward[i]                          # > 0
                t_batch = t.repeat( num_ensembles).to(device)
                drift = model(t_batch, Xt, omega_rep)

                # 4.3) Euler–Maruyama backward step (no sign flip: dt>0, but X_{t-dt}=X_t - drift*dt + ...)
                if i < sample_steps - 1:
                    noise = torch.randn_like(Xt)
                    gdot_scalar = 1 - t
                    gdot = gdot_scalar.repeat(num_ensembles)
                    gdot = gdot.view([num_ensembles] + [1] * (Xt.ndim - 1))
                    Xt = Xt + drift * step_size + gdot * math.sqrt(step_size) * noise
                else:
                    Xt = Xt + drift * step_size

        # Xt = Xt.view(B, num_ensembles, H, W)
        return Xt


    elif method == 'interpolant_gaussiansde':
        # 3) Initialize X_t at t=ts[0]=1 as x1 = omega_rep
        Xt = torch.randn_like(omega_rep)

        with torch.no_grad():
            for i in range(sample_steps):
                t = ts_forward[i]
                step_size = dt_forward[i]                          # > 0
                # t_batch = t.repeat( num_ensembles).to(device)
                t_batch = t.repeat(num_ensembles).to(device)
                drift = model(t_batch, Xt, omega_rep)      # [B*num_ensembles, H, W]
                gdot_scalar = 1 - t
                gdot = gdot_scalar.repeat(num_ensembles)
                gdot = gdot.view([num_ensembles] + [1] * (Xt.ndim - 1))

                # 4.3) Euler–Maruyama backward step (no sign flip: dt>0, but X_{t-dt}=X_t - drift*dt + ...)
                if i < sample_steps - 1:
                    noise = torch.randn_like(Xt)
                    # Xt = Xt + drift * step_size
                    Xt = Xt + drift * step_size + gdot * math.sqrt(step_size) * noise
                else:
                    Xt = Xt + drift * step_size

        # Xt = Xt.view(B, num_ensembles, H, W)
        return Xt

    elif method == 'interpolant_gaussianode':
        # 3) Initialize X_t at t=ts[0]=1 as x1 = omega_rep
        Xt = torch.randn_like(omega_rep)
        with torch.no_grad():
            for i in range(sample_steps):
                t = ts_forward[i]
                step_size = dt_forward[i]                          # > 0
                t_batch = t.repeat( num_ensembles).to(device)
                drift = model(t_batch, Xt, omega_rep)      # [B*num_ensembles, H, W]
                Xt = Xt + drift * step_size
        # Xt = Xt.view(B, num_ensembles, H, W)
        return Xt

    else:
        raise ValueError(f"Unknown sampling method: {method}")

# P-SI
set_seed(42)

modes = 6
width = 40
SISDE = FNO2d_Orig(modes, modes, width, padding = 0, embed_dim = 256, length = 1).to(device)

model_name = 'Models/Stochastic_Interpolant/Stochastic_Interpolant_gaussianbase_sde.pth'
SISDE.load_state_dict(torch.load(model_name))

SI_Sampler = partial(sample_closure_field, method='interpolant_gaussiansde', model=SISDE, num_ensembles=num_ensembles, sample_steps=10, sigma_coef=1.0, time_min=0, time_max=1.0, time_schedule='uniform', device=device)


sol_sisde, sol_t_sisde, exec_time_sisde = navier_stokes_2d_nonlinear(
    a=[1.0, 1.0],
    w0=sol_start,
    f=f,
    visc=nu,
    sampler=SI_Sampler,
    AEW=None,
    AEG=None,
    num_ensembles=num_ensembles,
    closure=True,
    delta_t=1e-3, record_steps=20000, eval_steps=5
)

# P-FM
modes = 6
width = 40
FMODE = FNO2d_Orig(modes, modes, width, padding = 0, embed_dim = 256, length = 1).to(device)

model_name = 'Models/Stochastic_Interpolant/Stochastic_Interpolant_gaussianbase_ode.pth'
FMODE.load_state_dict(torch.load(model_name))

FM_Sampler = partial(sample_closure_field, method='interpolant_gaussianode', model=FMODE, num_ensembles=num_ensembles, sample_steps=2, sigma_coef=1.0, time_min=0, time_max=1.0, time_schedule='uniform', device=device)

sol_fmode, sol_t_fmode, exec_time_fmode = navier_stokes_2d_nonlinear(
    a=[1.0, 1.0],
    w0=sol_start,
    f=f,
    visc=nu,
    sampler=FM_Sampler,
    AEW=None,
    AEG=None,
    num_ensembles=num_ensembles,
    closure=True,
    delta_t=1e-3, record_steps=20000, eval_steps=5
)

# P-CDM
modes = 6
width = 40
sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
pcdm = FNO2d_Diffusion(marginal_prob_std_fn, modes, modes, width, padding = 0, embed_dim = 512, length = 1).to(device)
model_name = 'Models/Score_Diffusion/P-CDM.pth'
pcdm.load_state_dict(torch.load(model_name))

CDM_Sampler = partial(sample_closure_field, method='diffusion', model=pcdm, num_ensembles=num_ensembles, sample_steps=10, sigma_coef=sigma, time_min=1e-3, time_max=0.1, time_schedule='karras', device=device)

sol_pcdm, sol_t_pcdm, exec_time_pcdm = navier_stokes_2d_nonlinear(
    a=[1.0, 1.0],
    w0=sol_start,
    f=f,
    visc=nu,
    sampler=CDM_Sampler,
    AEW=None,
    AEG=None,
    num_ensembles=num_ensembles,
    closure=True,
    delta_t=1e-3, record_steps=20000, eval_steps=5
)

# L-SI with MP
modes = 4
width = 20
LSIMP = FNO2d_Orig(modes, modes, width, padding = 0, embed_dim = 256, length = 1).to(device)

model_name = 'Models/Stochastic_Interpolant/Latent_SI/Stochastic_Interpolant_gaussianbase_sde_MP.pth'
LSIMP.load_state_dict(torch.load(model_name))
AEG = VariationalAutoEncoder().to(device)
AEW = VariationalAutoEncoder().to(device)

AEG_path = 'Models/Autoencoders/AE_6416_nonlinear_MP.pth'
AEW_path = 'Models/Autoencoders/AE_6416_vorticity_MP.pth'

AEG.load_state_dict(torch.load(AEG_path))
AEW.load_state_dict(torch.load(AEW_path))

LSI_Sampler_MP = partial(sample_closure_field, method='interpolant_gaussiansde', model=LSIMP, num_ensembles=num_ensembles, sample_steps=10, sigma_coef=1.0, time_min=0, time_max=1.0, time_schedule='uniform', device=device)

sol_lsi_mp, sol_t_lsi_mp, exec_time_lsi_mp = navier_stokes_2d_nonlinear(
    a=[1.0, 1.0],
    w0=sol_start,
    f=f,
    visc=nu,
    sampler=LSI_Sampler_MP,
    AEW=AEW,
    AEG=AEG,
    num_ensembles=num_ensembles,
    closure=True,
    delta_t=1e-3, record_steps=20000, eval_steps=5
)


# L-SI with GA
modes = 4
width = 20
LSIGA = FNO2d_Orig(modes, modes, width, padding = 0, embed_dim = 256, length = 1).to(device)

model_name = 'Models/Stochastic_Interpolant/Latent_SI/Stochastic_Interpolant_gaussianbase_sde_GA.pth'
LSIGA.load_state_dict(torch.load(model_name))
AEG = VariationalAutoEncoder().to(device)
AEW = VariationalAutoEncoder().to(device)

AEG_path = 'Models/Autoencoders/AE_6416_nonlinear_GA.pth'
AEW_path = 'Models/Autoencoders/AE_6416_vorticity_GA.pth'

AEG.load_state_dict(torch.load(AEG_path))
AEW.load_state_dict(torch.load(AEW_path))

LSI_Sampler_GA = partial(sample_closure_field, method='interpolant_gaussiansde', model=LSIGA, num_ensembles=num_ensembles, sample_steps=10, sigma_coef=1.0, time_min=0, time_max=1.0, time_schedule='uniform', device=device)

sol_lsi_ga, sol_t_lsi_ga, exec_time_lsi_ga = navier_stokes_2d_nonlinear(
    a=[1.0, 1.0],
    w0=sol_start,
    f=f,
    visc=nu,
    sampler=LSI_Sampler_GA,
    AEW=AEW,
    AEG=AEG,
    num_ensembles=num_ensembles,
    closure=True,
    delta_t=1e-3, record_steps=20000, eval_steps=5
)

# Joint L-CDM
sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)

modes = 4
width = 20
JointLCDM = FNO2d_Diffusion(marginal_prob_std_fn, modes, modes, width, padding = 0, embed_dim = 256, length = 1).to(device)
model_name = 'Models/Score_Diffusion/Latent_Diffusion/JointAE_Diffusion.pth'
JointLCDM.load_state_dict(torch.load(model_name))

AEG = VariationalAutoEncoder().to(device)
AEW = VariationalAutoEncoder().to(device)
AEG_path = 'Models/Autoencoders/Joint_AE_Nonlinear_Diffusion.pth'
AEW_path = 'Models/Autoencoders/Joint_AE_Vorticity_Diffusion.pth'

AEG.load_state_dict(torch.load(AEG_path))
AEW.load_state_dict(torch.load(AEW_path))

JointCDM_Sampler = partial(sample_closure_field, method='diffusion', model=JointLCDM, num_ensembles=num_ensembles, sample_steps=10, sigma_coef=sigma, time_min=1e-3, time_max=0.4, time_schedule='karras', device=device)

sol_jointcdm, sol_t_jointcdm, exec_time_jointcdm = navier_stokes_2d_nonlinear(
    a=[1.0, 1.0],
    w0=sol_start,
    f=f,
    visc=nu,
    sampler=JointCDM_Sampler,
    AEW=AEW,
    AEG=AEG,
    num_ensembles=num_ensembles,
    closure=True,
    delta_t=1e-3, record_steps=20000, eval_steps=5
)




rel_err_col = torch.zeros(100, device=device)
mse_err_col = torch.zeros(100, device=device)

for j in range(5):
    for i in range(100):
        rel_err_col[i] = metrics.frobenius(sol_lsi_ga[i:i+1, :, :, j], sol[:, :, :, (j) * 5000])

    print(f"Frobenius Error Mean: {torch.mean(rel_err_col):.2e}")
    print(f"Frobenius Error 2Std: {2 * torch.std(rel_err_col):.2e}")