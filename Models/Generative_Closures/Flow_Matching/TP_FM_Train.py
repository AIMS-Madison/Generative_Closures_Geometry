import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
from Models.Generative_Closures.Interpolant import *
from Models.Generative_Closures.Generative_Models import FNO2d_Orig
from project_paths import project_path, resolve_output_path
from utils import resolve_device


device = resolve_device('auto')


# GA latent space
train_file = project_path('Data_Generation', 'train_diffusion_nonlinear_encoded_GA.h5')
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_encoded'][:], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_encoded'][:], device=device)
model = FNO2d_Orig(modes1=4, modes2=4, width=20).to(device)

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode='gaussianbase_ode',
    batch_size=100,
    lr=1e-3,
    epochs=1000,
    device=device,
    save_path=resolve_output_path('Trained_Models/FM/Latent_FM/L-FM_GA.pth'),
    save_loss=resolve_output_path('training_logs/FM_GA_loss.txt')
)



# MP latent space
train_file = project_path('Data_Generation', 'train_diffusion_nonlinear_encoded_MP.h5')
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_encoded'][:], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_encoded'][:], device=device)
model = FNO2d_Orig(modes1=4, modes2=4, width=20).to(device)

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode='gaussianbase_ode',
    batch_size=100,
    lr=1e-3,
    epochs=1000,
    device=device,
    save_path=resolve_output_path('Trained_Models/FM/Latent_FM/L-FM_MP.pth'),
    save_loss=resolve_output_path('training_logs/FM_MP_loss.txt')
)

# Reconstruction-only latent space
train_file = project_path('Data_Generation', 'train_diffusion_nonlinear_encoded.h5')
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_encoded'][:], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_encoded'][:], device=device)
model = FNO2d_Orig(modes1=4, modes2=4, width=20).to(device)

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode='gaussianbase_ode',
    batch_size=100,
    lr=1e-3,
    epochs=1000,
    device=device,
    save_path=resolve_output_path('Trained_Models/FM/Latent_FM/L-FM_ReconOnly.pth'),
    save_loss=resolve_output_path('training_logs/FM_NoReg_loss.txt')
)

# Physical space
train_file = project_path('Data_Generation', 'train_diffusion_nonlinear.h5')
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_64'][:], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_64'][:], device=device)
model = FNO2d_Orig(modes1=6, modes2=6, width=40).to(device)

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode = 'gaussianbase_ode',
    batch_size=100,
    lr=1e-3,
    epochs=1000,
    device=device,
    save_path=resolve_output_path('Trained_Models/FM/Physics_FM/P-FM.pth'),
    save_loss=resolve_output_path('training_logs/FM_physical_loss.txt')
)

