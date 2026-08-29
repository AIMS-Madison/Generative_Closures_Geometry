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


train_file = project_path('Data_Generation', 'train_diffusion_nonlinear_encoded_GA.h5')
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_encoded'][:], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_encoded'][:], device=device)
model = FNO2d_Orig(modes1=4, modes2=4, width=20).to(device)

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode='gaussianbase_sde',
    batch_size=200,
    lr=1e-3,
    epochs=500,
    device=device,
    save_path=resolve_output_path('Trained_Models/SI/Latent_SI/L-SI_GA_Gaussian.pth'),
)


train_file = project_path('Data_Generation', 'train_diffusion_nonlinear_encoded_MP.h5')
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_encoded'][:], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_encoded'][:], device=device)
model = FNO2d_Orig(modes1=4, modes2=4, width=20).to(device)

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode='gaussianbase_sde',
    batch_size=200,
    lr=1e-3,
    epochs=500,
    device=device,
    save_path=resolve_output_path('Trained_Models/SI/Latent_SI/L-SI_MP_Gaussian.pth'),
)

train_file = project_path('Data_Generation', 'train_diffusion_nonlinear_encoded.h5')
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_encoded'][:], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_encoded'][:], device=device)
model = FNO2d_Orig(modes1=4, modes2=4, width=20).to(device)

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode='gaussianbase_sde',
    batch_size=200,
    lr=1e-3,
    epochs=500,
    device=device,
    save_path=resolve_output_path('Trained_Models/SI/Latent_SI/L-SI_ReconOnly_Gaussian.pth'),
)

train_file = project_path('Data_Generation', 'train_diffusion_nonlinear.h5')
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_64'][:], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_64'][:], device=device)
model = FNO2d_Orig(modes1=6, modes2=6, width=40).to(device)

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode = 'gaussianbase_sde',
    batch_size=200,
    lr=1e-3,
    epochs=200,
    device=device,
    save_path=resolve_output_path('Trained_Models/SI/Physics_SI/P-SI_Gaussian.pth'),
)

