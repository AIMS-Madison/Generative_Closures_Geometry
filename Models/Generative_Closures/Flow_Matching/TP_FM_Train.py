import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import h5py
from Interpolant import *
from Generative_Models import FNO2d_Orig


# 1) Train in GA latent space
train_file = 'Data_Generation\\train_diffusion_nonlinear_encoded_GA.h5'
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_encoded'][:], device='cuda')
    train_vorticity = torch.tensor(file['train_vorticity_encoded'][:], device='cuda')
model = FNO2d_Orig(modes1=4, modes2=4, width=20).to('cuda')

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode='gaussianbase_ode',
    batch_size=100,
    lr=1e-3,
    epochs=1000,
    device='cuda',
    save_path='FM_GA.pth',
    save_loss='FM_GA_loss.txt'
)



# 2) Train in MP latent space
train_file = 'Data_Generation\\train_diffusion_nonlinear_encoded_MP.h5'
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_encoded'][:], device='cuda')
    train_vorticity = torch.tensor(file['train_vorticity_encoded'][:], device='cuda')
model = FNO2d_Orig(modes1=4, modes2=4, width=20).to('cuda')

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode='gaussianbase_ode',
    batch_size=100,
    lr=1e-3,
    epochs=1000,
    device='cuda',
    save_path='FM_MP.pth',
    save_loss='FM_MP_loss.txt'
)

# 3) Train in vanilla latent space
train_file = 'Data_Generation\\train_diffusion_nonlinear_encoded.h5'
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_encoded'][:], device='cuda')
    train_vorticity = torch.tensor(file['train_vorticity_encoded'][:], device='cuda')
model = FNO2d_Orig(modes1=4, modes2=4, width=20).to('cuda')

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode='gaussianbase_ode',
    batch_size=100,
    lr=1e-3,
    epochs=1000,
    device='cuda',
    save_path='FM_NoReg.pth',
    save_loss='FM_NoReg_loss.txt'
)

# 4) Train in physical space
train_file = 'Data_Generation\\train_diffusion_nonlinear.h5'
with h5py.File(train_file, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_64'][:], device='cuda')
    train_vorticity = torch.tensor(file['train_vorticity_64'][:], device='cuda')
model = FNO2d_Orig(modes1=6, modes2=6, width=40).to('cuda')

trained_model = train_fno_interpolant(
    cond_data = train_vorticity,
    targ_data=train_nonlinear,
    model=model,
    mode = 'gaussianbase_ode',
    batch_size=100,
    lr=1e-3,
    epochs=1000,
    device='cuda',
    save_path='FM_largestd.pth',
    save_loss='FM_loss_largestd.txt'
)

