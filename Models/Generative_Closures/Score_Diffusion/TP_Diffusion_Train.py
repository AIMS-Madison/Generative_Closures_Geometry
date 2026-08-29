import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import h5py
import torch
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
from project_paths import project_path, resolve_output_path

if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')

train_name = project_path('Data_Generation', 'train_diffusion_nonlinear.h5')


with h5py.File(train_name, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_64'][:10000], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_64'][:10000], device=device)


train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_nonlinear, train_vorticity),
    batch_size=500, shuffle=True
)

with torch.no_grad():
    current_max_dist = 0
    lam = 1e-6
    for i, (x, w) in enumerate(train_loader):
        x = x.to(device)
        x_ = x.view(x.shape[0], -1)
        max_dist = torch.cdist(x_, x_).max().item()

        if current_max_dist < max_dist:
            current_max_dist = max_dist
        print(current_max_dist)
    print('Final, max eucledian distance: {}'.format(current_max_dist))

sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 6
width = 40
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

model = FNO2d_Diffusion(marginal_prob_std_fn, modes, modes, width, padding = 0, embed_dim = 512, length = 1).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

tqdm_epoch = trange(epochs)

loss_history = []
rel_err_history = []

set_seed(42)
for epoch in tqdm_epoch:
    model.train()
    avg_loss = 0.
    num_items = 0
    for x, w in train_loader:
        x, w = x.to(device), w.to(device)
        optimizer.zero_grad()
        loss, _, _ = loss_fn(model, x, w, None, marginal_prob_std_fn, sparse=False)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    scheduler.step()
    avg_loss_epoch = avg_loss / num_items
    loss_history.append(avg_loss_epoch)
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

savepath = resolve_output_path('Trained_Models/DM/Physics_DM/P-CDM.pth')
torch.save(model.state_dict(), savepath)
