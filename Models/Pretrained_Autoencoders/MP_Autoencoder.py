import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
import h5py
import torch
import torch.nn as nn
from AE import VariationalAutoEncoder
from utils import ErrorMetrics

# ------------------------------
# Local closeness loss
# ------------------------------
def local_closeness_loss(x, z, k=5):
    B = x.shape[0]
    # flatten spatial dims if present
    x_flat = x.view(B, -1)
    # flatten latent dims
    z_flat = z.view(B, -1)
    d_x = torch.cdist(x_flat, x_flat, p=2)
    d_z = torch.cdist(z_flat, z_flat, p=2)
    loss = 0.0
    for i in range(B):
        dx = d_x[i].clone()
        dx[i] = float('inf')
        idxs = torch.topk(dx, k, largest=False).indices
        loss += ((d_z[i,idxs] - d_x[i,idxs])**2).mean()
    return loss / B
# ------------------------------
# Data loading
# ------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

file_name = 'Data_Generation/train_diffusion_nonlinear.h5'
with h5py.File(file_name, 'r') as file:
    X_train = torch.tensor(file['train_nonlinear_64'][:], device=device)
    X_test = torch.tensor(file['train_nonlinear_64'][:100], device=device)

train_loader = torch.utils.data.DataLoader(X_train, batch_size=180, shuffle=True)
test_loader = torch.utils.data.DataLoader(X_test, batch_size=10, shuffle=False)

# ------------------------------
# Model, optimizer, etc.
# ------------------------------
model = VariationalAutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=30)
recon_criterion = nn.MSELoss()

# Hyperparameters for AE loss
beta_kl = 0.0       # no KL for baseline
gamma_loc = 1e-2    # weight for local closeness term
k_neighbors = 5

# Training loop with early stopping
num_epochs = 1000
patience   = 1000
best_val_loss = float('inf')
counter = 0
metrics = ErrorMetrics()
# Training loop with local closeness
for epoch in range(1, num_epochs+1):
    model.train()
    train_loss, train_fro = 0.0, 0.0
    for inputs in train_loader:
        inputs = inputs.to(device)
        # encode and decode
        latent = model.encode(inputs)
        decoded = model.decode(latent)
        # reconstruction loss
        recon_loss = recon_criterion(decoded, inputs)
        # local closeness on manifolds
        loc_loss = local_closeness_loss(inputs, latent, k=5) * gamma_loc
        # total
        loss = recon_loss + loc_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_fro  += metrics.frobenius(inputs, decoded)
    train_loss /= len(train_loader.dataset)
    train_fro  /= len(train_loader)
    # validation
    model.eval()
    val_loss, val_fro = 0.0, 0.0
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            latent = model.encode(inputs)
            decoded = model.decode(latent)
            val_loss += recon_criterion(decoded, inputs).item() * inputs.size(0)
            val_fro  += metrics.frobenius(inputs, decoded)
    val_loss /= len(test_loader.dataset)
    val_fro  /= len(test_loader)

    print(f'Epoch {epoch}/{num_epochs} | Train Loss:{train_loss:.4e}'
            f'| Train Frobenius:{train_fro:.4e} '
          f'| Val Loss:{val_loss:.4e} | Val Frobenius:{val_fro:.4e} ')
    scheduler.step(val_loss)
    # early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'mp_autoencoder.pth')
    else:
        counter += 1
        if counter >= patience:
            print('Early stopping.')
            break

print('Best validation loss:', best_val_loss)
