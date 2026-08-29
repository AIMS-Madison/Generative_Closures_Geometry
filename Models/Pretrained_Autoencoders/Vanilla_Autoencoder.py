import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import h5py
import torch
import torch.nn as nn
from Models.Pretrained_Autoencoders.AE import VariationalAutoEncoder
from project_paths import project_path, resolve_output_path
from utils import ErrorMetrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'

file_name = project_path('Data_Generation', 'train_diffusion_nonlinear.h5')
with h5py.File(file_name, 'r') as file:
    X_train = torch.tensor(file['train_nonlinear_64'][:], device=device)
    X_test = torch.tensor(file['train_nonlinear_64'][:100], device=device)

train_loader = torch.utils.data.DataLoader(X_train, batch_size=180, shuffle=True)
test_loader = torch.utils.data.DataLoader(X_test, batch_size=10, shuffle=False)

model = VariationalAutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=30)
recon_criterion = nn.MSELoss()

num_epochs = 1000
patience   = 1000
best_val_loss = float('inf')
counter = 0
metrics = ErrorMetrics()
for epoch in range(1, num_epochs+1):
    model.train()
    train_loss, train_fro = 0.0, 0.0
    for inputs in train_loader:
        inputs = inputs.to(device)
        latent = model.encode(inputs)
        decoded = model.decode(latent)
        recon_loss = recon_criterion(decoded, inputs)
        loss = recon_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        train_fro  += metrics.frobenius(inputs, decoded)
    train_loss /= len(train_loader.dataset)
    train_fro  /= len(train_loader)
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
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(
            model.state_dict(),
            resolve_output_path('Trained_Models/AE/Nonlinear/AE_Nonlinear_ReconOnly.pth'),
        )
    else:
        counter += 1
        if counter >= patience:
            print('Early stopping.')
            break

print('Best validation loss:', best_val_loss)
