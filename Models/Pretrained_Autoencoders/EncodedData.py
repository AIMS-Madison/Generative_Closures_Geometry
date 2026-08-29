import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import h5py
import torch
from Models.Pretrained_Autoencoders.AE import VariationalAutoEncoder
from project_paths import model_path, resolve_input_path, resolve_output_path

import warnings
warnings.filterwarnings("ignore")


def load_weights(path, map_location):
    """Load a tensor-only state dict on both old and new PyTorch releases."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')

train_name = resolve_input_path(
    'GCG_TRAIN_DATA', 'Data_Generation/train_diffusion_nonlinear.h5'
)
with h5py.File(train_name, 'r') as file:
    train_vorticity = torch.tensor(file['train_vorticity_64'][:], device=device)
    train_nonlinear = torch.tensor(file['train_nonlinear_64'][:], device=device)

test_name = resolve_input_path(
    'GCG_TEST_DATA', 'Data_Generation/test_diffusion_nonlinear.h5'
)
with h5py.File(test_name, 'r') as file:
    test_vorticity = torch.tensor(file['test_vorticity_64'][:], device=device)
    test_nonlinear = torch.tensor(file['test_nonlinear_64'][:], device=device)


convection_AE = VariationalAutoEncoder().to(device)
convection_AE.load_state_dict(load_weights(
    model_path('AE', 'Nonlinear', 'Joint_AE_Nonlinear_FM.pth'), device
))

vorticity_AE = VariationalAutoEncoder().to(device)
vorticity_AE.load_state_dict(load_weights(
    model_path('AE', 'Vorticity', 'Joint_AE_Vorticity_FM.pth'), device
))
convection_AE.eval()
vorticity_AE.eval()

filename = resolve_output_path(
    'Data_Generation/train_diffusion_nonlinear_encoded_joint_FM.h5'
)
batch_size = 1000
total_samples = min(18000, len(train_vorticity), len(train_nonlinear))

with torch.no_grad():
    sample_vort = vorticity_AE.encode(train_vorticity[0:1])
    sample_nonlin = convection_AE.encode(train_nonlinear[0:1])

sample_shape_vort = sample_vort[0].shape  # expected: (16, 16)
sample_shape_nonlin = sample_nonlin[0].shape  # expected: (16, 16)
print("Encoded vorticity sample shape:", sample_shape_vort)
print("Encoded nonlinear sample shape:", sample_shape_nonlin)

with h5py.File(filename, 'w') as file:
    dset_vort = file.create_dataset(
        'train_vorticity_encoded',
        shape=(0,) + sample_shape_vort,
        maxshape=(total_samples,) + sample_shape_vort,
        chunks=(1,) + sample_shape_vort,
        dtype='double'
    )
    dset_nonlin = file.create_dataset(
        'train_nonlinear_encoded',
        shape=(0,) + sample_shape_nonlin,
        maxshape=(total_samples,) + sample_shape_nonlin,
        chunks=(1,) + sample_shape_nonlin,
        dtype='double'
    )

    for i in range(0, total_samples, batch_size):
        with torch.no_grad():
            batch_vort = vorticity_AE.encode(
                train_vorticity[i:i + batch_size]
            )
            batch_nonlin = convection_AE.encode(
                train_nonlinear[i:i + batch_size]
            )

        batch_vort_np = batch_vort.cpu().numpy()
        batch_nonlin_np = batch_nonlin.cpu().numpy()

        cur_size = dset_vort.shape[0]
        new_size = cur_size + batch_vort_np.shape[0]

        dset_vort.resize(new_size, axis=0)
        dset_nonlin.resize(new_size, axis=0)

        dset_vort[cur_size:new_size] = batch_vort_np
        dset_nonlin[cur_size:new_size] = batch_nonlin_np

        print(f"Appended samples {cur_size} to {new_size - 1}")

print("All batches appended successfully.")







filename = resolve_output_path(
    'Data_Generation/test_diffusion_nonlinear_encoded_joint_FM.h5'
)
batch_size = 1000
total_samples = min(2000, len(test_vorticity), len(test_nonlinear))

with torch.no_grad():
    sample_vort = vorticity_AE.encode(test_vorticity[0:1])
    sample_nonlin = convection_AE.encode(test_nonlinear[0:1])

sample_shape_vort = sample_vort[0].shape  # expected: (16, 16)
sample_shape_nonlin = sample_nonlin[0].shape  # expected: (16, 16)
print("Encoded vorticity sample shape:", sample_shape_vort)
print("Encoded nonlinear sample shape:", sample_shape_nonlin)

with h5py.File(filename, 'w') as file:
    dset_vort = file.create_dataset(
        'test_vorticity_encoded',
        shape=(0,) + sample_shape_vort,
        maxshape=(total_samples,) + sample_shape_vort,
        chunks=(1,) + sample_shape_vort,
        dtype='double'
    )
    dset_nonlin = file.create_dataset(
        'test_nonlinear_encoded',
        shape=(0,) + sample_shape_nonlin,
        maxshape=(total_samples,) + sample_shape_nonlin,
        chunks=(1,) + sample_shape_nonlin,
        dtype='double'
    )

    for i in range(0, total_samples, batch_size):
        with torch.no_grad():
            batch_vort = vorticity_AE.encode(
                test_vorticity[i:i + batch_size]
            )
            batch_nonlin = convection_AE.encode(
                test_nonlinear[i:i + batch_size]
            )

        batch_vort_np = batch_vort.cpu().numpy()
        batch_nonlin_np = batch_nonlin.cpu().numpy()

        cur_size = dset_vort.shape[0]
        new_size = cur_size + batch_vort_np.shape[0]

        dset_vort.resize(new_size, axis=0)
        dset_nonlin.resize(new_size, axis=0)

        dset_vort[cur_size:new_size] = batch_vort_np
        dset_nonlin[cur_size:new_size] = batch_nonlin_np

        print(f"Appended samples {cur_size} to {new_size - 1}")

print("All batches appended successfully.")

with torch.no_grad():
    sample_vort = vorticity_AE.encode(train_vorticity[0:500])
    sample_nonlin = convection_AE.encode(test_nonlinear[0:500])
    decoded_vort = vorticity_AE.decode(sample_vort)
    decoded_nonlin = convection_AE.decode(sample_nonlin)

from utils import ErrorMetrics
metric = ErrorMetrics()

vorticity_error = metric.frobenius(train_vorticity[0:500], decoded_vort)
vorticity_mse = metric.mse(train_vorticity[0:500], decoded_vort)
print(f"Vorticity error: {vorticity_error.item():.3e}")
print(f"Vorticity MSE: {vorticity_mse.item():.3e}")
nonlinear_error = metric.frobenius(test_nonlinear[0:500], decoded_nonlin)
nonlinear_mse = metric.mse(test_nonlinear[0:500], decoded_nonlin)
print(f"Nonlinear error: {nonlinear_error.item():.3e}")
print(f"Nonlinear MSE: {nonlinear_mse.item():.3e}")
