# Synergizing transport-based generative models and latent geometry for stochastic closure modeling

Official implementation of the paper ["Synergizing transport-based generative models and latent geometry for stochastic closure modeling"](https://doi.org/10.1016/j.cpc.2026.110341).

Xinghao Dong, Huchen Yang, and Jin-Long Wu

---

## Overview

This repository implements transport-based generative models for stochastic closure modeling of two-dimensional Kolmogorov flow. It compares diffusion models (DM), flow matching (FM), and stochastic interpolants (SI) in physical and latent spaces. The latent models include conventional two-phase training with reconstruction-only, metric-preserving (MP), or geometry-aware (GA) autoencoders, as well as end-to-end joint training. Pretrained checkpoints, compact test data, and numerical demos are included.

### Highlights

- **Systematic transport-model comparison:** Diffusion, flow matching, and stochastic interpolant closures are evaluated under a common numerical setting. Straighter flow-matching transport paths enable single-step generation with sampling speedups of up to two orders of magnitude over iterative diffusion sampling.
- **Geometry-regularized latent modeling:** Reconstruction-only autoencoders can distort conditional structure. Joint training provides implicit regularization, while MP and GA explicitly organize the latent geometry, improving closure-sample fidelity and data efficiency.
- **Solver-coupled uncertainty quantification:** The regularized latent generators integrate with the physics-based solver and reproduce full-system statistics while reducing the computational cost of ensemble simulation.

This release focuses on the two-dimensional stochastic Kolmogorov-flow benchmark. Compact data and pretrained models are provided for the released demonstrations. 

## Repository Structure

```text
Data/                         Compact test and rollout data
Data_Generation/              Two-dimensional NSE data generator
Demos/                        Physical, latent, and solver-coupled notebooks
Models/
  Generative_Closures/        DM, FM, and SI models and trainers
  Pretrained_Autoencoders/    NoReg, MP, GA, and joint autoencoders
Trained_Models/
  AE/                         Autoencoder checkpoints
  DM/                         Physical and latent diffusion checkpoints
  FM/                         Physical and latent flow-matching checkpoints
  SI/                         Physical and latent interpolant checkpoints
project_paths.py              Repository-relative path configuration
utils.py                      Shared data, sampling, and metric utilities
```

## Setup

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/AIMS-Madison/Generative_Closures_Geometry.git
cd Generative_Closures_Geometry
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CUDA is recommended for training and generative sampling. The notebooks select CUDA automatically when it is available.

## Data

The repository includes compact datasets for evaluating the supplied checkpoints:

- `Data/test_diffusion_nonlinear_B100.h5`: paired vorticity and closure fields;
- `Data/closure_sim_5steps20sec_subset.h5`: a short trajectory for solver-coupled closure evaluation.

Full training data are not included. The training entry points expect generated files under `Data_Generation/`; these can be produced with the included two-dimensional NSE solver or obtained from the authors upon reasonable request.

## Training

Run commands from the repository root. Hyperparameters are defined near the top of each research training script.

### Autoencoders

```bash
python Models/Pretrained_Autoencoders/Vanilla_Autoencoder.py
python Models/Pretrained_Autoencoders/MP_Autoencoder.py
python Models/Pretrained_Autoencoders/GA_Autoencoder.py
```

### Two-Phase Generative Models

```bash
python Models/Generative_Closures/Score_Diffusion/TP_Diffusion_Train.py
python Models/Generative_Closures/Flow_Matching/TP_FM_Train.py
python Models/Generative_Closures/Stochastic_Interpolants/TP_SI_Train_GaussianSDE.py
python Models/Generative_Closures/Stochastic_Interpolants/TP_SI_Train_KnownSDE.py
```

### Joint Training

```bash
python Models/Generative_Closures/Score_Diffusion/Joint_Diffusion_Train.py
python Models/Generative_Closures/Flow_Matching/Joint_FM_Train.py
python Models/Generative_Closures/Stochastic_Interpolants/Joint_SI_Train.py
```

## Running Demos

- [Physical_Space_Models.ipynb](Demos/Physical_Space_Models.ipynb) evaluates physical-space DM, FM, and SI closures.
- [Latent_Space_Models.ipynb](Demos/Latent_Space_Models.ipynb) evaluates latent autoencoders and latent generative models.
- [Sims_With_Closures.ipynb](Demos/Sims_With_Closures.ipynb) couples a learned latent closure to the numerical solver.

Run the setup cells first and then execute the model section of interest in order. Compare results at the level of error magnitude and qualitative behavior rather than exact floating-point agreement across hardware.

## Citation

```bibtex
@article{dong2026synergizing,
  title   = {Synergizing transport-based generative models and latent geometry for stochastic closure modeling},
  author  = {Dong, Xinghao and Yang, Huchen and Wu, Jin-Long},
  journal = {Computer Physics Communications},
  volume  = {328},
  pages   = {110341},
  year    = {2026},
  doi     = {10.1016/j.cpc.2026.110341}
}
```
