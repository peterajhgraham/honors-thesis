# Honors Thesis - EEG Signal Detection and ROC Analysis

**Author:** Peter Alexander 捷虎 Graham

**Institution:** Brown University

**Contact:** peter_graham@brown.edu

---

## Abstract

This repository contains the complete codebase for an Honors Thesis investigating
the detectability of canonical EEG frequency bands (delta, theta, alpha, beta, gamma)
under varying noise conditions. Synthetic EEG signals are constructed from
Morlet-style wavelets governed by n-gram token sequences. Detection performance
is quantified via Receiver Operating Characteristic (ROC) analysis under three
noise regimes: white Gaussian noise, pink (1/f) noise, and structured noise
modelled after real EEG recordings.

---

## Thesis Document

The full written thesis is available in this repository:

**[`honors_thesis_comp_neuro.pdf`](honors_thesis_comp_neuro.pdf)**

---

## Repository Structure

```
Honors_Thesis/
│
├── README.md                             # This file
├── honors_thesis_comp_neuro.pdf          # Full thesis document
├── code.py                               # Original monolithic script (archived)
├── requirements.txt                      # Python dependencies
│
├── src/                                  # Reusable library modules
│   ├── __init__.py
│   │
│   ├── sequences/                        # Token-sequence generators
│   │   ├── __init__.py
│   │   ├── bigram.py                     # Bigram sequence generation & heatmap plotting
│   │   └── ngram.py                      # Higher-order n-gram sequence generation
│   │
│   ├── gradient_descent/                 # Manual gradient-descent implementation
│   │   ├── __init__.py
│   │   └── gradient_descent.py           # Analytical gradient computation & training loop
│   │
│   ├── visualization/                    # General-purpose visualisation utilities
│   │   ├── __init__.py
│   │   └── surface_3d.py                 # Interactive 3-D loss-surface plot (Plotly)
│   │
│   └── eeg/                              # EEG-specific modules
│       ├── __init__.py
│       │
│       ├── synthetic/                    # Signal generation
│       │   ├── __init__.py
│       │   ├── signal_generation.py      # Multi-band sinusoidal EEG synthesis
│       │   └── wavelet_generation.py     # Morlet-wavelet EEG synthesis from token sequences
│       │
│       ├── decomposition/                # Time-frequency decomposition
│       │   ├── __init__.py
│       │   ├── stft_decomposition.py     # STFT band-pass decomposition & reconstruction
│       │   └── dwt_decomposition.py      # Discrete Wavelet Transform (DWT) decomposition
│       │
│       └── analysis/                     # Detection & evaluation
│           ├── __init__.py
│           ├── noise.py                  # White, pink, and EEG-like noise generators
│           ├── roc_analysis.py           # STFT-based ROC analysis over noise levels
│           ├── roc_real_eeg.py           # Correlation-based ROC with real-EEG-like noise
│           └── plotting.py              # Publication-quality plotting utilities
│
└── experiments/                          # Runnable experiment scripts
    ├── run_bigram.py                     # Bigram validation & heatmaps
    ├── run_gradient_descent.py           # Gradient descent training & loss curve
    ├── run_eeg_synthetic.py              # EEG generation, STFT/DWT decomposition
    ├── run_roc_white_noise.py            # ROC analysis under white noise
    ├── run_roc_pink_noise.py             # ROC analysis under pink (1/f) noise
    └── run_roc_real_eeg_noise.py         # ROC analysis with real-EEG-layered noise
```

---

## Module Descriptions

### `src/sequences/` — Token Sequence Generation

| File | Purpose |
|------|---------|
| **`bigram.py`** | Generates random token sequences governed by bigram (first-order Markov) transition probabilities sampled from a Dirichlet prior. Includes a heatmap visualisation function to compare empirical vs. theoretical transition matrices. |
| **`ngram.py`** | Extends sequence generation to arbitrary n-gram orders. Provides both a full Dirichlet-sampled version (for realistic statistical structure) and a simplified uniform-random version (for controlled experiments). |

### `src/gradient_descent/` — Manual Gradient Descent

| File | Purpose |
|------|---------|
| **`gradient_descent.py`** | A from-scratch implementation of gradient descent for a single-layer linear network `y = Wx`. Gradients are computed analytically (`dL/dW = (y_pred - y_true) * x^T`) without automatic differentiation. Includes a training loop that logs per-epoch loss values for visualisation. |

### `src/visualization/` — General Visualisation

| File | Purpose |
|------|---------|
| **`surface_3d.py`** | Creates an interactive 3-D surface plot using Plotly to visualise a combined quadratic-sinusoidal function `z = x^2 - y^2 + sin(3x)cos(3y)`, illustrating the optimisation landscapes that gradient descent must navigate. |

### `src/eeg/synthetic/` — Synthetic EEG Signal Generation

| File | Purpose |
|------|---------|
| **`signal_generation.py`** | Generates synthetic multi-band EEG signals by superimposing sinusoidal components at the five canonical frequency bands (delta 0.5–4 Hz, theta 4–8 Hz, alpha 8–12 Hz, beta 12–30 Hz, gamma 30–100 Hz) with additive Gaussian noise. Also provides a single-frequency chunk generator for token-based signal construction. |
| **`wavelet_generation.py`** | Creates Morlet-style wavelet signals from token sequences. Each token maps to a Gaussian-windowed sinusoidal burst at its characteristic frequency, producing time-localised oscillatory events that mimic transient EEG activity. |

### `src/eeg/decomposition/` — Time-Frequency Decomposition

| File | Purpose |
|------|---------|
| **`stft_decomposition.py`** | Implements STFT-based frequency-band decomposition. A band-pass filter is applied in the STFT domain by zeroing out frequency bins outside the target range, then the inverse STFT reconstructs the isolated band. Also provides spectrogram computation. |
| **`dwt_decomposition.py`** | Performs multi-level Discrete Wavelet Transform decomposition using the Daubechies-4 (`db4`) wavelet, demonstrating perfect reconstruction from wavelet coefficients. |

### `src/eeg/analysis/` — Detection and Evaluation

| File | Purpose |
|------|---------|
| **`noise.py`** | Provides three noise generators: (1) **white noise** via `np.random.randn`, (2) **pink noise** with a 1/f^alpha spectrum constructed by FFT filtering, and (3) **structured EEG-like noise** built from a deterministic harmonic basis that can be circularly shifted to produce 100+ reproducible yet distinct noise realisations. |
| **`roc_analysis.py`** | Core ROC evaluation engine. For each noise level, generates a wavelet signal from a random token sequence, corrupts it with noise, computes the STFT, extracts band-specific power, and sweeps detection thresholds to produce per-band ROC curves with AUC and precision metrics. |
| **`roc_real_eeg.py`** | Extends ROC analysis to use structured EEG-like noise and a correlation-based template-matching detector. Generates signal-present and signal-absent trials using circularly shifted noise patterns, producing realistic ROC curves that reflect actual detection performance. |
| **`plotting.py`** | Publication-quality plotting utilities with a consistent colour palette (Delta=blue, Theta=orange, Alpha=green, Beta=red, Gamma=purple). Includes functions for ROC subplot grids, AUC-vs-noise curves, precision-vs-noise curves, clean/noisy signal grids, and wave-component breakdowns. |

---

## Experiments

Each script in `experiments/` is a self-contained experiment that imports
from `src/` and can be run from the repository root:

```bash
python -m experiments.run_bigram
python -m experiments.run_gradient_descent
python -m experiments.run_eeg_synthetic
python -m experiments.run_roc_white_noise
python -m experiments.run_roc_pink_noise
python -m experiments.run_roc_real_eeg_noise
```

| Script | Description |
|--------|-------------|
| **`run_bigram.py`** | Generates a 10,000-token bigram sequence, validates that empirical transition probabilities converge to the sampled conditional distribution, and renders comparative heatmaps. |
| **`run_gradient_descent.py`** | Trains a single-layer linear network for 100 epochs using hand-coded gradient descent, plots the loss curve, and renders the 3-D loss surface. |
| **`run_eeg_synthetic.py`** | Generates a synthetic multi-band EEG signal, decomposes it into five bands via STFT, demonstrates DWT reconstruction, renders spectrograms, and performs threshold-based event detection on noisy signals. |
| **`run_roc_white_noise.py`** | Runs the full ROC experiment across 12 white-noise levels (0.1–500), producing ROC curves, AUC-vs-noise plots (log and linear), precision curves, and a performance summary table. |
| **`run_roc_pink_noise.py`** | Identical experimental protocol to the white-noise version but using pink (1/f) noise, enabling direct comparison of detection robustness under different noise spectral profiles. |
| **`run_roc_real_eeg_noise.py`** | Uses structured EEG-like noise with 100 circularly shifted realisations per condition. Employs a correlation-based template detector and includes a variability analysis showing how ROC curves fluctuate across different noise patterns. |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation and linear algebra |
| `scipy` | Signal processing (STFT, iSTFT, filtering) |
| `matplotlib` | 2-D plotting and figure generation |
| `seaborn` | Statistical heatmap visualisation |
| `plotly` | Interactive 3-D surface plots |
| `scikit-learn` | ROC curve computation and AUC scoring |
| `pandas` | Tabular data management for experiment results |
| `PyWavelets` | Discrete Wavelet Transform |
| `tqdm` | Progress bars for long-running experiments |
| `torch` | PyTorch (imported in original; available for extensions) |

Install all dependencies:

```bash
pip install numpy scipy matplotlib seaborn plotly scikit-learn pandas PyWavelets tqdm torch
```

---

## Key Concepts

### EEG Frequency Bands

| Band  | Frequency Range | Associated Activity |
|-------|----------------|---------------------|
| Delta | 0.5–4 Hz | Deep sleep, unconscious processes |
| Theta | 4–8 Hz | Drowsiness, light sleep, meditation |
| Alpha | 8–12 Hz | Relaxed wakefulness, eyes closed |
| Beta  | 12–30 Hz | Active thinking, focus, anxiety |
| Gamma | 30–100 Hz | Higher cognitive functions, perception |

### Signal Detection Pipeline

1. **Sequence Generation** — Random token sequences define which frequency band is active at each time step
2. **Wavelet Synthesis** — Each token produces a Gaussian-windowed sinusoidal burst at its characteristic frequency
3. **Noise Injection** — White, pink, or structured EEG-like noise is added at controlled levels
4. **STFT Decomposition** — The noisy signal is transformed to the time-frequency domain
5. **Band Power Extraction** — Mean spectral power is computed within each canonical band
6. **Threshold Detection** — Band power is compared against a sweep of thresholds
7. **ROC Evaluation** — True/false positive rates and AUC are computed against ground-truth token locations
