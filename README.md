# EEG Frequency-Band Detection Under Noise

Brown University Honors Thesis in Computational Neuroscience. Python 3.9+. MIT License.

**Canonical EEG frequency bands remain reliably detectable under realistic noise regimes when the underlying signal is modeled as a sequence of time-localised Morlet wavelet bursts rather than as stationary band-limited oscillations.**

## Abstract

This repository builds a controlled testbed for evaluating frequency-band detection in EEG-like signals. Synthetic signals are generated as sequences of Morlet-style wavelet bursts indexed by n-gram token sequences, then corrupted with three noise families - white Gaussian, pink (1/f), and structured noise modelled on real EEG - across twelve noise levels spanning four orders of magnitude. Detection is quantified via STFT band-power and template-correlation detectors, evaluated with full ROC curves, AUC, and precision metrics per canonical band (delta, theta, alpha, beta, gamma). The result is a reproducible benchmark for the brittleness of common time-frequency detectors and a generative model that captures the transient, event-like character of real neural oscillations.

## Why This Matters

EEG analysis pipelines - and most ML models built on top of them - implicitly assume frequency-band power is a stable, slowly-varying feature. In practice neural oscillations are *bursty*: they appear as short-lived events embedded in coloured noise. Detecting them well is the prerequisite for almost every downstream task: BCI decoding, sleep staging, seizure prediction, cognitive workload estimation. This work asks the question quantitatively - *under what noise regime does each canonical band become undetectable?* - and provides a generative model (token sequences over wavelet bursts) that is closer to the statistics of cortical activity than the band-limited-noise priors used in most synthetic EEG benchmarks. More broadly, the n-gram-over-wavelets formulation is a small step toward treating neural recordings as discrete-token sequences with learnable temporal structure, a framing increasingly relevant to neural foundation models.

## Repository Structure

```
honors-thesis/
├── README.md                             # This file
├── RESEARCH_CONTEXT.md                   # Plain-language ML-audience explainer
├── honors_thesis_comp_neuro.pdf          # Full thesis document
├── code.py                               # Original monolithic script (archived)
├── requirements.txt                      # Python dependencies
│
├── src/                                  # Library modules
│   ├── sequences/                        # Token-sequence generators (bigram, n-gram)
│   ├── gradient_descent/                 # From-scratch gradient descent
│   ├── visualization/                    # 3-D loss-surface visualisation
│   └── eeg/
│       ├── synthetic/                    # Multi-band sinusoidal + Morlet wavelet synthesis
│       ├── decomposition/                # STFT and DWT decomposition
│       └── analysis/                     # Noise generators, ROC analysis, plotting
│
└── experiments/                          # Runnable experiment scripts
    ├── run_bigram.py
    ├── run_gradient_descent.py
    ├── run_eeg_synthetic.py
    ├── run_roc_white_noise.py
    ├── run_roc_pink_noise.py
    └── run_roc_real_eeg_noise.py
```

## Quickstart

```bash
pip install -r requirements.txt

python -m experiments.run_eeg_synthetic        # Generate + decompose a synthetic EEG signal
python -m experiments.run_roc_white_noise      # ROC sweep under white Gaussian noise
python -m experiments.run_roc_pink_noise       # ROC sweep under pink (1/f) noise
python -m experiments.run_roc_real_eeg_noise   # ROC sweep with structured EEG-like noise
python -m experiments.run_bigram               # Bigram sequence validation
python -m experiments.run_gradient_descent     # Gradient descent demo + 3D loss surface
```

## Method

### 1. Token sequences over canonical bands

Tokens index the five canonical EEG bands. Sequences are drawn from n-gram models with transition matrices sampled from a Dirichlet prior, giving controlled temporal structure (see `src/sequences/`). This lets us decouple the *what* (which band is active) from the *when* (the temporal statistics of band switching).

### 2. Morlet wavelet synthesis

Each token emits a Gaussian-windowed sinusoidal burst at its band's characteristic frequency (`src/eeg/synthetic/wavelet_generation.py`). The resulting signal is a sum of time-localised oscillatory events - the same form used to model evoked oscillations and burst-like cortical activity - rather than a stationary mixture of band-limited sinusoids.

### 3. Noise regimes

Three noise families (`src/eeg/analysis/noise.py`):
- **White Gaussian** - flat-spectrum baseline.
- **Pink (1/f^α)** - constructed by FFT filtering; matches the spectral slope of real EEG.
- **Structured EEG-like** - a deterministic harmonic basis circularly shifted to yield 100+ reproducible noise realisations with realistic auto-correlation.

### 4. Detection and evaluation

Two detectors:
- **STFT band-power** - invert STFT after zeroing bins outside each target band, then integrate power (`stft_decomposition.py`, `roc_analysis.py`).
- **Template-correlation** - match against the wavelet template for each band (`roc_real_eeg.py`).

Thresholds are swept to produce per-band ROC curves; we report AUC and precision across twelve noise levels per regime. DWT reconstruction (`dwt_decomposition.py`, Daubechies-4) is provided as an orthogonal time-frequency reference.

## Module Reference

### `src/sequences/` - Token sequence generation

| File | Purpose |
|------|---------|
| `bigram.py` | Bigram (first-order Markov) sequences with Dirichlet-prior transition matrices; heatmap of empirical vs. theoretical transitions. |
| `ngram.py` | Higher-order n-gram generation; both Dirichlet-sampled and uniform variants. |

### `src/gradient_descent/` - From-scratch gradient descent

| File | Purpose |
|------|---------|
| `gradient_descent.py` | Single-layer linear network `y = Wx`, analytic gradient `dL/dW = (y_pred − y_true) xᵀ`, no autodiff. |

### `src/visualization/`

| File | Purpose |
|------|---------|
| `surface_3d.py` | Interactive Plotly surface for `z = x² − y² + sin(3x)cos(3y)`. |

### `src/eeg/synthetic/` - Signal generation

| File | Purpose |
|------|---------|
| `signal_generation.py` | Multi-band sinusoidal EEG synthesis across delta/theta/alpha/beta/gamma with additive Gaussian noise. |
| `wavelet_generation.py` | Morlet-style wavelet synthesis from token sequences; each token → Gaussian-windowed sinusoidal burst. |

### `src/eeg/decomposition/` - Time-frequency decomposition

| File | Purpose |
|------|---------|
| `stft_decomposition.py` | STFT band-pass via bin-zeroing + inverse STFT; spectrogram utility. |
| `dwt_decomposition.py` | Multi-level DWT with Daubechies-4; perfect reconstruction. |

### `src/eeg/analysis/` - Detection and evaluation

| File | Purpose |
|------|---------|
| `noise.py` | White, pink (1/f^α), and structured EEG-like noise (100+ circularly shifted realisations). |
| `roc_analysis.py` | STFT band-power ROC engine: signal + noise → STFT → band power → threshold sweep → per-band ROC/AUC/precision. |
| `roc_real_eeg.py` | Correlation-based template detector with structured noise; signal-present/absent trials via circular shifts. |
| `plotting.py` | Publication-quality plots; consistent palette (Δ blue, θ orange, α green, β red, γ purple). |

## Canonical EEG Frequency Bands

| Band  | Range | Associated activity |
|-------|--------|---------------------|
| Delta | 0.5–4 Hz | Deep sleep, unconscious processes |
| Theta | 4–8 Hz | Drowsiness, light sleep, meditation |
| Alpha | 8–12 Hz | Relaxed wakefulness, eyes closed |
| Beta  | 12–30 Hz | Active thinking, focus |
| Gamma | 30–100 Hz | Higher cognition, binding, perception |

New to EEG? See [`RESEARCH_CONTEXT.md`](RESEARCH_CONTEXT.md) for a plain-language walkthrough of the bands, why detecting them under noise is hard, and what the n-gram-over-wavelets formulation is doing conceptually.

## Detection Pipeline

1. **Sequence** - token sequence over canonical bands (n-gram model).
2. **Synthesis** - each token emits a Morlet wavelet burst at its band frequency.
3. **Noise** - additive white, pink, or structured EEG-like noise at controlled SNR.
4. **Decomposition** - STFT (or DWT) into the time-frequency domain.
5. **Feature** - band power (STFT) or template correlation.
6. **Threshold sweep** - produce ROC curve against ground-truth token locations.
7. **Metrics** - per-band AUC and precision across noise levels.

## Related Work

- **Morlet wavelets for neural oscillations.** Tallon-Baudry & Bertrand (1999), *Oscillatory γ-band activity in humans and its role in object representation*, Trends in Cognitive Sciences - the canonical reference for Morlet wavelet analysis of transient oscillatory events in EEG/MEG, which motivates the burst-based generative model used here.
- **ROC analysis in BCI.** Lotte, Bougrain, Cichocki, Clerc, Congedo, Rakotomamonjy & Yger (2018), *A review of classification algorithms for EEG-based brain–computer interfaces: a 10-year update*, J. Neural Engineering - surveys detection / classification metrics (including ROC/AUC) on EEG and frames the noise-robustness question this benchmark targets.
- **Discrete token structure in neural sequences.** Vahidi et al. (2024), *Modeling neural activity with conditionally linear dynamical systems / discrete latent sequence models* (and the broader neural-foundation-model literature, e.g. Ye et al., *Neural Data Transformer*) - recent work treating spike and field recordings as token sequences with learnable temporal structure, the framing this thesis's n-gram-over-wavelets construction prefigures.

## Citation

```bibtex
@thesis{graham2024eeg,
  author = {Graham, Peter Alexander},
  title  = {EEG Signal Detection and ROC Analysis under Structured Noise},
  school = {Brown University},
  year   = {2024},
  type   = {Honors Thesis, Computational Neuroscience}
}
```

## Author

**Peter Alexander 捷虎 Graham** - Brown University - peter_graham@brown.edu

Full thesis: [`honors_thesis_comp_neuro.pdf`](honors_thesis_comp_neuro.pdf)
