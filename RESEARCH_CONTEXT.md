# Research Context

*A plain-language explainer for ML folks who have not spent years staring at EEG.*

## What EEG actually is

EEG is a handful of electrodes glued to a scalp recording voltage fluctuations at ~250–1000 Hz. The signal is the smeared, distance-attenuated sum of synchronous post-synaptic potentials from millions of cortical neurons. Think of it as a microphone in a stadium: you can hear that the crowd is roaring, you cannot hear any one person.

Because you are summing huge populations, the only things that survive are activities where lots of neurons fire together in time. That synchrony tends to be rhythmic — populations of neurons get into oscillatory regimes — so EEG is dominated by **band-limited rhythms** rather than the spike-like activity you would see with an intracortical electrode.

## The canonical frequency bands

Neuroscientists carved the EEG spectrum into five bands long before anyone had a principled reason to:

| Band   | Range      | Folk-meaning |
|--------|------------|--------------|
| Delta  | 0.5–4 Hz   | Deep sleep |
| Theta  | 4–8 Hz     | Drowsy, memory encoding, navigation |
| Alpha  | 8–12 Hz    | Eyes closed, idling visual cortex |
| Beta   | 12–30 Hz   | Active focus, motor control |
| Gamma  | 30–100 Hz  | "Binding" — perception, attention, higher cognition |

The bands are real in the sense that distinct generative mechanisms produce them, but the boundaries are conventional. Half of applied EEG / BCI work boils down to: *is band X elevated right now?*

## Why detecting bands is harder than it looks

Three reasons:

**1. The noise floor is not white.** Real neural noise has a roughly 1/f spectrum — more power at low frequencies, less at high. So a "5 µV bump at 2 Hz" is buried in much more background than a 5 µV bump at 40 Hz. Detectors tuned on white-noise assumptions will be wildly miscalibrated on real data.

**2. Oscillations are bursty, not stationary.** The textbook picture — "alpha is an 8–12 Hz sine wave" — is wrong. Cortical rhythms appear in *bursts*: a few cycles, then silence, then more. Methods that average power over long windows smear these bursts into something that looks like weak stationary band power, and the detection problem becomes much harder than it needs to be.

**3. Inter-trial / inter-subject noise is structured, not random.** When you record from a real brain, the "noise" is other brain activity. It has temporal structure, it correlates across electrodes, and it changes slowly. A detector that handles white Gaussian noise gracefully can fall apart when the noise has actual statistics.

This thesis tackles all three: pink noise for the spectrum problem, wavelet bursts for the non-stationarity, and a structured-noise regime (built from a harmonic basis with circular shifts) for the "noise has structure" problem.

## What the n-gram-over-wavelets thing is doing

This is the conceptually interesting bit, and the part most likely to feel familiar to an ML audience.

The standard synthetic EEG model is: pick five band-limited oscillators, sum them, add Gaussian noise, call it a brain. This is *bad*. It says the brain is in all five bands simultaneously and uniformly forever. Real cortex switches: it produces an alpha burst, then a theta burst, then gamma, transitioning between regimes on timescales of tens to hundreds of milliseconds.

So instead, model the signal as a **sequence of discrete tokens**, where each token is one of the five bands:

```
[θ θ α α α β γ β β α α α θ δ δ ...]
```

Each token, when "rendered," becomes a short Morlet wavelet — a Gaussian-windowed sinusoid at that band's characteristic frequency. The full signal is the concatenation (or overlap) of those bursts. That is the generative model.

The sequence itself is sampled from an **n-gram model** whose transition matrix is drawn from a Dirichlet prior. Bigrams give first-order Markov dependence (the probability of gamma next depends on what just happened); higher-order n-grams give longer-range structure. You can crank the Dirichlet concentration to interpolate between near-deterministic sequences and uniform random ones.

Why this is interesting from an ML angle:

- **It is a language model over neural activity.** The "vocabulary" is band-tokens; the "sentences" are temporal sequences of cortical states. Detection becomes: *given a noisy realisation of the rendered sequence, recover the tokens.* That is sequence labeling.
- **It separates content from timing.** The Dirichlet-sampled transition matrix carries the temporal grammar; the wavelet renderer carries the per-token waveform. You can vary either independently and study what each contributes to detectability.
- **It is closer to neural foundation models than the standard synthetic prior.** Recent work on neural data transformers and discrete latent sequence models for spike trains treats neural recordings as token streams. This thesis's construction is a small, controlled version of the same idea, applied to scalp EEG bands.

The wavelet-burst rendering is biologically motivated (this is how Tallon-Baudry, Bertrand, and the rest of the time-frequency literature model evoked oscillations), but the n-gram sequencing on top is what makes the synthetic signal *temporally structured* in a way the rest-of-the-literature single-Gaussian-mixture model is not.

## What the experiments actually measure

For each (band, noise type, noise level) triple:

1. Sample a token sequence.
2. Render it to a wavelet signal.
3. Add noise.
4. Run STFT (or template correlation, in the structured-noise regime).
5. Sweep detection thresholds.
6. Compute the ROC, AUC, and precision against the ground-truth token positions.

Then plot AUC vs. noise level per band per noise regime. This tells you, concretely: *at what SNR does gamma detection break? Does pink noise hurt delta more than white noise does? How much variance does the structured-noise regime add?* These are the kinds of numbers downstream BCI / clinical EEG work needs and rarely has in clean form.

## The headline

EEG bands stay detectable across a surprisingly wide noise range *when you model the signal correctly* — as bursts indexed by structured sequences, not as stationary mixtures. The point of the thesis is partly the numbers, but mostly the framework: a clean, reproducible benchmark for time-frequency detection on a generative model that respects the actual statistics of neural oscillations.
