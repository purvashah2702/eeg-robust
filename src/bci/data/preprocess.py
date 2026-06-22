"""MNE preprocessing pipeline: filtering, re-referencing, resampling, epoching, ICA."""
import mne
import numpy as np
from mne.preprocessing import ICA

mne.set_log_level("WARNING")

LOW_FREQ = 4.0
HIGH_FREQ = 40.0
NOTCH_FREQ = 50.0   # use 60.0 if you're in the US
RESAMPLE_SFREQ = 250.0


def preprocess_raw(raw: mne.io.BaseRaw, apply_ica: bool = True) -> mne.io.BaseRaw:
    """Apply bandpass, notch, re-ref, resample, and optional ICA cleaning."""
    raw = raw.copy()

    # 1. Bandpass filter — keep motor imagery frequencies
    raw.filter(l_freq=LOW_FREQ, h_freq=HIGH_FREQ, fir_design="firwin")

    # 2. Notch filter — remove power line noise
    raw.notch_filter(freqs=[NOTCH_FREQ])

    # 3. Re-reference — common average reference
    raw.set_eeg_reference("average", projection=False)

    # 4. Resample — standardize sampling rate across datasets
    raw.resample(RESAMPLE_SFREQ)

    # 5. ICA — remove eye/muscle artifacts
    if apply_ica:
        ica = ICA(n_components=15, random_state=42, max_iter="auto")
        ica.fit(raw)
        raw = ica.apply(raw)

    return raw


def make_epochs(raw: mne.io.BaseRaw, events, event_id, tmin=-0.5, tmax=4.0):
    """Cut continuous EEG into trial-aligned epochs."""
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=None, preload=True,
    )
    return epochs
