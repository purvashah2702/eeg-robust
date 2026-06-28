"""Debug why subjects 6, 8, 9 fail in CSP."""
import mne
import moabb
import numpy as np
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw, make_epochs

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")

dataset = BNCI2014_001()

for subj in [6, 8, 9]:
    print(f"\n{'='*50}\nSubject {subj}\n{'='*50}")
    data = dataset.get_data(subjects=[subj])
    subject_data = data[subj]

    for sk in subject_data.keys():
        for rk in subject_data[sk].keys():
            raw = subject_data[sk][rk]
            clean_raw = preprocess_raw(raw, apply_ica=False)
            events, event_id = mne.events_from_annotations(clean_raw)
            epochs = make_epochs(clean_raw, events, event_id)
            X = epochs.get_data(copy=True).astype("float64")
            y = epochs.events[:, 2]

            print(f"\n  Session {sk}, run {rk}:")
            print(f"    X shape: {X.shape}, classes: {np.unique(y, return_counts=True)}")

            # Check for flat/zero-variance channels per trial
            channel_vars = X.var(axis=2)  # (trials, channels)
            zero_var_channels = (channel_vars < 1e-10).sum(axis=0)
            print(f"    Channels with near-zero variance in >0 trials: {(zero_var_channels > 0).sum()} / {X.shape[1]}")
            if (zero_var_channels > 0).any():
                bad_ch_idx = np.where(zero_var_channels > 0)[0]
                print(f"    Bad channel indices: {bad_ch_idx}, affecting trials: {zero_var_channels[bad_ch_idx]}")

            # Check overall stats
            print(f"    X min/max/mean/std: {X.min():.3f}/{X.max():.3f}/{X.mean():.3f}/{X.std():.3f}")
            print(f"    Any NaN: {np.isnan(X).any()}, Any Inf: {np.isinf(X).any()}")

print("\n✅ Debug complete!")
