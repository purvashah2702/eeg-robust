"""Deeper debug: check raw signal (before epoching) for subjects 6, 8, 9."""
import mne
import moabb
import numpy as np
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw

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
            raw_data = raw.get_data()
            print(f"\n  Session {sk}, run {rk}:")
            print(f"    RAW (before any processing) shape: {raw_data.shape}")
            print(f"    RAW min/max/mean/std: {raw_data.min():.6f}/{raw_data.max():.6f}/{raw_data.mean():.6f}/{raw_data.std():.6f}")

            clean_raw = preprocess_raw(raw, apply_ica=False)
            clean_data = clean_raw.get_data()
            print(f"    AFTER preprocess (no ICA) min/max/mean/std: {clean_data.min():.6f}/{clean_data.max():.6f}/{clean_data.mean():.6f}/{clean_data.std():.6f}")

print("\n✅ Debug complete!")
