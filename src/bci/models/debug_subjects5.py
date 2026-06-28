"""Test the EXACT load_subject function used in CSP script, on subject 6."""
import mne
import moabb
import numpy as np
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw, make_epochs

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")

dataset = BNCI2014_001()
subj = 6

data = dataset.get_data(subjects=[subj])
subject_data = data[subj]

all_X, all_y = [], []
for sk in subject_data.keys():
    for rk in subject_data[sk].keys():
        raw = subject_data[sk][rk]
        clean_raw = preprocess_raw(raw, apply_ica=False)
        events, event_id = mne.events_from_annotations(clean_raw)
        epochs = make_epochs(clean_raw, events, event_id)
        X = epochs.get_data(copy=True).astype("float64")
        y = epochs.events[:, 2]

        print(f"Session {sk}, run {rk}: X shape {X.shape}, std={X.std():.6f}, "
              f"id(raw)={id(raw)}, id(clean_raw)={id(clean_raw)}")

        all_X.append(X)
        all_y.append(y)

X_final = np.concatenate(all_X, axis=0)
y_final = np.concatenate(all_y, axis=0)
print(f"\nFINAL concatenated X shape: {X_final.shape}, std: {X_final.std():.6f}")
print(f"Per-array stds before concat: {[x.std() for x in all_X]}")

print("\n✅ Debug complete!")
