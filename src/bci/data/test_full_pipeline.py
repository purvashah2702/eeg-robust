"""End-to-end test: load -> preprocess -> remap labels -> split."""
import mne
import moabb
import numpy as np
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw, make_epochs
from bci.data.label_map import remap_event_id
from bci.data.splits import within_subject_split, cross_subject_split

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")

dataset = BNCI2014_001()
subject_data_dict = {}

for subj in [1, 2]:
    print(f"\nProcessing subject {subj}...")
    data = dataset.get_data(subjects=[subj])
    subject_data = data[subj]
    session_key = list(subject_data.keys())[0]
    run_key = list(subject_data[session_key].keys())[0]
    raw = subject_data[session_key][run_key]

    clean_raw = preprocess_raw(raw, apply_ica=True)
    events, event_id = mne.events_from_annotations(clean_raw)

    common_event_id = remap_event_id(event_id, "bnci2014_001")
    print(f"  Remapped labels: {common_event_id}")

    epochs = make_epochs(clean_raw, events, event_id, tmin=-0.5, tmax=4.0)
    X = epochs.get_data(copy=True)
    y = epochs.events[:, 2]

    subject_data_dict[subj] = (X, y)
    print(f"  Subject {subj}: X shape {X.shape}, y shape {y.shape}")

print("\n--- Within-subject split (subject 1) ---")
X1, y1 = subject_data_dict[1]
X_train, X_test, y_train, y_test = within_subject_split(X1, y1)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

print("\n--- Cross-subject split (train=1, test=2) ---")
X_train, y_train, X_test, y_test = cross_subject_split(subject_data_dict, test_subjects=[2])
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

print("\n✅ Full Week 2 pipeline works end-to-end!")
