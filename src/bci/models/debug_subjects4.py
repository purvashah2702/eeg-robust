"""Check the TEST session specifically for subjects 6, 8, 9."""
import mne
import moabb
import numpy as np
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw, make_epochs

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")

dataset = BNCI2014_001()

for subj in [1, 6, 8, 9]:
    print(f"\n{'='*50}\nSubject {subj}\n{'='*50}")
    data = dataset.get_data(subjects=[subj])
    subject_data = data[subj]

    session_keys = list(subject_data.keys())
    test_key = [k for k in session_keys if "test" in k.lower()][0]
    rk = list(subject_data[test_key].keys())[0]
    raw = subject_data[test_key][rk]

    print(f"  TEST session key: {test_key}, run key: {rk}")

    clean_raw = preprocess_raw(raw, apply_ica=False)
    events, event_id = mne.events_from_annotations(clean_raw)
    print(f"  Events: {len(events)}, event_id: {event_id}")

    epochs = make_epochs(clean_raw, events, event_id)
    X = epochs.get_data(copy=True)
    print(f"  Epoch X shape: {X.shape}")
    print(f"  Epoch X min/max/mean/std: {X.min():.6f}/{X.max():.6f}/{X.mean():.6f}/{X.std():.6f}")

    first_event_sample = events[0, 0]
    window = clean_raw.get_data(start=first_event_sample - 125, stop=first_event_sample + 1000)
    print(f"  Raw signal around first event: min/max/std: {window.min():.6f}/{window.max():.6f}/{window.std():.6f}")

print("\n✅ Debug complete!")
