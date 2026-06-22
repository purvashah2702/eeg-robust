"""Quick test: run preprocessing on one real subject."""
import mne
import moabb
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw, make_epochs

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")

print("Loading subject 1 from BNCI2014_001...")
dataset = BNCI2014_001()
data = dataset.get_data(subjects=[1])

# Navigate MOABB's nested dict: subject -> session -> run -> raw
subject_data = data[1]
session_key = list(subject_data.keys())[0]
run_key = list(subject_data[session_key].keys())[0]
raw = subject_data[session_key][run_key]

print(f"Raw data loaded: {raw.info['sfreq']} Hz, {len(raw.ch_names)} channels")
print(f"Duration: {raw.times[-1]:.1f} seconds")

print("\nRunning preprocessing pipeline...")
clean_raw = preprocess_raw(raw, apply_ica=True)

print(f"\nAfter preprocessing:")
print(f"  Sampling rate: {clean_raw.info['sfreq']} Hz")
print(f"  Channels: {len(clean_raw.ch_names)}")
print(f"  Duration: {clean_raw.times[-1]:.1f} seconds")

print("\nExtracting events...")
events, event_id = mne.events_from_annotations(clean_raw)
print(f"Found {len(events)} events: {event_id}")

print("\nCreating epochs...")
epochs = make_epochs(clean_raw, events, event_id)
print(f"Created {len(epochs)} epochs of shape {epochs.get_data().shape}")

print("\n✅ Preprocessing pipeline works end-to-end!")
