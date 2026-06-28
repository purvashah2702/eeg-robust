"""Check events and epoch extraction details for subjects 6, 8, 9."""
import mne
import moabb
import numpy as np
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")

dataset = BNCI2014_001()

for subj in [1, 6]:  # compare a working subject (1) vs failing subject (6)
    print(f"\n{'='*50}\nSubject {subj}\n{'='*50}")
    data = dataset.get_data(subjects=[subj])
    subject_data = data[subj]

    sk = list(subject_data.keys())[0]
    rk = list(subject_data[sk].keys())[0]
    raw = subject_data[sk][rk]

    print(f"  Raw duration: {raw.times[-1]:.2f} sec, sfreq: {raw.info['sfreq']}")

    clean_raw = preprocess_raw(raw, apply_ica=False)
    print(f"  After preprocess - duration: {clean_raw.times[-1]:.2f} sec, sfreq: {clean_raw.info['sfreq']}")

    events, event_id = mne.events_from_annotations(clean_raw)
    print(f"  Events found: {len(events)}, event_id: {event_id}")
    print(f"  First 5 event sample indices: {events[:5, 0]}")
    print(f"  Last 5 event sample indices: {events[-5:, 0]}")
    print(f"  Max valid sample index (duration*sfreq): {clean_raw.times[-1] * clean_raw.info['sfreq']:.0f}")

    # Check if any event + tmax window goes past the recording
    tmax_samples = 4.0 * clean_raw.info['sfreq']
    last_valid_start = (clean_raw.times[-1] * clean_raw.info['sfreq']) - tmax_samples
    out_of_bounds = (events[:, 0] > last_valid_start).sum()
    print(f"  Events whose epoch window would exceed recording: {out_of_bounds} / {len(events)}")

print("\n✅ Debug complete!")
