"""Download all EEG datasets via MOABB."""
import time
import moabb
from moabb.datasets import BNCI2014_001, BNCI2014_002, PhysionetMI
import mne

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")

DATASETS = {
    "bnci2014_001": BNCI2014_001(),
    "bnci2014_002": BNCI2014_002(),
    "physionet":    PhysionetMI(),
}

def download_one(name, dataset, subjects, retries=3):
    dataset.subject_list = subjects
    for attempt in range(1, retries + 1):
        try:
            print(f"\n[{name}] attempt {attempt}/{retries} — subjects {subjects}")
            dataset.download()
            print(f"[{name}] done.")
            return True
        except Exception as e:
            print(f"[{name}] failed: {e}")
            if attempt < retries:
                print("Retrying in 10s...")
                time.sleep(10)
    print(f"[{name}] FAILED after {retries} attempts — skipping for now.")
    return False

def download_all(subjects=None):
    subs = subjects or list(range(1, 6))
    results = {}
    for name, dataset in DATASETS.items():
        results[name] = download_one(name, dataset, subs)
    print("\n--- Summary ---")
    for name, ok in results.items():
        print(f"{name}: {'OK' if ok else 'FAILED'}")

if __name__ == "__main__":
    download_all()
