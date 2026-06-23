"""Test CSP baseline on real preprocessed EEG data."""
import mne
import moabb
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw, make_epochs
from bci.data.splits import within_subject_split
from bci.models.csp_baseline import build_csp_lda, build_csp_svm, fit_predict
from bci.eval.metrics import evaluate_classifier

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")

print("Loading and preprocessing subject 1...")
dataset = BNCI2014_001()
data = dataset.get_data(subjects=[1])
subject_data = data[1]
session_key = list(subject_data.keys())[0]
run_key = list(subject_data[session_key].keys())[0]
raw = subject_data[session_key][run_key]

clean_raw = preprocess_raw(raw, apply_ica=True)
events, event_id = mne.events_from_annotations(clean_raw)
epochs = make_epochs(clean_raw, events, event_id)
X = epochs.get_data(copy=True).astype("float64")
y = epochs.events[:, 2]

print(f"X shape: {X.shape}, y shape: {y.shape}, classes: {set(y)}")

X_train, X_test, y_train, y_test = within_subject_split(X, y, test_size=0.3)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

print("\n--- CSP + LDA ---")
pipeline = build_csp_lda(n_components=6)
y_pred, y_proba = fit_predict(pipeline, X_train, y_train, X_test)
results = evaluate_classifier(y_test, y_pred, y_proba, n_classes=len(set(y)))
for k, v in results.items():
    print(f"  {k}: {v}")

print("\n--- CSP + SVM ---")
pipeline = build_csp_svm(n_components=6)
y_pred, y_proba = fit_predict(pipeline, X_train, y_train, X_test)
results = evaluate_classifier(y_test, y_pred, y_proba, n_classes=len(set(y)))
for k, v in results.items():
    print(f"  {k}: {v}")

print("\n✅ CSP baseline works end-to-end!")
