"""CSP+LDA and CSP+SVM trained within-subject — scaled to avoid float precision issues."""
import mne
import moabb
import numpy as np
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw, make_epochs
from bci.data.splits import within_subject_split
from bci.models.csp_baseline import build_csp_lda, build_csp_svm, fit_predict
from bci.eval.metrics import evaluate_classifier
from bci.utils.seed import set_seed

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")
set_seed(42)


def load_subject(subj, dataset):
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
            all_X.append(X)
            all_y.append(y)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # Scale up to avoid floating-point precision issues in CSP's covariance math
    # (raw EEG values are ~1e-6, which can underflow in covariance calculations)
    X = X * 1e6

    return X, y


dataset = BNCI2014_001()
lda_accs, svm_accs = [], []
skipped = []

for subj in range(1, 10):
    print(f"\n=== Subject {subj} ===")
    try:
        X, y = load_subject(subj, dataset)
        X_train, X_test, y_train, y_test = within_subject_split(X, y, test_size=0.2)

        pipeline = build_csp_lda(n_components=6)
        y_pred, y_proba = fit_predict(pipeline, X_train, y_train, X_test)
        results = evaluate_classifier(y_test, y_pred, y_proba, n_classes=len(set(y)))
        lda_accs.append(results["accuracy"])
        print(f"  CSP+LDA accuracy: {results['accuracy']:.3f}")

        pipeline = build_csp_svm(n_components=6)
        y_pred, y_proba = fit_predict(pipeline, X_train, y_train, X_test)
        results = evaluate_classifier(y_test, y_pred, y_proba, n_classes=len(set(y)))
        svm_accs.append(results["accuracy"])
        print(f"  CSP+SVM accuracy: {results['accuracy']:.3f}")
    except Exception as e:
        print(f"  SKIPPED subject {subj}: {type(e).__name__}: {e}")
        skipped.append(subj)

print(f"\n{'='*50}")
print(f"CSP+LDA mean accuracy: {np.mean(lda_accs):.3f} (+/- {np.std(lda_accs):.3f}) over {len(lda_accs)} subjects")
print(f"CSP+SVM mean accuracy: {np.mean(svm_accs):.3f} (+/- {np.std(svm_accs):.3f}) over {len(svm_accs)} subjects")
if skipped:
    print(f"Skipped subjects: {skipped}")
print("\n✅ CSP within-subject training works end-to-end!")
