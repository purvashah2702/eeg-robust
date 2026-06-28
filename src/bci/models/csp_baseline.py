"""Classical baseline: CSP (Common Spatial Patterns) + LDA/SVM."""
import numpy as np
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


def build_csp_lda(n_components: int = 6):
    """CSP feature extraction -> LDA classifier."""
    return Pipeline([
        ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True)),
        ("lda", LinearDiscriminantAnalysis()),
    ])


def build_csp_svm(n_components: int = 6):
    """CSP feature extraction -> SVM classifier."""
    return Pipeline([
        ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True)),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
    ])


def fit_predict(pipeline, X_train, y_train, X_test):
    """Fit pipeline and return predictions + probabilities."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    return y_pred, y_proba
