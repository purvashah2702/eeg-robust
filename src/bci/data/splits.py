"""Within-subject and cross-subject train/test splits."""
import numpy as np
from sklearn.model_selection import train_test_split


def within_subject_split(X, y, test_size=0.2, seed=42):
    """Split one subject's trials into train/test (same subject in both)."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def cross_subject_split(subject_data: dict, test_subjects: list):
    """
    Train on some subjects, test on entirely different subjects.
    subject_data: {subject_id: (X, y)}
    test_subjects: list of subject IDs held out for testing
    """
    X_train, y_train, X_test, y_test = [], [], [], []

    for subj_id, (X, y) in subject_data.items():
        if subj_id in test_subjects:
            X_test.append(X)
            y_test.append(y)
        else:
            X_train.append(X)
            y_train.append(y)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return X_train, y_train, X_test, y_test
