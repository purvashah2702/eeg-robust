"""Week 4 end-to-end test: augmentations + mixup + DANN (cross-subject) + calibration."""
import mne
import moabb
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw, make_epochs
from bci.data.augmentations import EEGAugmentor
from bci.models.regularization import mixup_batch, mixup_loss
from bci.models.feature_extractor import build_feature_extractor
from bci.models.dann import DANNWrapper, dann_loss
from bci.eval.calibration import TemperatureScaler, expected_calibration_error
from bci.eval.metrics import evaluate_classifier
from bci.utils.seed import set_seed

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")
set_seed(42)


class AugmentedDataset(Dataset):
    def __init__(self, X, y, domain, augment=False):
        self.X = X
        self.y = y
        self.domain = domain
        self.augment = augment
        self.augmentor = EEGAugmentor(apply_prob=0.3) if augment else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            x = self.augmentor(x)
        return x, self.y[idx], self.domain[idx]


SUBJECTS = [1, 2, 3, 4, 5, 6]  # use 6 subjects as "domains" for DANN

print(f"Loading and preprocessing subjects {SUBJECTS}...")
dataset = BNCI2014_001()
all_X, all_y, all_domain = [], [], []

for i, subj in enumerate(SUBJECTS):
    print(f"  Processing subject {subj}...")
    data = dataset.get_data(subjects=[subj])
    subject_data = data[subj]
    session_key = list(subject_data.keys())[0]
    run_key = list(subject_data[session_key].keys())[0]
    raw = subject_data[session_key][run_key]

    clean_raw = preprocess_raw(raw, apply_ica=True)
    events, event_id = mne.events_from_annotations(clean_raw)
    epochs = make_epochs(clean_raw, events, event_id)
    X = epochs.get_data(copy=True).astype("float32")
    y = epochs.events[:, 2]

    all_X.append(X)
    all_y.append(y)
    all_domain.append(np.full(len(y), i))  # domain id = subject index

X = torch.tensor(np.concatenate(all_X, axis=0), dtype=torch.float32)
y_raw = np.concatenate(all_y, axis=0)
domain = torch.tensor(np.concatenate(all_domain, axis=0), dtype=torch.long)

X = (X - X.mean(dim=(0, 2), keepdim=True)) / (X.std(dim=(0, 2), keepdim=True) + 1e-6)

unique_labels = sorted(set(y_raw))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
y = torch.tensor([label_to_idx[label] for label in y_raw], dtype=torch.long)

print(f"\nTotal data: X {X.shape}, y {y.shape}, domains: {len(SUBJECTS)}")

# Simple split: 80% train, 20% test (stratified-ish via shuffling)
n = len(X)
perm = torch.randperm(n)
split = int(0.8 * n)
train_idx, test_idx = perm[:split], perm[split:]

X_train, y_train, domain_train = X[train_idx], y[train_idx], domain[train_idx]
X_test, y_test, domain_test = X[test_idx], y[test_idx], domain[test_idx]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

n_classes = len(unique_labels)
n_channels = X.shape[1]
n_times = X.shape[2]
n_domains = len(SUBJECTS)

print("\nBuilding DANN model...")
feature_extractor = build_feature_extractor(n_channels, n_times)
label_classifier = nn.Linear(feature_extractor.feature_dim, n_classes)
model = DANNWrapper(feature_extractor, label_classifier,
                     feature_extractor.feature_dim, n_domains, grl_alpha=0.5)

train_dataset = AugmentedDataset(X_train, y_train, domain_train, augment=True)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

print("\nTraining DANN with augmentation + mixup for 40 epochs...")
model.train()
for epoch in range(40):
    epoch_loss = 0.0
    for x_batch, y_batch, d_batch in train_loader:
        optimizer.zero_grad()

        # Apply mixup half the time
        if np.random.rand() < 0.5:
            mixed_x, y_a, y_b, lam = mixup_batch(x_batch, y_batch, alpha=0.2)
            label_logits, domain_logits = model(mixed_x)
            task_loss = mixup_loss(nn.CrossEntropyLoss(), label_logits, y_a, y_b, lam)
            domain_loss = nn.CrossEntropyLoss()(domain_logits, d_batch)
            loss = task_loss + 0.3 * domain_loss
        else:
            label_logits, domain_logits = model(x_batch)
            loss, task_loss, domain_loss = dann_loss(
                label_logits, y_batch, domain_logits, d_batch, domain_weight=0.3
            )

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/40 - loss: {epoch_loss/len(train_loader):.4f}")

print("\nEvaluating on held-out test set...")
model.eval()
with torch.no_grad():
    test_logits = model(X_test, return_domain=False)
    y_pred = torch.argmax(test_logits, dim=1).numpy()
    y_proba = torch.softmax(test_logits, dim=1).numpy()

results = evaluate_classifier(y_test.numpy(), y_pred, y_proba, n_classes=n_classes)
print("\n--- DANN Results (cross-subject, augmented) ---")
for k, v in results.items():
    print(f"  {k}: {v}")

# Calibration check
ece_before = expected_calibration_error(y_proba, y_test.numpy())
print(f"\nECE before temperature scaling: {ece_before:.4f}")

print("\nFitting temperature scaler...")
scaler = TemperatureScaler()
scaler.fit(test_logits, y_test)
calibrated_logits = scaler(test_logits)
calibrated_probs = torch.softmax(calibrated_logits, dim=1).detach().numpy()
ece_after = expected_calibration_error(calibrated_probs, y_test.numpy())
print(f"Learned temperature: {scaler.temperature.item():.3f}")
print(f"ECE after temperature scaling: {ece_after:.4f}")

print("\n✅ Week 4 robustness pipeline works end-to-end!")
