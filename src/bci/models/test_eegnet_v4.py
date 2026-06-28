"""EEGNet v4: 9 subjects, augmentation, longer patience, LR scheduler."""
import mne
import moabb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw, make_epochs
from bci.data.splits import train_val_test_split
from bci.models.eegnet_model import build_eegnet
from bci.eval.metrics import evaluate_classifier
from bci.utils.seed import set_seed

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")
set_seed(42)


class AugmentedEEGDataset(Dataset):
    def __init__(self, X, y, augment=False, noise_std=0.03, max_shift=5):
        self.X = X
        self.y = y
        self.augment = augment
        self.noise_std = noise_std
        self.max_shift = max_shift

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        y = self.y[idx]
        if self.augment:
            x = x + torch.randn_like(x) * self.noise_std
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            x = torch.roll(x, shifts=shift, dims=-1)
        return x, y


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, lr=2e-4):
        super().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (torch.argmax(logits, dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 9]

print(f"Loading and preprocessing subjects {SUBJECTS}...")
dataset = BNCI2014_001()
all_X, all_y = [], []

for subj in SUBJECTS:
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

X = torch.tensor(np.concatenate(all_X, axis=0), dtype=torch.float32)
y_raw = np.concatenate(all_y, axis=0)

X = (X - X.mean(dim=(0, 2), keepdim=True)) / (X.std(dim=(0, 2), keepdim=True) + 1e-6)

unique_labels = sorted(set(y_raw))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
y = torch.tensor([label_to_idx[label] for label in y_raw], dtype=torch.long)

print(f"\nTotal pooled data: X shape {X.shape}, y shape {y.shape}")

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X.numpy(), y.numpy(), val_size=0.15, test_size=0.15
)
X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
n_classes = len(unique_labels)
n_channels = X_train.shape[1]
n_times = X_train.shape[2]

print(f"\nBuilding EEGNet: {n_channels} channels, {n_classes} classes, {n_times} timepoints")
model = build_eegnet(n_channels, n_classes, n_times)
lightning_model = LightningWrapper(model, lr=2e-4)

train_dataset = AugmentedEEGDataset(X_train, y_train, augment=True)
val_dataset = AugmentedEEGDataset(X_val, y_val, augment=False)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

early_stop = EarlyStopping(monitor="val_loss", patience=40, mode="min")

print("\nTraining EEGNet v4 (patience=40, LR scheduler, augmentation)...")
trainer = pl.Trainer(max_epochs=200, enable_progress_bar=True, logger=False,
                      enable_checkpointing=False, accelerator="auto",
                      gradient_clip_val=1.0, callbacks=[early_stop])
trainer.fit(lightning_model, train_loader, val_loader)

print(f"\nTraining stopped at epoch {trainer.current_epoch}")

print("\nEvaluating on TEST set...")
lightning_model.eval()
with torch.no_grad():
    logits = lightning_model(X_test)
    y_pred = torch.argmax(logits, dim=1).numpy()
    y_proba = torch.softmax(logits, dim=1).numpy()

results = evaluate_classifier(y_test.numpy(), y_pred, y_proba, n_classes=n_classes)
print("\n--- EEGNet v4 Results (patience=40, LR scheduler) ---")
for k, v in results.items():
    print(f"  {k}: {v}")

print("\n✅ EEGNet v4 works end-to-end!")
