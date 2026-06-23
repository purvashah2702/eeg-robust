"""Test EEGNet on real preprocessed EEG data using PyTorch Lightning."""
import mne
import moabb
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw, make_epochs
from bci.data.splits import within_subject_split
from bci.models.eegnet_model import build_eegnet
from bci.eval.metrics import evaluate_classifier
from bci.utils.seed import set_seed

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")
set_seed(42)


class EEGNetLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-3):
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


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
X = epochs.get_data(copy=True).astype("float32")
y = epochs.events[:, 2]

# Remap labels to 0-indexed contiguous integers (required for CrossEntropyLoss)
unique_labels = sorted(set(y))
label_to_idx = {label: i for i, label in enumerate(unique_labels)}
y = torch.tensor([label_to_idx[label] for label in y], dtype=torch.long)
X = torch.tensor(X, dtype=torch.float32)

X_train, X_test, y_train, y_test = within_subject_split(X.numpy(), y.numpy(), test_size=0.3)
X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
n_classes = len(unique_labels)
n_channels = X_train.shape[1]
n_times = X_train.shape[2]

print(f"\nBuilding EEGNet: {n_channels} channels, {n_classes} classes, {n_times} timepoints")
model = build_eegnet(n_channels, n_classes, n_times)
lightning_model = EEGNetLightning(model)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)

print("\nTraining EEGNet for 20 epochs...")
trainer = pl.Trainer(max_epochs=20, enable_progress_bar=True, logger=False,
                      enable_checkpointing=False, accelerator="auto")
trainer.fit(lightning_model, train_loader)

print("\nEvaluating on test set...")
lightning_model.eval()
with torch.no_grad():
    logits = lightning_model(X_test)
    y_pred = torch.argmax(logits, dim=1).numpy()
    y_proba = torch.softmax(logits, dim=1).numpy()

results = evaluate_classifier(y_test.numpy(), y_pred, y_proba, n_classes=n_classes)
print("\n--- EEGNet Results ---")
for k, v in results.items():
    print(f"  {k}: {v}")

print("\n✅ EEGNet works end-to-end!")
