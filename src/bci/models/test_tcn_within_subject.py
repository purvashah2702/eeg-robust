"""TCN trained within-subject (standard protocol) for subjects 1-9."""
import mne
import moabb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from moabb.datasets import BNCI2014_001
from bci.data.preprocess import preprocess_raw, make_epochs
from bci.data.splits import train_val_test_split
from bci.models.tcn_model import build_tcn
from bci.eval.metrics import evaluate_classifier
from bci.utils.seed import set_seed

mne.set_log_level("WARNING")
moabb.set_log_level("WARNING")
set_seed(42)


class LightningWrapper(pl.LightningModule):
    def __init__(self, model, lr=1e-4):
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
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)


def train_one_subject(subj, dataset):
    print(f"\n{'='*50}\nSubject {subj}\n{'='*50}")
    data = dataset.get_data(subjects=[subj])
    subject_data = data[subj]
    session_keys = list(subject_data.keys())

    all_X, all_y = [], []
    for sk in session_keys:
        for rk in subject_data[sk].keys():
            raw = subject_data[sk][rk]
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

    print(f"Subject {subj}: X shape {X.shape}, {len(unique_labels)} classes")

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X.numpy(), y.numpy(), val_size=0.15, test_size=0.15
    )
    X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
    X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
    X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

    n_classes = len(unique_labels)
    n_channels = X_train.shape[1]
    n_times = X_train.shape[2]

    model = build_tcn(n_channels, n_classes, n_times)
    lightning_model = LightningWrapper(model, lr=1e-4)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)

    early_stop = EarlyStopping(monitor="val_loss", patience=30, mode="min")
    trainer = pl.Trainer(max_epochs=300, enable_progress_bar=False, logger=False,
                          enable_checkpointing=False, accelerator="auto",
                          gradient_clip_val=1.0, callbacks=[early_stop],
                          enable_model_summary=False)
    trainer.fit(lightning_model, train_loader, val_loader)

    lightning_model.eval()
    with torch.no_grad():
        logits = lightning_model(X_test)
        y_pred = torch.argmax(logits, dim=1).numpy()
        y_proba = torch.softmax(logits, dim=1).numpy()

    results = evaluate_classifier(y_test.numpy(), y_pred, y_proba, n_classes=n_classes)
    print(f"Subject {subj} -- stopped at epoch {trainer.current_epoch} -- "
          f"accuracy: {results['accuracy']:.3f}, macro_f1: {results['macro_f1']:.3f}")
    return results["accuracy"]


dataset = BNCI2014_001()
accuracies = []
for subj in range(1, 10):
    acc = train_one_subject(subj, dataset)
    accuracies.append(acc)

print(f"\n{'='*50}")
print(f"MEAN accuracy across 9 subjects: {np.mean(accuracies):.3f} (+/- {np.std(accuracies):.3f})")
print("✅ Within-subject TCN training works end-to-end!")
