# -*- coding: utf-8 -*-
"""MLP surrogate model: thickness -> reflectance spectrum.

Input:  4 thicknesses (min-max normalized to [0,1]).
Output: 41 reflectance values (linear output, no sigmoid).
"""
import numpy as np
import torch
import torch.nn as nn


class MLPSurrogate(nn.Module):
    def __init__(self, in_dim=4, hidden_sizes=(128, 128, 64), out_dim=41,
                 activation="relu"):
        super().__init__()
        layers = []
        dims = [in_dim] + list(hidden_sizes) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:          # hidden layers get activation
                layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ThicknessScaler:
    """Min-max normalization of thickness to [0, 1]."""

    def __init__(self, d_min, d_max):
        self.d_min = float(d_min)
        self.d_max = float(d_max)

    def fit(self, X):
        self.d_min = float(X.min())
        self.d_max = float(X.max())
        return self

    def transform(self, X):
        return (X - self.d_min) / (self.d_max - self.d_min)

    def inverse(self, Xs):
        return Xs * (self.d_max - self.d_min) + self.d_min


def train_model(model, X_train, Y_train, X_val, Y_val,
                lr, batch_size, max_epochs, patience, seed, device="cpu",
                verbose=True):
    """Train the MLP with Adam + MSE and early stopping on the val loss.

    Returns (model, history_dict) where model is the best checkpoint (lowest
    validation loss) and history contains per-epoch train/val MSE.
    """
    torch.set_num_threads(1)   # deterministic + faster for this small model
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    Yv = torch.tensor(Y_val, dtype=torch.float32, device=device)

    n = Xt.shape[0]
    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss, n_batch = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xt[idx], Yt[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * idx.shape[0]
            n_batch += idx.shape[0]
        train_mse = total_loss / n

        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(model(Xv), Yv).item()

        history["train"].append(train_mse)
        history["val"].append(val_mse)

        if verbose and ((epoch + 1) % 200 == 0 or epoch == 0):
            print(f"  epoch {epoch + 1:4d}: train={train_mse:.6e} val={val_mse:.6e}",
                  flush=True)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, history
