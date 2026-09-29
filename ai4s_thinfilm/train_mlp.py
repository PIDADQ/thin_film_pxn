# -*- coding: utf-8 -*-
"""Train the MLP surrogate model on the full 4000-sample training set.

Produces:
  - models/mlp.pt            trained weights + scaler parameters
  - figures/fig4_loss.png    training/validation loss curves
  - figures/fig5_predict.png TMM vs MLP spectra on 3 representative test samples
  - test-set RMSE/MSE printed to stdout and saved to results.npz
"""
import os
import random

import numpy as np
import torch

import config
import mlp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
MODEL_DIR = os.path.join(BASE, "models")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    data = np.load(os.path.join(BASE, "data", "dataset.npz"))
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_val, Y_val = data["X_val"], data["Y_val"]
    X_test, Y_test = data["X_test"], data["Y_test"]
    wavelengths = data["wavelengths"]

    # Normalize thickness to [0, 1] using the *training* range.
    scaler = mlp.ThicknessScaler(config.D_MIN, config.D_MAX).fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_val = scaler.transform(X_val)
    Xs_test = scaler.transform(X_test)

    torch.manual_seed(config.SEED)   # fixed weight initialization
    model = mlp.MLPSurrogate(
        in_dim=4, hidden_sizes=config.HIDDEN_SIZES, out_dim=config.N_WL,
        activation=config.ACTIVATION,
    )
    model, history = mlp.train_model(
        model, Xs_train, Y_train, Xs_val, Y_val,
        lr=config.LEARNING_RATE, batch_size=config.BATCH_SIZE,
        max_epochs=config.MAX_EPOCHS, patience=config.PATIENCE,
        seed=config.SEED,
    )

    torch.save({
        "state_dict": model.state_dict(),
        "hidden_sizes": config.HIDDEN_SIZES,
        "d_min": scaler.d_min, "d_max": scaler.d_max,
    }, os.path.join(MODEL_DIR, "mlp.pt"))

    # ---- test metrics ----------------------------------------------------
    model.eval()
    with torch.no_grad():
        Y_test_pred = model(torch.tensor(Xs_test, dtype=torch.float32)).numpy()
    mse = float(np.mean((Y_test_pred - Y_test) ** 2))
    rmse = float(np.sqrt(mse))
    print(f"Best epoch = {len(history['val'])}")
    print(f"Final val MSE = {history['val'][-1]:.6e}")
    print(f"Test MSE  = {mse:.6e}")
    print(f"Test RMSE = {rmse:.6e}")

    np.savez(os.path.join(BASE, "data", "results_mlp.npz"),
             Y_test_pred=Y_test_pred, test_mse=mse, test_rmse=rmse,
             history_train=np.array(history["train"]),
             history_val=np.array(history["val"]))

    # ---- Figure 4: loss curves ------------------------------------------
    plt.figure(figsize=(6, 4))
    epochs = np.arange(1, len(history["train"]) + 1)
    plt.plot(epochs, history["train"], label="Training loss", lw=1.5)
    plt.plot(epochs, history["val"], label="Validation loss", lw=1.5)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.title("Training and validation losses")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_loss.png"), dpi=300)
    plt.close()

    # ---- Figure 5: TMM vs MLP on 3 test samples -------------------------
    # Pick samples that span the reflectance range for a representative view.
    test_idx = [0, len(X_test) // 2, len(X_test) - 1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, i in zip(axes, test_idx):
        ax.plot(wavelengths, Y_test[i], "k-", lw=1.5, label="TMM")
        ax.plot(wavelengths, Y_test_pred[i], "r--", lw=1.5, label="MLP")
        ax.set_xlabel("Wavelength (nm)")
        if i == 0:
            ax.set_ylabel("Reflectance")
        ax.set_title(f"Test sample #{i}")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("TMM-calculated vs MLP-predicted reflectance spectra", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_predict.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
