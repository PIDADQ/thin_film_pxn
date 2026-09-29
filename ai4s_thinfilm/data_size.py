# -*- coding: utf-8 -*-
"""Training-size experiment: keep val/test fixed, vary training samples.

Four MLPs are trained on the first 500 / 1000 / 2000 / 4000 samples of the
training pool with identical hyperparameters.  The validation (500) and test
(500) sets never change.  The test RMSE/MSE of each model is recorded.

Produces figures/fig6_datasize.png and data/results_datasize.npz.
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
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    data = np.load(os.path.join(BASE, "data", "dataset.npz"))
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_val, Y_val = data["X_val"], data["Y_val"]
    X_test, Y_test = data["X_test"], data["Y_test"]

    scaler = mlp.ThicknessScaler(config.D_MIN, config.D_MAX).fit(X_train)
    Xs_val = scaler.transform(X_val)
    Xs_test = scaler.transform(X_test)

    test_mse_list, test_rmse_list = [], []

    for n in config.TRAIN_SIZES:
        print(f"--- training on {n} samples ---", flush=True)
        Xs_sub = scaler.transform(X_train[:n])
        Y_sub = Y_train[:n]

        torch.manual_seed(config.SEED)   # fixed weight initialization
        model = mlp.MLPSurrogate(
            in_dim=4, hidden_sizes=config.HIDDEN_SIZES, out_dim=config.N_WL,
            activation=config.ACTIVATION,
        )
        model, _ = mlp.train_model(
            model, Xs_sub, Y_sub, Xs_val, Y_val,
            lr=config.LEARNING_RATE, batch_size=config.BATCH_SIZE,
            max_epochs=config.MAX_EPOCHS, patience=config.PATIENCE,
            seed=config.SEED, verbose=False,
        )

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xs_test, dtype=torch.float32)).numpy()
        mse = float(np.mean((pred - Y_test) ** 2))
        rmse = float(np.sqrt(mse))
        test_mse_list.append(mse)
        test_rmse_list.append(rmse)
        print(f"  n={n:5d}  test MSE={mse:.6e}  RMSE={rmse:.6e}", flush=True)

    np.savez(os.path.join(BASE, "data", "results_datasize.npz"),
             train_sizes=np.array(config.TRAIN_SIZES),
             test_mse=np.array(test_mse_list),
             test_rmse=np.array(test_rmse_list))

    # ---- Figure 6: training size vs test error ---------------------------
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(config.TRAIN_SIZES, test_rmse_list, "o-", color="tab:red",
             lw=1.5, markersize=7, label="RMSE")
    ax1.set_xlabel("Number of training samples")
    ax1.set_ylabel("Test RMSE", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xticks(config.TRAIN_SIZES)

    ax2 = ax1.twinx()
    ax2.plot(config.TRAIN_SIZES, test_mse_list, "s--", color="tab:blue",
             lw=1.5, markersize=6, label="MSE")
    ax2.set_ylabel("Test MSE", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_yscale("log")

    ax1.set_title("Effect of training-set size on test error")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_datasize.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
