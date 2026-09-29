# -*- coding: utf-8 -*-
"""Locate the worst MLP prediction on the test set and plot it.

Produces figures/fig8_failure.png and reports the sample index and MSE.
"""
import os

import numpy as np
import torch

import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    data = np.load(os.path.join(BASE, "data", "dataset.npz"))
    res = np.load(os.path.join(BASE, "data", "results_mlp.npz"))
    X_test, Y_test = data["X_test"], data["Y_test"]
    wavelengths = data["wavelengths"]
    Y_pred = res["Y_test_pred"]

    per_sample_mse = np.mean((Y_pred - Y_test) ** 2, axis=1)
    worst = int(np.argmax(per_sample_mse))
    print(f"Worst test sample index = {worst}")
    print(f"  thickness d = {X_test[worst]}")
    print(f"  per-sample MSE = {per_sample_mse[worst]:.6e}")
    print(f"  mean test MSE   = {per_sample_mse.mean():.6e}")

    plt.figure(figsize=(6.5, 4))
    plt.plot(wavelengths, Y_test[worst], "k-", lw=1.8, label="TMM")
    plt.plot(wavelengths, Y_pred[worst], "r--", lw=1.5, label="MLP")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(f"Failure case: test sample #{worst}, "
              f"d=[{X_test[worst][0]:.1f}, {X_test[worst][1]:.1f}, "
              f"{X_test[worst][2]:.1f}, {X_test[worst][3]:.1f}] nm")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig8_failure.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
