# -*- coding: utf-8 -*-
"""Generate the thickness->reflectance dataset with TMM and split it.

5000 thickness vectors are sampled with ``seed``, each of the four layers
uniform in [40, 180] nm.  For every vector the TMM reflectance at 41
wavelengths (400-800 nm, step 10 nm) is computed.  The 5000 samples are then
split *sequentially* (no shuffling) into 4000 train / 500 validation / 500
test, and this split is fixed for all subsequent experiments.
"""
import os
import random

import numpy as np

import config
import tmm

random.seed(config.SEED)
np.random.seed(config.SEED)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def generate_dataset(n_total, seed):
    """Sample thickness vectors and compute their TMM spectra."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(config.D_MIN, config.D_MAX, size=(n_total, 4))
    Y = np.empty((n_total, config.N_WL), dtype=float)
    for i in range(n_total):
        Y[i] = tmm.tmm_reflectance(
            X[i], config.LAYER_INDICES, config.N_SUB,
            config.N_INC, config.WAVELENGTHS,
        )
    return X, Y


def main():
    X, Y = generate_dataset(config.N_TOTAL, config.SEED)

    # Sequential, fixed split: 4000 / 500 / 500 (no shuffle).
    X_train, X_val, X_test = X[:config.N_TRAIN], X[config.N_TRAIN:config.N_TRAIN + config.N_VAL], X[config.N_TRAIN + config.N_VAL:]
    Y_train, Y_val, Y_test = Y[:config.N_TRAIN], Y[config.N_TRAIN:config.N_TRAIN + config.N_VAL], Y[config.N_TRAIN + config.N_VAL:]

    np.savez_compressed(
        os.path.join(DATA_DIR, "dataset.npz"),
        X=X, Y=Y,
        X_train=X_train, Y_train=Y_train,
        X_val=X_val, Y_val=Y_val,
        X_test=X_test, Y_test=Y_test,
        wavelengths=config.WAVELENGTHS,
    )
    print(f"Dataset saved to data/dataset.npz")
    print(f"  X: {X.shape}, Y: {Y.shape}")
    print(f"  train: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}")
    print(f"  thickness range: [{X.min():.2f}, {X.max():.2f}] nm")
    print(f"  reflectance range: [{Y.min():.4f}, {Y.max():.4f}]")


if __name__ == "__main__":
    main()
