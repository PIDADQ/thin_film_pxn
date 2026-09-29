# -*- coding: utf-8 -*-
"""Student-specific parameters and global experiment settings.

All personal parameters are derived from the student ID 2023270026:
    N           = last two digits        -> 26
    seed        = last six digits (int)  -> 270026
    design_seed = seed + 1               -> 270027
    lambda_target = 450 + 10 * (N mod 31) = 710 nm
"""
import numpy as np

STUDENT_ID = "2023270026"

# --- Student-specific parameters ------------------------------------------
N = int(STUDENT_ID[-2:])                 # 26
SEED = int(STUDENT_ID[-6:])              # 270026
DESIGN_SEED = SEED + 1                   # 270027
LAMBDA_TARGET = 450 + 10 * (N % 31)      # 710 nm

# --- Optical model (fixed by the assignment) ------------------------------
N_H = 2.30       # high-index layer
N_L = 1.45       # low-index layer
N_SUB = 1.52     # glass substrate
N_INC = 1.00     # air (incident medium)
LAYER_INDICES = [N_H, N_L, N_H, N_L]     # Air / H / L / H / L / Glass
D_MIN, D_MAX = 40.0, 180.0               # thickness range (nm)

WAVELENGTHS = np.arange(400, 801, 10, dtype=float)   # 41 points, 400-800 nm
N_WL = len(WAVELENGTHS)
WL_TARGET_IDX = int(round((LAMBDA_TARGET - 400.0) / 10.0))  # index of 710 nm

# --- Dataset --------------------------------------------------------------
N_TOTAL = 5000
N_TRAIN = 4000
N_VAL = 500
N_TEST = 500
TRAIN_SIZES = [500, 1000, 2000, 4000]     # training-size experiment

# --- MLP surrogate model --------------------------------------------------
HIDDEN_SIZES = [128, 128, 64]
ACTIVATION = "relu"
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 1000
PATIENCE = 100          # early-stopping patience (monitor val loss)

# --- Screening ------------------------------------------------------------
N_CANDIDATES = 10000
TOP_K_MLP = 10
TOP_K_FINAL = 5
