# -*- coding: utf-8 -*-
"""MLP-assisted thin-film screening.

10000 fresh candidate thickness vectors are generated with design_seed
(seed + 1).  The trained MLP predicts their 41-point spectra; candidates are
ranked by the MLP reflectance at lambda_target.  The top-10 are re-evaluated
with the exact TMM, and the final top-5 (by TMM reflectance) are reported.

Produces:
  - data/results_screen.npz  (all candidate thicknesses, MLP/TMM spectra)
  - data/top5.csv            (Top-5 table)
  - figures/fig7_design.png  (Top-1 MLP vs TMM spectra with lambda_target line)
"""
import os
import random

import numpy as np
import torch

import config
import tmm
import mlp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(config.DESIGN_SEED)
np.random.seed(config.DESIGN_SEED)

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    # 1) generate 10000 fresh candidates with design_seed
    rng = np.random.default_rng(config.DESIGN_SEED)
    X_cand = rng.uniform(config.D_MIN, config.D_MAX,
                         size=(config.N_CANDIDATES, 4))

    # 2) load trained MLP and predict
    ckpt = torch.load(os.path.join(BASE, "models", "mlp.pt"),
                      map_location="cpu")
    scaler = mlp.ThicknessScaler(ckpt["d_min"], ckpt["d_max"])
    model = mlp.MLPSurrogate(in_dim=4, hidden_sizes=ckpt["hidden_sizes"],
                             out_dim=config.N_WL, activation=config.ACTIVATION)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    Xs_cand = scaler.transform(X_cand)
    with torch.no_grad():
        Y_cand_mlp = model(torch.tensor(Xs_cand, dtype=torch.float32)).numpy()

    # 3) rank by MLP reflectance at lambda_target
    idx = config.WL_TARGET_IDX
    R_mlp_target = Y_cand_mlp[:, idx]
    order = np.argsort(-R_mlp_target)   # descending
    top10_idx = order[:config.TOP_K_MLP]

    # 4) re-evaluate top-10 with exact TMM
    top10_X = X_cand[top10_idx]
    R_tmm_target = np.array([
        tmm.tmm_reflectance(d, config.LAYER_INDICES, config.N_SUB,
                            config.N_INC, config.WAVELENGTHS)[idx]
        for d in top10_X
    ])

    # 5) final top-5 by TMM reflectance at lambda_target
    top10_tmm_order = np.argsort(-R_tmm_target)
    top5 = top10_tmm_order[:config.TOP_K_FINAL]

    rows = []
    print(f"lambda_target = {config.LAMBDA_TARGET} nm (idx {idx})")
    print(f"{'Rank':<5}{'d1':>8}{'d2':>8}{'d3':>8}{'d4':>8}"
          f"{'MLP R':>10}{'TMM R':>10}")
    for rank, j in enumerate(top5, start=1):
        d = top10_X[j]
        r_mlp = R_mlp_target[top10_idx[j]]
        r_tmm = R_tmm_target[j]
        rows.append([rank, *d, r_mlp, r_tmm])
        print(f"{rank:<5}{d[0]:>8.2f}{d[1]:>8.2f}{d[2]:>8.2f}{d[3]:>8.2f}"
              f"{r_mlp:>10.4f}{r_tmm:>10.4f}")

    header = ["Rank", "d1/nm", "d2/nm", "d3/nm", "d4/nm",
              "MLP Rtarget", "TMM Rtarget"]
    np.savetxt(os.path.join(BASE, "data", "top5.csv"), rows,
               delimiter=",", fmt=["%d", "%.2f", "%.2f", "%.2f", "%.2f",
                                   "%.4f", "%.4f"],
               header=",".join(header), comments="")

    np.savez(os.path.join(BASE, "data", "results_screen.npz"),
             X_cand=X_cand, Y_cand_mlp=Y_cand_mlp,
             top10_idx=top10_idx, R_tmm_target=R_tmm_target)

    # 6) Figure 7: Top-1 candidate MLP vs TMM spectra
    j_best = top5[0]
    d_best = top10_X[j_best]
    spec_tmm = tmm.tmm_reflectance(d_best, config.LAYER_INDICES,
                                   config.N_SUB, config.N_INC,
                                   config.WAVELENGTHS)
    spec_mlp = Y_cand_mlp[top10_idx[j_best]]

    plt.figure(figsize=(6.5, 4))
    plt.plot(config.WAVELENGTHS, spec_tmm, "k-", lw=1.8, label="TMM")
    plt.plot(config.WAVELENGTHS, spec_mlp, "r--", lw=1.5, label="MLP")
    plt.axvline(config.LAMBDA_TARGET, color="b", ls=":", lw=1.2,
                label=f"$\\lambda_{{target}}$={config.LAMBDA_TARGET} nm")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title(f"Top-1 design: d=[{d_best[0]:.1f}, {d_best[1]:.1f}, "
              f"{d_best[2]:.1f}, {d_best[3]:.1f}] nm")
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_design.png"), dpi=300)
    plt.close()

    # also save the top-1 full TMM spectrum for the failure/design reuse
    print(f"\nTop-1 design saved: d = {d_best} nm, TMM R({config.LAMBDA_TARGET} nm) = "
          f"{R_tmm_target[j_best]:.4f}")


if __name__ == "__main__":
    main()
