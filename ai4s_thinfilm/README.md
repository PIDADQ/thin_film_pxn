# AI4S Thin-Film Assignment: MLP-Based Spectral Prediction and Assisted Design

Research article code for the course assignment *《薄膜技术》AI4S大作业*:
**基于MLP的多层介质膜光谱预测与辅助设计**
(MLP-Based Spectral Prediction and Data-Driven Design of Multilayer Dielectric Thin Films).

## Student-specific parameters

| Parameter | Value | Derivation |
|---|---|---|
| Student ID | 2023270026 | — |
| `N` | 26 | last two digits |
| `λ_target` | **710 nm** | 450 + 10 × (26 mod 31) |
| `seed` | **270026** | last six digits (dataset sampling, split, network init) |
| `design_seed` | **270027** | seed + 1 (candidate screening, disjoint from training data) |

Fixed film stack: `Air / H / L / H / L / Glass` with
`n_H = 2.30`, `n_L = 1.45`, `n_sub = 1.52`, layer thickness in [40, 180] nm,
spectrum evaluated at 400–800 nm with a 10 nm step (41 points),
normal incidence, no absorption, no dispersion.

## Repository layout

```
├── config.py              # student parameters + all experiment settings
├── tmm.py                 # transfer-matrix method + sanity checks
├── gen_dataset.py         # generate 5000 samples, sequential 4000/500/500 split
├── mlp.py                 # MLP model (4-128-128-64-41) + training loop
├── train_mlp.py           # train main surrogate model, Fig.4 & Fig.5
├── data_size.py           # training-size experiment (500/1000/2000/4000), Fig.6
├── screen.py              # MLP screening of 10000 candidates + TMM check, Fig.7
├── failure.py             # worst test-sample analysis, Fig.8
├── figures_schematic.py   # schematic Fig.1-3 (workflow, model, MLP)
├── requirements.txt
├── data/                  # generated at runtime (dataset.npz, results, top5.csv)
├── figures/               # generated at runtime (Fig.1-8, PNG, 300 dpi)
└── models/                # generated at runtime (mlp.pt)
```

## Environment

- Python 3.10+ (tested on 3.12)
- numpy, torch (CPU is enough), matplotlib

```bash
pip install -r requirements.txt
```

## Reproduction steps

Run the scripts in order (each takes seconds to a few minutes on CPU):

```bash
# 1. sanity-check the TMM implementation
python tmm.py

# 2. generate the 5000-sample dataset (seed = 270026), split 4000/500/500
python gen_dataset.py

# 3. train the main MLP surrogate model, produce Fig.4 (loss) and Fig.5 (spectra)
python train_mlp.py

# 4. training-size experiment: 500/1000/2000/4000 samples, fixed val/test, Fig.6
python data_size.py

# 5. screening: 10000 candidates (design_seed = 270027), Top-5 table + Fig.7
python screen.py

# 6. failure case analysis, Fig.8
python failure.py

# 7. schematic figures Fig.1-3
python figures_schematic.py
```

Key outputs:

- `data/top5.csv` — Top-5 designs at λ_target = 710 nm
  (MLP ranking → TMM re-evaluation of Top-10 → final Top-5)
- `figures/fig4_loss.png` ... `fig8_failure.png` — all paper figures
- Test RMSE of the main model: **0.0107** (MSE = 1.15e-4)

## Reproducibility notes

- `seed = 270026` is set for `random`, `numpy` and `torch` at the start of
  every script; the network weights are initialized with this same seed in
  every experiment (including the training-size study), so all models start
  from identical weights.
- `design_seed = seed + 1 = 270027` is used **only** for the 10000 screening
  candidates, guaranteeing they are disjoint from the training set.
- `torch.set_num_threads(1)` is set inside the training loop: single-thread
  CPU execution is deterministic and, for this small model, also faster.
- The dataset split is **sequential** (first 4000 train / next 500 val /
  last 500 test), never shuffled, as required by the assignment.

## Key results

| Training samples | Test MSE | Test RMSE |
|---:|---:|---:|
| 500 | 9.53e-4 | 0.0309 |
| 1000 | 3.95e-4 | 0.0199 |
| 2000 | 1.84e-4 | 0.0136 |
| 4000 | 1.15e-4 | 0.0107 |

Top-1 screened design: `d = [77.86, 119.78, 78.73, 107.04] nm`,
TMM reflectance at 710 nm = **0.6578** (MLP prediction 0.6619).

> The MLP is used only as a fast pre-screening tool; every reported final
> performance value is re-computed with the exact TMM.
