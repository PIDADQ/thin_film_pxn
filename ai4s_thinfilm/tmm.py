# -*- coding: utf-8 -*-
"""Transfer-matrix method (TMM) for a lossless, non-dispersive multilayer stack.

The stack is fixed as  Air / H / L / H / L / Glass  with four layers whose
refractive indices are given by ``layer_indices`` (from the air side to the
substrate side).  Light is at normal incidence; absorption and dispersion are
ignored, so the per-layer phase thickness is real.

For each layer j the 2x2 characteristic matrix is

    M_j = [ cos(delta_j)            (i / n_j) sin(delta_j) ]
          [ i n_j sin(delta_j)      cos(delta_j)            ]

with  delta_j = 2 pi n_j d_j / lambda.  The system matrix is the ordered
product M = M_1 M_2 ... M_N.  The amplitude reflection coefficient is

    r = [ (m11 + m12 ns) n0 - (m21 + m22 ns) ] /
        [ (m11 + m12 ns) n0 + (m21 + m22 ns) ]

and the reflectance is R = |r|^2.
"""
import numpy as np


def tmm_reflectance(d_layers, layer_indices, n_sub, n_inc, wavelengths):
    """Return the reflectance spectrum R(lambda) of a multilayer stack.

    Parameters
    ----------
    d_layers : array_like, shape (N_layers,)
        Thickness of each layer in nm, ordered from the air side to substrate.
    layer_indices : array_like, shape (N_layers,)
        Refractive index of each layer (same order as d_layers).
    n_sub : float
        Refractive index of the substrate.
    n_inc : float
        Refractive index of the incident medium (air = 1.0).
    wavelengths : array_like, shape (N_wl,)
        Wavelengths in nm at which R is evaluated.

    Returns
    -------
    R : np.ndarray, shape (N_wl,)
        Reflectance values in [0, 1].
    """
    d_layers = np.asarray(d_layers, dtype=float)
    layer_indices = np.asarray(layer_indices, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    R = np.empty_like(wavelengths, dtype=float)

    for k, lam in enumerate(wavelengths):
        M = np.eye(2, dtype=complex)
        for n, d in zip(layer_indices, d_layers):
            delta = 2.0 * np.pi * n * d / lam
            Mj = np.array(
                [[np.cos(delta), 1j * np.sin(delta) / n],
                 [1j * n * np.sin(delta), np.cos(delta)]],
                dtype=complex,
            )
            M = M @ Mj

        m11, m12 = M[0, 0], M[0, 1]
        m21, m22 = M[1, 0], M[1, 1]
        num = (m11 + m12 * n_sub) * n_inc - (m21 + m22 * n_sub)
        den = (m11 + m12 * n_sub) * n_inc + (m21 + m22 * n_sub)
        r = num / den
        R[k] = (r * r.conjugate()).real

    return R


if __name__ == "__main__":
    import config

    # --- Sanity check 1: quarter-wave high-index layer --------------------
    # A single H layer with d = lambda0/(4 n_H) on glass gives a reflection
    # peak at lambda0 with R = ((n_H^2 - n_inc*n_sub)/(n_H^2 + n_inc*n_sub))^2.
    lam0 = 500.0
    d_qw = lam0 / (4.0 * config.N_H)
    R = tmm_reflectance(
        [d_qw], [config.N_H], config.N_SUB, config.N_INC,
        np.array([lam0]),
    )
    nH, n0, ns = config.N_H, config.N_INC, config.N_SUB
    R_theory = ((nH ** 2 - n0 * ns) / (nH ** 2 + n0 * ns)) ** 2
    print(f"Quarter-wave H layer: R(TMM)={R[0]:.4f}, R(theory)={R_theory:.4f}")

    # --- Sanity check 2: full four-layer spectrum is well-behaved ---------
    d = np.array([90.0, 110.0, 95.0, 120.0])
    spec = tmm_reflectance(d, config.LAYER_INDICES, config.N_SUB,
                           config.N_INC, config.WAVELENGTHS)
    assert np.all((spec >= 0.0) & (spec <= 1.0)), "R out of [0,1]"
    print(f"Four-layer spectrum: R in [{spec.min():.4f}, {spec.max():.4f}], "
          f"mean={spec.mean():.4f}")
    print("TMM sanity checks passed.")
