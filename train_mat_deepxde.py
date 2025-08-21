import argparse
import os
from typing import Tuple, Optional


def train_from_mat(
    mat_path: str,
    c: float = 1500.0,
    hidden=(256, 256, 256),
    num_frequencies: int = 64,
    fourier_scale: float = 10.0,
    omega0_first: float = 30.0,
    omega0_hidden: float = 1.0,
    num_domain: int = 10000,
    add_zmin_zero: bool = False,
    boundary_num: int = 500,
    epochs: int = 50000,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
    k_override: Optional[float] = None,
):
    """Load MAT data and train DeepXDE model programmatically (no CLI).

    Returns (model, state).
    """
    from scipy.io import loadmat
    import numpy as np

    from deepxde_pinn import train_deepxde_pinn

    mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)

    pressure = mat['pressure'].astype(np.complex64)

    pos = mat['Pos']
    R_vec, Z_vec = _extract_pos_grids(pos)
    ZZ, RR = np.meshgrid(Z_vec, R_vec, indexing="ij")
    z_len, r_len = ZZ.shape[0], RR.shape[1]

    pr2d, pi2d = _prepare_pressure_ZR(pressure, z_len=z_len, r_len=r_len)

    if k_override is not None:
        k_value = float(k_override)
    else:
        if 'freqVec' not in mat:
            raise ValueError("freqVec missing; provide k_override or add freqVec to MAT file.")
        freq_vec = np.asarray(mat['freqVec']).astype(np.float64).ravel()
        if freq_vec.size == 0:
            raise ValueError("freqVec is empty; provide k_override.")
        omega = 2.0 * np.pi * float(freq_vec[0])
        k_value = omega / float(c)

    r_coords = RR.reshape(-1, 1).astype(np.float32)
    z_coords = ZZ.reshape(-1, 1).astype(np.float32)
    pressure_real = pr2d.reshape(-1, 1).astype(np.float32)
    pressure_imag = pi2d.reshape(-1, 1).astype(np.float32)

    if save_path is not None and len(save_path) > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model, state = train_deepxde_pinn(
        r_coords=r_coords,
        z_coords=z_coords,
        pressure_real=pressure_real,
        pressure_imag=pressure_imag,
        k_value=float(k_value),
        hidden_layer_sizes=tuple(hidden),
        num_frequencies=int(num_frequencies),
        fourier_scale=float(fourier_scale),
        omega_0_first=float(omega0_first),
        omega_0_hidden=float(omega0_hidden),
        num_domain=int(num_domain),
        add_boundary_samples=bool(add_zmin_zero),
        boundary_num=int(boundary_num),
        optimizer="adam",
        learning_rate=float(lr),
        epochs=int(epochs),
        model_save_path=save_path,
    )

    return model, state