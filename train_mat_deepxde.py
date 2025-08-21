import argparse
import os
from typing import Tuple

import numpy as np
from scipy.io import loadmat

from deepxde_pinn import train_deepxde_pinn


def _extract_pos_grids(pos_struct) -> Tuple[np.ndarray, np.ndarray]:
    """Extract R (range) and Z grids from MATLAB Pos struct.

    Expected access pattern (following user's code):
      pos = mat['Pos']
      rangeR = pos['r'][0, 0]
      R = rangeR['r'][0, 0]
      Z = rangeR['z'][0, 0]
    Returns R, Z as 2D or 1D arrays depending on file; we convert to 1D vectors.
    """

    rangeR = pos_struct['r'][0, 0]
    R = rangeR['r'][0, 0]
    Z = rangeR['z'][0, 0]

    R = np.asarray(R)
    Z = np.asarray(Z)

    # User's code drops the first radial sample
    if R.ndim == 2:
        R = R[1:, :]
        if R.shape[1] == 1:
            R = R[:, 0]
    else:
        R = R[1:]

    # Flatten to 1D vectors
    R_vec = np.ravel(R)
    Z_vec = np.ravel(Z)
    return R_vec, Z_vec


def _select_frequency_and_reshape(pressure_real: np.ndarray, pressure_imag: np.ndarray, freq_idx_in_sliced: int,
                                  z_len: int, r_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Select one frequency slice from pressure arrays and reshape to (z_len, r_len).

    pressure_* are arrays after slicing with [..., 1:]. This function further selects one
    frequency by index and ensures shape matches the meshgrid (len(Z), len(R)).
    """

    pr = np.squeeze(pressure_real[..., freq_idx_in_sliced])
    pi = np.squeeze(pressure_imag[..., freq_idx_in_sliced])

    # Try to coerce to (z_len, r_len)
    if pr.shape == (z_len, r_len):
        pass
    elif pr.shape == (r_len, z_len):
        pr = pr.T
        pi = pi.T
    else:
        raise ValueError(f"Unexpected pressure slice shape {pr.shape}, expected {(z_len, r_len)} or {(r_len, z_len)}")

    return pr, pi


def main():
    parser = argparse.ArgumentParser(description="Train DeepXDE PINN from .mat pressure data")
    parser.add_argument("--mat_path", type=str, required=True, help="Path to .mat file")
    parser.add_argument("--freq_index_after_drop", type=int, default=0, help="Frequency index after dropping the first (corresponds to ...[..., 1:])")
    parser.add_argument("--c", type=float, default=1500.0, help="Sound speed (m/s)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256], help="Hidden layer sizes")
    parser.add_argument("--num_frequencies", type=int, default=64, help="Number of Fourier features")
    parser.add_argument("--fourier_scale", type=float, default=10.0, help="Fourier projection scale")
    parser.add_argument("--omega0_first", type=float, default=30.0, help="SIREN omega_0 for first layer")
    parser.add_argument("--omega0_hidden", type=float, default=1.0, help="SIREN omega_0 for hidden layers")
    parser.add_argument("--num_domain", type=int, default=10000, help="# of interior PDE points")
    parser.add_argument("--add_zmin_zero", action="store_true", help="Add zero Dirichlet samples on z = z_min")
    parser.add_argument("--boundary_num", type=int, default=500, help="# of boundary samples if add_zmin_zero")
    parser.add_argument("--epochs", type=int, default=50000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="/workspace/best_deepxde.ckpt", help="Checkpoint path")
    args = parser.parse_args()

    mat = loadmat(args.mat_path, squeeze_me=False, struct_as_record=False)

    # Pressure data: complex array with last dim as frequency.
    pressure = mat['pressure']
    pressure = pressure.astype(np.complex64)

    real_pressure = np.real(pressure)[..., 1:]
    imag_pressure = np.imag(pressure)[..., 1:]

    # Frequency vector and wavenumber k
    freq_vec = np.asarray(mat['freqVec']).astype(np.float64)
    # Convert to 1D vector
    freq_vec = np.ravel(freq_vec)
    if freq_vec.size < 2:
        raise ValueError("Expected at least 2 frequency points to drop the first; found < 2.")
    freq_selected = freq_vec[1 + args.freq_index_after_drop]
    omega = 2.0 * np.pi * float(freq_selected)
    k_value = omega / float(args.c)

    # Extract grids
    pos = mat['Pos']
    R_vec, Z_vec = _extract_pos_grids(pos)

    # Meshgrid (Z first, R second) as in the user's code
    ZZ, RR = np.meshgrid(Z_vec, R_vec, indexing="ij")

    # Select one frequency slice and ensure it matches (len(Z), len(R))
    pr2d, pi2d = _select_frequency_and_reshape(
        real_pressure, imag_pressure, args.freq_index_after_drop, ZZ.shape[0], RR.shape[1]
    )

    # Flatten to (N, 1)
    r_coords = RR.reshape(-1, 1).astype(np.float32)
    z_coords = ZZ.reshape(-1, 1).astype(np.float32)
    pressure_real = pr2d.reshape(-1, 1).astype(np.float32)
    pressure_imag = pi2d.reshape(-1, 1).astype(np.float32)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    model, state = train_deepxde_pinn(
        r_coords=r_coords,
        z_coords=z_coords,
        pressure_real=pressure_real,
        pressure_imag=pressure_imag,
        k_value=float(k_value),
        hidden_layer_sizes=tuple(args.hidden),
        num_frequencies=int(args.num_frequencies),
        fourier_scale=float(args.fourier_scale),
        omega_0_first=float(args.omega0_first),
        omega_0_hidden=float(args.omega0_hidden),
        num_domain=int(args.num_domain),
        add_boundary_samples=bool(args.add_zmin_zero),
        boundary_num=int(args.boundary_num),
        optimizer="adam",
        learning_rate=float(args.lr),
        epochs=int(args.epochs),
        model_save_path=args.save_path,
    )

    # Optionally: print best metrics
    print("Training complete. Best step:", state.best_step)


if __name__ == "__main__":
    main()

