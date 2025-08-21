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

    # Do NOT drop any element here. The MAT file you provided already has R shaped like (50, 1)
    # after removing r=0 elsewhere. We just flatten to 1D vectors.
    if R.ndim == 2 and R.shape[1] == 1:
        R_vec = R[:, 0]
    else:
        R_vec = np.ravel(R)
    Z_vec = np.ravel(Z)
    return R_vec, Z_vec


def _prepare_pressure_ZR(pressure_complex: np.ndarray, z_len: int, r_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce complex pressure to 2D (Z, R) by possibly dropping r=0 and/or transposing.

    Logic tries the following in order:
      - If shape == (Z, R): OK
      - If shape == (R, Z): transpose
      - If shape == (Z, R+1): drop first column [:, 1:]
      - If shape == (R+1, Z): drop first row [1:, :] then transpose
      - Otherwise, attempt to adapt based on last dimension matching R or R+1.
    """

    pc = np.squeeze(pressure_complex)
    if pc.ndim < 2:
        raise ValueError(f"Pressure must be at least 2D after squeeze; got {pc.shape}")

    # Keep only the last two dimensions, assuming any leading dims are singleton/meta
    if pc.ndim > 2:
        pc = pc.reshape(pc.shape[-2], pc.shape[-1])

    shape = pc.shape
    # Direct matches
    if shape == (z_len, r_len):
        pass
    elif shape == (r_len, z_len):
        pc = pc.T
    elif shape == (z_len, r_len + 1):
        pc = pc[:, 1:]
    elif shape == (r_len + 1, z_len):
        pc = pc[1:, :].T
    else:
        # Heuristics on last dimension
        if shape[-1] == r_len + 1 and shape[0] == z_len:
            pc = pc[:, 1:]
        elif shape[-1] == r_len and shape[0] == z_len:
            pass
        elif shape[-1] == z_len and shape[0] == r_len + 1:
            pc = pc[1:, :].T
        elif shape[-1] == z_len and shape[0] == r_len:
            pc = pc.T
        else:
            raise ValueError(
                f"Cannot align pressure shape {shape} to (Z,R)=({z_len},{r_len}). "
                "Please inspect MAT dimensions."
            )

    if pc.shape != (z_len, r_len):
        raise ValueError(
            f"Aligned pressure shape {pc.shape} still does not equal (Z,R)=({z_len},{r_len})."
        )
    return np.real(pc), np.imag(pc)


def main():
    parser = argparse.ArgumentParser(description="Train DeepXDE PINN from .mat pressure data")
    parser.add_argument("--mat_path", type=str, required=True, help="Path to .mat file")
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

    # Wavenumber from freqVec if available, else raise to force explicit value
    if 'freqVec' in mat:
        freq_vec = np.asarray(mat['freqVec']).astype(np.float64)
        freq_vec = np.ravel(freq_vec)
        if freq_vec.size == 0:
            raise ValueError("freqVec is empty; cannot compute k. Provide freqVec in MAT file.")
        freq_selected = float(freq_vec[0])
        omega = 2.0 * np.pi * freq_selected
        k_value = omega / float(args.c)
    else:
        raise ValueError("MAT file has single-frequency pressure but no freqVec; please add freqVec or modify script to pass k explicitly.")

    # Extract grids (no further dropping here)
    pos = mat['Pos']
    R_vec, Z_vec = _extract_pos_grids(pos)

    # Meshgrid (Z first, R second) as in the user's code
    ZZ, RR = np.meshgrid(Z_vec, R_vec, indexing="ij")

    # Coerce pressure to (Z, R) and align with RR/ZZ shapes
    z_len, r_len = ZZ.shape[0], RR.shape[1]
    pr2d, pi2d = _prepare_pressure_ZR(pressure, z_len=z_len, r_len=r_len)

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

