import math
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

import deepxde as dde


class FourierFeatureMapping(torch.nn.Module):
    """Learnable Fourier feature mapping.

    Projects input coordinates using a learnable matrix and applies sin/cos.
    Output feature dimension is 2 * num_frequencies.
    """

    def __init__(self, input_dim: int, num_frequencies: int, scale: float = 10.0):
        super().__init__()
        self.num_frequencies = int(num_frequencies)
        self.scale = float(scale)
        self.projection_matrix = torch.nn.Parameter(
            torch.randn(input_dim, num_frequencies) * scale, requires_grad=True
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = 2.0 * math.pi * inputs @ self.projection_matrix
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class SineLayer(torch.nn.Module):
    """SIREN linear layer: linear -> sin(omega_0 * ·).

    Weight init follows SIREN: first layer uses U(-1/in, 1/in);
    subsequent layers use U(-sqrt(6/in)/omega_0, sqrt(6/in)/omega_0).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.is_first = bool(is_first)
        self.linear = torch.nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / float(self.linear.in_features)
                self.linear.weight.uniform_(-bound, bound)
            else:
                bound = math.sqrt(6.0 / float(self.linear.in_features)) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(inputs))


class SirenFourierNet(torch.nn.Module):
    """Fourier features + SIREN MLP, output two channels [u_r, u_i]."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_layer_sizes: Iterable[int] = (256, 256, 256),
        num_frequencies: int = 64,
        fourier_scale: float = 10.0,
        omega_0_first: float = 30.0,
        omega_0_hidden: float = 1.0,
        output_dim: int = 2,
    ):
        super().__init__()

        self.fourier = FourierFeatureMapping(
            input_dim=input_dim, num_frequencies=num_frequencies, scale=fourier_scale
        )

        layer_dims: List[int] = [2 * num_frequencies] + list(hidden_layer_sizes)

        layers: List[torch.nn.Module] = []
        if len(layer_dims) >= 2:
            layers.append(
                SineLayer(
                    in_features=layer_dims[0],
                    out_features=layer_dims[1],
                    is_first=True,
                    omega_0=omega_0_first,
                )
            )
        for i in range(1, len(layer_dims) - 1):
            layers.append(
                SineLayer(
                    in_features=layer_dims[i],
                    out_features=layer_dims[i + 1],
                    is_first=False,
                    omega_0=omega_0_hidden,
                )
            )

        self.hidden = torch.nn.Sequential(*layers)
        self.output_layer = torch.nn.Linear(layer_dims[-1], output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.fourier(inputs)
        hidden = self.hidden(features) if len(self.hidden) > 0 else features
        outputs = self.output_layer(hidden)
        return outputs


def build_axisymmetric_helmholtz_pde(k_value: float):
    """Create PDE residual function for axisymmetric Helmholtz in (r, z).

    PDE: u_rr + (1/r) u_r + u_zz + k^2 u = 0, applied to real and imaginary parts.
    Returns a function pde(x, y) -> [res_r, res_i].
    """

    k_value = float(k_value)

    def pde(x: torch.Tensor, y: torch.Tensor):
        # y: [u_r, u_i]
        u_r = y[:, 0:1]
        u_i = y[:, 1:2]

        # Derivatives wrt r (x[:, 0]) and z (x[:, 1])
        # First- and second-order derivatives for scalar fields u_r, u_i
        u_r_r_first = dde.grad.jacobian(u_r, x, i=0, j=0)
        u_r_rr = dde.grad.hessian(u_r, x, i=0, j=0)
        u_r_zz = dde.grad.hessian(u_r, x, i=1, j=1)

        u_i_r_first = dde.grad.jacobian(u_i, x, i=0, j=0)
        u_i_rr = dde.grad.hessian(u_i, x, i=0, j=0)
        u_i_zz = dde.grad.hessian(u_i, x, i=1, j=1)

        r = x[:, 0:1]
        res_r = u_r_rr + u_r_zz + (u_r_r_first / r) + (k_value ** 2) * u_r
        res_i = u_i_rr + u_i_zz + (u_i_r_first / r) + (k_value ** 2) * u_i
        return [res_r, res_i]

    return pde


def make_pointset_bc(
    xy: np.ndarray,
    values: np.ndarray,
    component: int,
    description: str = "",
):
    """Create a PointSetBC for supervised data on a specific output component."""

    assert xy.ndim == 2 and xy.shape[1] == 2, "xy must be (N, 2)"
    assert values.ndim == 2 and values.shape[1] == 1, "values must be (N, 1)"
    return dde.icbc.PointSetBC(xy, values, component=component, description=description)


def train_deepxde_pinn(
    r_coords: np.ndarray,
    z_coords: np.ndarray,
    pressure_real: np.ndarray,
    pressure_imag: np.ndarray,
    k_value: float,
    hidden_layer_sizes: Iterable[int] = (256, 256, 256),
    num_frequencies: int = 64,
    fourier_scale: float = 10.0,
    omega_0_first: float = 30.0,
    omega_0_hidden: float = 1.0,
    num_domain: int = 10000,
    add_boundary_samples: bool = True,
    boundary_num: int = 500,
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    epochs: int = 50000,
    loss_weights: Optional[List[float]] = None,
    model_save_path: Optional[str] = None,
) -> Tuple[dde.Model, dde.callbacks.TrainState]:
    """Train DeepXDE PINN with PDE residual and supervised point data.

    Inputs are numpy arrays of shape (N, 1) for r_coords, z_coords, pressure_real, pressure_imag.
    The geometry is set to the bounding rectangle of provided coordinates.
    Optionally adds zero Dirichlet samples along z = z_min to mirror the original code's BC.
    """

    # Validate and stack supervised points
    assert r_coords.shape == z_coords.shape == pressure_real.shape == pressure_imag.shape
    assert r_coords.ndim == 2 and r_coords.shape[1] == 1

    xy_supervised = np.hstack([r_coords, z_coords])
    values_real = pressure_real
    values_imag = pressure_imag

    r_min, r_max = float(np.min(r_coords)), float(np.max(r_coords))
    z_min, z_max = float(np.min(z_coords)), float(np.max(z_coords))

    # Geometry over the observed domain
    geometry = dde.geometry.Rectangle([r_min, z_min], [r_max, z_max])

    # PDE residual for real and imaginary parts
    pde = build_axisymmetric_helmholtz_pde(k_value=k_value)

    # Supervised data at given interior points
    bc_data_real = make_pointset_bc(
        xy=xy_supervised, values=values_real, component=0, description="data_real"
    )
    bc_data_imag = make_pointset_bc(
        xy=xy_supervised, values=values_imag, component=1, description="data_imag"
    )

    # Optional zero boundary along z = z_min (Dirichlet samples)
    bcs: List[dde.icbc.ICBC] = [bc_data_real, bc_data_imag]
    if add_boundary_samples and boundary_num > 0:
        r_bc = np.linspace(r_min, r_max, int(boundary_num))[:, None]
        z_bc = np.full_like(r_bc, z_min)
        xy_bc = np.hstack([r_bc, z_bc])
        zeros = np.zeros_like(r_bc)
        bcs.append(
            dde.icbc.PointSetBC(
                xy_bc, zeros, component=0, description="zmin_zero_real"
            )
        )
        bcs.append(
            dde.icbc.PointSetBC(
                xy_bc, zeros, component=1, description="zmin_zero_imag"
            )
        )

    data = dde.data.PDE(
        geometry,
        pde,
        bcs,
        num_domain=int(num_domain),
        num_boundary=0,
        anchors=None,
    )

    # Build network
    net = SirenFourierNet(
        input_dim=2,
        hidden_layer_sizes=hidden_layer_sizes,
        num_frequencies=num_frequencies,
        fourier_scale=fourier_scale,
        omega_0_first=omega_0_first,
        omega_0_hidden=omega_0_hidden,
        output_dim=2,
    )

    model = dde.Model(data, net)

    # Default loss weights: [pde_r, pde_i] + per-BC terms in order
    if loss_weights is None:
        # Emphasize PDE and data equally by default
        loss_weights = [1.0, 1.0] + [1.0 for _ in bcs]

    callbacks: List[dde.callbacks.Callback] = []
    if model_save_path is not None:
        callbacks.append(
            dde.callbacks.ModelCheckpoint(
                filepath=model_save_path, verbose=1, save_better_only=True
            )
        )

    model.compile(optimizer, lr=learning_rate, loss_weights=loss_weights)
    loss_history, train_state = model.train(epochs=int(epochs), callbacks=callbacks)

    return model, train_state


__all__ = [
    "FourierFeatureMapping",
    "SineLayer",
    "SirenFourierNet",
    "build_axisymmetric_helmholtz_pde",
    "train_deepxde_pinn",
]

