"""Efficient batched B1 step for uniform-with-depth material columns."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union

import torch


@dataclass(frozen=True)
class TorchB1Material:
    thermal_conductivity_w_m_k: torch.Tensor
    surface_conductance_w_m2_k: torch.Tensor
    cell_heat_capacity_j_m2_k: float
    maximum_stability_number: float


@dataclass(frozen=True)
class TorchB1Step:
    surface_temperature_k: torch.Tensor
    frost_mass_kg_m2: torch.Tensor
    subsurface_temperature_k: torch.Tensor
    condensed_kg_m2: torch.Tensor
    sublimated_kg_m2: torch.Tensor
    stored_state_energy_residual_j_m2: torch.Tensor
    energy_roundoff_projection_j_m2: torch.Tensor


def prepare_b1_material(
    thermal_inertia_j_m2_k_sqrt_s: torch.Tensor,
    *,
    volumetric_heat_capacity_j_m3_k: float,
    layer_thickness_m: float,
    surface_heat_capacity_j_m2_k: float,
    time_step_s: float,
    stability_tolerance: float = 1.0e-12,
) -> TorchB1Material:
    """Precompute per-column conductivity and the explicit stability gate."""

    if not thermal_inertia_j_m2_k_sqrt_s.is_floating_point():
        raise ValueError("thermal inertia tensor must be floating point")
    if not bool(torch.isfinite(thermal_inertia_j_m2_k_sqrt_s).all()):
        raise ValueError("thermal inertia values must be finite")
    if bool((thermal_inertia_j_m2_k_sqrt_s <= 0).any()):
        raise ValueError("thermal inertia values must be positive")
    scalar_values = (
        volumetric_heat_capacity_j_m3_k,
        layer_thickness_m,
        surface_heat_capacity_j_m2_k,
        time_step_s,
        stability_tolerance,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("material scalar inputs must be finite")
    if (
        volumetric_heat_capacity_j_m3_k <= 0
        or layer_thickness_m <= 0
        or surface_heat_capacity_j_m2_k <= 0
        or time_step_s <= 0
    ):
        raise ValueError("material, grid and time-step values must be positive")
    if stability_tolerance < 0:
        raise ValueError("stability tolerance must be non-negative")

    conductivity = (
        thermal_inertia_j_m2_k_sqrt_s**2
        / volumetric_heat_capacity_j_m3_k
    )
    surface_conductance = 2.0 * conductivity / layer_thickness_m
    surface_limit = surface_heat_capacity_j_m2_k / surface_conductance
    first_cell_limit = (
        volumetric_heat_capacity_j_m3_k
        * layer_thickness_m**2
        / (3.0 * conductivity)
    )
    stability_limit = torch.minimum(surface_limit, first_cell_limit).min()
    maximum_stability_number = time_step_s / float(stability_limit)
    if maximum_stability_number > 1.0 + stability_tolerance:
        raise ValueError(
            "explicit time step exceeds the B1 monotonic stability limit"
        )
    return TorchB1Material(
        thermal_conductivity_w_m_k=conductivity,
        surface_conductance_w_m2_k=surface_conductance,
        cell_heat_capacity_j_m2_k=(
            volumetric_heat_capacity_j_m3_k * layer_thickness_m
        ),
        maximum_stability_number=maximum_stability_number,
    )


def advance_b1_columns(
    surface_temperature_k: torch.Tensor,
    frost_mass_kg_m2: torch.Tensor,
    subsurface_temperature_k: torch.Tensor,
    *,
    material: TorchB1Material,
    time_step_s: float,
    layer_thickness_m: float,
    surface_heat_capacity_j_m2_k: float,
    frost_temperature_k: torch.Tensor,
    latent_heat_j_kg: float,
    external_surface_heat_flux_w_m2: torch.Tensor,
    bottom_downward_heat_flux_w_m2: Union[float, torch.Tensor] = 0.0,
    conserve_stored_state_energy: bool = False,
) -> TorchB1Step:
    """Advance one explicit B1 step without per-step validation or sync."""

    batch_shape = subsurface_temperature_k.shape[:-1]
    if surface_temperature_k.shape != batch_shape:
        raise ValueError("surface temperature shape must match the batch")
    if frost_mass_kg_m2.shape != batch_shape:
        raise ValueError("frost mass shape must match the batch")
    if material.thermal_conductivity_w_m_k.shape != batch_shape:
        raise ValueError("material shape must match the batch")
    if not isinstance(conserve_stored_state_energy, bool):
        raise ValueError("conserve_stored_state_energy must be boolean")

    frost_temperature = torch.broadcast_to(
        frost_temperature_k.to(
            device=surface_temperature_k.device,
            dtype=surface_temperature_k.dtype,
        ),
        batch_shape,
    )
    forcing = torch.broadcast_to(
        external_surface_heat_flux_w_m2.to(
            device=surface_temperature_k.device,
            dtype=surface_temperature_k.dtype,
        ),
        batch_shape,
    )
    bottom_flux = torch.broadcast_to(
        torch.as_tensor(
            bottom_downward_heat_flux_w_m2,
            device=surface_temperature_k.device,
            dtype=surface_temperature_k.dtype,
        ),
        batch_shape,
    )
    surface_to_subsurface_flux = (
        material.surface_conductance_w_m2_k
        * (surface_temperature_k - subsurface_temperature_k[..., 0])
    )
    surface_enthalpy = (
        surface_heat_capacity_j_m2_k * surface_temperature_k
        - latent_heat_j_kg * frost_mass_kg_m2
    )
    next_enthalpy = surface_enthalpy + time_step_s * (
        forcing - surface_to_subsurface_flux
    )
    phase_boundary_enthalpy = (
        surface_heat_capacity_j_m2_k * frost_temperature
    )
    bare = next_enthalpy >= phase_boundary_enthalpy
    next_surface_temperature = torch.where(
        bare,
        next_enthalpy / surface_heat_capacity_j_m2_k,
        frost_temperature,
    )
    next_frost_mass = torch.where(
        bare,
        torch.zeros_like(next_enthalpy),
        (phase_boundary_enthalpy - next_enthalpy) / latent_heat_j_kg,
    )

    conductivity = material.thermal_conductivity_w_m_k.unsqueeze(-1)
    face_flux = (
        -conductivity
        * (
            subsurface_temperature_k[..., 1:]
            - subsurface_temperature_k[..., :-1]
        )
        / layer_thickness_m
    )
    net_flux = torch.empty_like(subsurface_temperature_k)
    net_flux[..., 0] = surface_to_subsurface_flux - face_flux[..., 0]
    net_flux[..., 1:-1] = face_flux[..., :-1] - face_flux[..., 1:]
    net_flux[..., -1] = face_flux[..., -1] - bottom_flux
    next_subsurface_temperature = (
        subsurface_temperature_k
        + time_step_s
        * net_flux
        / material.cell_heat_capacity_j_m2_k
    )
    stored_state_energy_residual = (
        surface_heat_capacity_j_m2_k
        * (next_surface_temperature - surface_temperature_k)
        - latent_heat_j_kg * (next_frost_mass - frost_mass_kg_m2)
        + material.cell_heat_capacity_j_m2_k
        * torch.sum(
            next_subsurface_temperature - subsurface_temperature_k,
            dim=-1,
        )
        - time_step_s * (forcing - bottom_flux)
    )
    energy_roundoff_projection = torch.zeros_like(
        stored_state_energy_residual
    )
    if conserve_stored_state_energy:
        next_subsurface_temperature = next_subsurface_temperature.clone()
        for _ in range(2):
            previous_bottom_temperature = (
                next_subsurface_temperature[..., -1].clone()
            )
            next_subsurface_temperature[..., -1] -= (
                stored_state_energy_residual
                / material.cell_heat_capacity_j_m2_k
            )
            energy_roundoff_projection += (
                material.cell_heat_capacity_j_m2_k
                * (
                    next_subsurface_temperature[..., -1]
                    - previous_bottom_temperature
                )
            )
            stored_state_energy_residual = (
                surface_heat_capacity_j_m2_k
                * (next_surface_temperature - surface_temperature_k)
                - latent_heat_j_kg
                * (next_frost_mass - frost_mass_kg_m2)
                + material.cell_heat_capacity_j_m2_k
                * torch.sum(
                    next_subsurface_temperature - subsurface_temperature_k,
                    dim=-1,
                )
                - time_step_s * (forcing - bottom_flux)
            )
    return TorchB1Step(
        surface_temperature_k=next_surface_temperature,
        frost_mass_kg_m2=next_frost_mass,
        subsurface_temperature_k=next_subsurface_temperature,
        condensed_kg_m2=torch.clamp(
            next_frost_mass - frost_mass_kg_m2,
            min=0,
        ),
        sublimated_kg_m2=torch.clamp(
            frost_mass_kg_m2 - next_frost_mass,
            min=0,
        ),
        stored_state_energy_residual_j_m2=stored_state_energy_residual,
        energy_roundoff_projection_j_m2=energy_roundoff_projection,
    )


def total_b1_energy_j_m2(
    surface_temperature_k: torch.Tensor,
    frost_mass_kg_m2: torch.Tensor,
    subsurface_temperature_k: torch.Tensor,
    *,
    frost_temperature_k: torch.Tensor,
    surface_heat_capacity_j_m2_k: float,
    latent_heat_j_kg: float,
    cell_heat_capacity_j_m2_k: float,
) -> torch.Tensor:
    """Return surface enthalpy plus subsurface sensible energy."""

    return (
        surface_heat_capacity_j_m2_k
        * (surface_temperature_k - frost_temperature_k)
        - latent_heat_j_kg * frost_mass_kg_m2
        + cell_heat_capacity_j_m2_k
        * torch.sum(subsurface_temperature_k, dim=-1)
    )


def total_b1_absolute_energy_j_m2(
    surface_temperature_k: torch.Tensor,
    frost_mass_kg_m2: torch.Tensor,
    subsurface_temperature_k: torch.Tensor,
    *,
    surface_heat_capacity_j_m2_k: float,
    latent_heat_j_kg: float,
    cell_heat_capacity_j_m2_k: float,
) -> torch.Tensor:
    """Return energy using a pressure-independent temperature reference."""

    return (
        surface_heat_capacity_j_m2_k * surface_temperature_k
        - latent_heat_j_kg * frost_mass_kg_m2
        + cell_heat_capacity_j_m2_k
        * torch.sum(subsurface_temperature_k, dim=-1)
    )
