"""Deterministic horizon shadowing on a fixed regular-grid DEM."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_functional


def surface_normals_from_dem(
    elevation_m: torch.Tensor,
    *,
    spacing_m: float,
) -> torch.Tensor:
    """Return east, north, up unit normals for a north-up DEM."""

    if elevation_m.ndim != 2 or not elevation_m.is_floating_point():
        raise ValueError("elevation must be a floating-point 2D tensor")
    if spacing_m <= 0 or not math.isfinite(spacing_m):
        raise ValueError("spacing must be finite and positive")
    row_gradient, column_gradient = torch.gradient(
        elevation_m,
        spacing=(spacing_m, spacing_m),
    )
    normal = torch.stack(
        (
            -column_gradient,
            row_gradient,
            torch.ones_like(elevation_m),
        ),
        dim=-1,
    )
    return normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)


def horizon_angles_rad(
    elevation_m: torch.Tensor,
    *,
    spacing_m: float,
    azimuth_count: int,
    maximum_distance_cells: int,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute nearest-bin horizons for azimuths clockwise from north."""

    if elevation_m.ndim != 2 or not elevation_m.is_floating_point():
        raise ValueError("elevation must be a floating-point 2D tensor")
    if azimuth_count < 4:
        raise ValueError("at least four horizon azimuths are required")
    if maximum_distance_cells < 1:
        raise ValueError("maximum horizon distance must be positive")
    if spacing_m <= 0 or not math.isfinite(spacing_m):
        raise ValueError("spacing must be finite and positive")
    if valid_mask is None:
        valid_mask = torch.isfinite(elevation_m)
    if valid_mask.shape != elevation_m.shape:
        raise ValueError("valid mask must match the DEM")
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid mask must be boolean")

    rows, columns = elevation_m.shape
    dtype = elevation_m.dtype
    device = elevation_m.device
    azimuths = torch.arange(
        azimuth_count,
        device=device,
        dtype=dtype,
    ) * (2.0 * math.pi / azimuth_count)
    row = torch.arange(rows, device=device, dtype=dtype)
    column = torch.arange(columns, device=device, dtype=dtype)
    base_row, base_column = torch.meshgrid(
        row,
        column,
        indexing="ij",
    )
    horizon = torch.full(
        (azimuth_count, rows, columns),
        -0.5 * math.pi,
        device=device,
        dtype=dtype,
    )
    batched_elevation = elevation_m[None, None].expand(
        azimuth_count,
        1,
        rows,
        columns,
    )
    batched_valid = valid_mask.to(dtype)[None, None].expand(
        azimuth_count,
        1,
        rows,
        columns,
    )
    for distance_cells in range(1, maximum_distance_cells + 1):
        sample_row = (
            base_row[None]
            - torch.cos(azimuths)[:, None, None] * distance_cells
        )
        sample_column = (
            base_column[None]
            + torch.sin(azimuths)[:, None, None] * distance_cells
        )
        inside = (
            (sample_row >= 0)
            & (sample_row <= rows - 1)
            & (sample_column >= 0)
            & (sample_column <= columns - 1)
        )
        grid = torch.stack(
            (
                2.0 * sample_column / max(columns - 1, 1) - 1.0,
                2.0 * sample_row / max(rows - 1, 1) - 1.0,
            ),
            dim=-1,
        )
        sampled_elevation = torch_functional.grid_sample(
            batched_elevation,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        sampled_valid = torch_functional.grid_sample(
            batched_valid,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0] > 0.5
        angle = torch.atan2(
            sampled_elevation - elevation_m[None],
            torch.as_tensor(
                distance_cells * spacing_m,
                device=device,
                dtype=dtype,
            ),
        )
        usable = (
            inside
            & sampled_valid
            & valid_mask[None]
        )
        horizon = torch.maximum(
            horizon,
            torch.where(
                usable,
                angle,
                torch.full_like(angle, -0.5 * math.pi),
            ),
        )
    return azimuths, horizon


def direct_solar_visibility(
    horizon_angles: torch.Tensor,
    *,
    solar_azimuth_rad: torch.Tensor,
    solar_altitude_rad: torch.Tensor,
) -> torch.Tensor:
    """Return a binary visibility mask using the nearest horizon azimuth."""

    if horizon_angles.ndim != 3:
        raise ValueError("horizon angles must have shape [azimuth,row,column]")
    azimuth_count = horizon_angles.shape[0]
    bin_width = 2.0 * math.pi / azimuth_count
    index = torch.floor(
        torch.remainder(solar_azimuth_rad + 0.5 * bin_width, 2.0 * math.pi)
        / bin_width
    ).to(torch.int64)
    selected = horizon_angles[index]
    return (
        (solar_altitude_rad > 0)
        & (solar_altitude_rad > selected)
    )
