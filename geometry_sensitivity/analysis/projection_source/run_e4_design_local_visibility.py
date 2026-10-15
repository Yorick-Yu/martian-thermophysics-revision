#!/usr/bin/env python3
"""Freeze the P3-1 local-structure operator on the Gasa Design strip."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def event_grid_coordinate(metadata: dict[str, object]) -> tuple[float, float]:
    factor = float(metadata["downsample_factor"])
    half_block = 0.5 * (factor - 1.0)
    line_in_crop = (
        float(metadata["event_center_line_zero_based"])
        - float(metadata["downloaded_row_start_zero_based"])
        - float(metadata["raw_crop_row_start_in_download"])
    )
    sample_in_crop = float(metadata["event_center_sample_zero_based"]) - float(
        metadata["raw_crop_column_start"]
    )
    return (
        (line_in_crop - half_block) / factor,
        (sample_in_crop - half_block) / factor,
    )


def load_dtm(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        valid = np.asarray(payload["fully_valid"], dtype=bool)
        metadata = json.loads(str(payload["metadata_json"].item()))
    return valid, metadata


def load_final_temperature(state_paths: list[Path], valid_count: int) -> torch.Tensor:
    values: list[torch.Tensor] = []
    scenario_ids: list[torch.Tensor] = []
    for path in state_paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        values.append(state["shadow_final_surface_temperature_k"].to(torch.float64))
        scenario_ids.append(state["scenario_id"].to(torch.int64))
    value = torch.cat(values)
    ids = torch.cat(scenario_ids)
    order = torch.argsort(ids)
    require(
        torch.equal(ids[order], torch.arange(valid_count, dtype=torch.int64)),
        "state scenario IDs do not cover the valid DTM cells",
    )
    return value[order]


def load_observed_patch(config: dict[str, object]) -> torch.Tensor:
    shape = tuple(int(value) for value in config["observed_cube_shape"])
    cube = np.memmap(
        config["observed_cube_path"],
        mode="r",
        dtype="<f4",
        offset=int(config["observed_cube_offset_bytes"]),
        shape=shape,
        order="C",
    )
    line, sample = (int(value) for value in config["event_pixel_line_sample_zero_based"])
    radius = int(config["patch_radius_pixels"])
    band_indices = [int(value) for value in config["surface_band_indices_zero_based"]]
    patch = np.asarray(
        cube[
            band_indices,
            line - radius : line + radius + 1,
            sample - radius : sample + radius + 1,
        ],
        dtype=np.float64,
    )
    side = 2 * radius + 1
    require(patch.shape == (len(band_indices), side, side), "observed patch shape changed")
    require(np.isfinite(patch).all() and (patch > 0.0).all(), "observed patch is invalid")
    return torch.tensor(patch, dtype=torch.float64)


def gaussian_kernel(sigma_pixels: float, device: torch.device) -> torch.Tensor:
    require(sigma_pixels > 0.0, "Gaussian sigma must be positive")
    radius = max(2, int(math.ceil(4.0 * sigma_pixels)))
    axis = torch.arange(-radius, radius + 1, dtype=torch.float64, device=device)
    kernel = torch.exp(-0.5 * (axis / sigma_pixels) ** 2)
    kernel /= kernel.sum()
    return torch.outer(kernel, kernel)


def masked_gaussian(
    values: torch.Tensor,
    mask: torch.Tensor,
    sigma_pixels: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kernel = gaussian_kernel(sigma_pixels, values.device)
    radius = kernel.shape[-1] // 2
    weight = kernel.reshape(1, 1, *kernel.shape)
    denominator = F.conv2d(mask, weight, padding=radius)
    numerator = F.conv2d(
        values * mask,
        weight.expand(values.shape[1], 1, -1, -1),
        padding=radius,
        groups=values.shape[1],
    )
    return numerator / denominator.clamp_min(1.0e-12), denominator


def dense_radiance(
    temperature: torch.Tensor,
    valid: np.ndarray,
    *,
    temperature_grid_k: torch.Tensor,
    band_radiance_w_cm2_sr_um: torch.Tensor,
    band_indices: list[int],
    emissivity: float,
    interpolate_band_radiance,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    radiance = interpolate_band_radiance(
        temperature.to(device),
        temperature_grid_k=temperature_grid_k,
        band_radiance_w_cm2_sr_um=band_radiance_w_cm2_sr_um,
        band_indices_zero_based=band_indices,
        emissivity=emissivity,
    )
    rows, columns = valid.shape
    indices = torch.tensor(np.flatnonzero(valid.reshape(-1)), dtype=torch.int64, device=device)
    dense = torch.zeros((1, len(band_indices), rows * columns), dtype=torch.float64, device=device)
    dense[0, :, indices] = radiance.T
    mask = torch.tensor(valid, dtype=torch.float64, device=device).reshape(1, 1, rows, columns)
    return dense.reshape(1, len(band_indices), rows, columns), mask


def sample_patch(
    dense: torch.Tensor,
    mask: torch.Tensor,
    *,
    event_line: float,
    event_sample: float,
    spacing_m: float,
    pixel_scale_m: float,
    patch_radius: int,
    quadrature_axis_count: int,
    registration_east_m: float,
    registration_south_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = dense.shape[-2:]
    pixels = torch.arange(-patch_radius, patch_radius + 1, dtype=torch.float64, device=dense.device)
    line_offset, sample_offset = torch.meshgrid(pixels, pixels, indexing="ij")
    fractions = (
        (torch.arange(quadrature_axis_count, dtype=torch.float64, device=dense.device) + 0.5)
        / quadrature_axis_count
        - 0.5
    ) * pixel_scale_m
    south_subpoint, east_subpoint = torch.meshgrid(fractions, fractions, indexing="ij")
    south_m = (
        line_offset.reshape(-1, 1) * pixel_scale_m
        + registration_south_m
        + south_subpoint.reshape(1, -1)
    )
    east_m = (
        sample_offset.reshape(-1, 1) * pixel_scale_m
        + registration_east_m
        + east_subpoint.reshape(1, -1)
    )
    line = event_line + south_m / spacing_m
    sample = event_sample + east_m / spacing_m
    grid = torch.stack(
        (2.0 * sample / (columns - 1) - 1.0, 2.0 * line / (rows - 1) - 1.0),
        dim=-1,
    ).reshape(1, -1, 1, 2)
    sample_count = quadrature_axis_count**2
    pixel_count = line_offset.numel()
    numerator = F.grid_sample(dense, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    numerator = numerator.reshape(dense.shape[1], pixel_count, sample_count)
    denominator = F.grid_sample(mask, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    denominator = denominator.reshape(pixel_count, sample_count)
    prediction = numerator.sum(dim=-1) / denominator.sum(dim=-1).clamp_min(1.0e-12)[None, :]
    return prediction, denominator.mean(dim=-1)


def local_field(image: torch.Tensor, sigma_pixels: float) -> torch.Tensor:
    kernel = gaussian_kernel(sigma_pixels, image.device)
    radius = kernel.shape[-1] // 2
    weight = kernel.reshape(1, 1, *kernel.shape).expand(image.shape[0], 1, -1, -1)
    padded = F.pad(image[None], (radius, radius, radius, radius), mode="reflect")
    smooth = F.conv2d(padded, weight, groups=image.shape[0])[0]
    return image - smooth


def fit_positive_affine(predicted: torch.Tensor, observed: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = predicted[:, valid]
    y = observed.reshape(observed.shape[0], -1)[:, valid]
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    gain = (x_centered * y_centered).sum(dim=1) / (x_centered.square().sum(dim=1).clamp_min(1.0e-24))
    # Match the frozen effective-atmosphere calibration contract: transmission
    # is non-negative and cannot exceed one; path radiance is non-negative.
    gain = gain.clamp(0.0, 1.0)
    offset = (y.mean(dim=1) - gain * x.mean(dim=1)).clamp_min(0.0)
    return gain, offset


def metrics(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    valid: torch.Tensor,
    smoothing_sigma_pixels: float,
) -> dict[str, object]:
    side = observed.shape[-1]
    gain, offset = fit_positive_affine(predicted, observed, valid)
    calibrated = (gain[:, None] * predicted + offset[:, None]).reshape(observed.shape)
    g_model = local_field(calibrated, smoothing_sigma_pixels).reshape(calibrated.shape[0], -1)
    g_observed = local_field(observed, smoothing_sigma_pixels).reshape(observed.shape[0], -1)
    gm = g_model[:, valid]
    go = g_observed[:, valid]
    e0 = torch.sqrt(torch.mean(go.square(), dim=1))
    e3 = torch.sqrt(torch.mean((gm - go).square(), dim=1))
    correlation = []
    for band in range(gm.shape[0]):
        a = gm[band] - gm[band].mean()
        b = go[band] - go[band].mean()
        correlation.append(float((a * b).sum() / torch.sqrt(a.square().sum() * b.square().sum()).clamp_min(1.0e-24)))
    return {
        "gain": gain.tolist(),
        "offset": offset.tolist(),
        "e0_by_band": e0.tolist(),
        "e3_by_band": e3.tolist(),
        "gain_g_by_band": (e0 - e3).tolist(),
        "mean_gain_g": float((e0 - e3).mean()),
        "correlation_by_band": correlation,
        "mean_correlation": float(np.mean(correlation)),
        "g_model": g_model.reshape(g_model.shape[0], side, side),
        "g_observed": g_observed.reshape(g_observed.shape[0], side, side),
        "calibrated": calibrated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    project_root = Path(config["project_root"])
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    from mars_frost.formal_themis_observation import interpolate_band_radiance, read_temperature_radiance_table

    started = time.time()
    device = torch.device(args.device)
    lookup = read_temperature_radiance_table(Path(config["pds_temperature_radiance_table"]))
    temperature_grid = torch.tensor(lookup["temperature_k"], dtype=torch.float64, device=device)
    radiance_grid = torch.tensor(lookup["band_radiance_w_cm2_sr_um"], dtype=torch.float64, device=device)
    observed = load_observed_patch(config).to(device)
    side = observed.shape[-1]
    levels: dict[int, dict[str, object]] = {}
    for level_config in config["model_levels"]:
        grid_side = int(level_config["grid_side"])
        valid, metadata = load_dtm(Path(level_config["dtm_path"]))
        paths = [
            Path(value)
            for pattern in level_config["state_globs"]
            for value in sorted(glob.glob(pattern))
        ]
        require(bool(paths), f"no states found for grid {grid_side}")
        temperature = load_final_temperature(paths, int(valid.sum()))
        dense, mask = dense_radiance(
            temperature,
            valid,
            temperature_grid_k=temperature_grid,
            band_radiance_w_cm2_sr_um=radiance_grid,
            band_indices=[int(value) for value in config["surface_band_indices_zero_based"]],
            emissivity=float(config["emissivity"]),
            interpolate_band_radiance=interpolate_band_radiance,
            device=device,
        )
        spacing = float(metadata["engineering_spacing_m"])
        levels[grid_side] = {
            "valid": valid,
            "metadata": metadata,
            "event": event_grid_coordinate(metadata),
            "spacing": spacing,
            "dense": dense,
            "mask": mask,
            "state_paths": [str(value) for value in paths],
        }

    psf_sigma = float(config["nominal_psf_sigma_m"])
    search_records: list[dict[str, object]] = []
    projected_cache: dict[tuple[int, float, float, float], tuple[torch.Tensor, torch.Tensor]] = {}

    def project(grid_side: int, sigma_m: float, east_m: float, south_m: float) -> tuple[torch.Tensor, torch.Tensor]:
        key = (grid_side, sigma_m, east_m, south_m)
        if key not in projected_cache:
            level = levels[grid_side]
            blurred, blurred_mask = masked_gaussian(
                level["dense"],
                level["mask"],
                sigma_m / float(level["spacing"]),
            )
            projected_cache[key] = sample_patch(
                blurred * blurred_mask,
                blurred_mask,
                event_line=float(level["event"][0]),
                event_sample=float(level["event"][1]),
                spacing_m=float(level["spacing"]),
                pixel_scale_m=float(config["pixel_scale_m"]),
                patch_radius=int(config["patch_radius_pixels"]),
                quadrature_axis_count=int(config["quadrature_axis_count"]),
                registration_east_m=east_m,
                registration_south_m=south_m,
            )
        return projected_cache[key]

    reference_grid = int(config["reference_grid"])
    for smoothing_sigma in config["smoothing_sigma_pixels_candidates"]:
        for south in config["registration_offsets_m"]:
            for east in config["registration_offsets_m"]:
                absolute_east = float(config["base_registration_east_m"]) + float(east)
                absolute_south = float(config["base_registration_south_m"]) + float(south)
                predicted, support = project(reference_grid, psf_sigma, absolute_east, absolute_south)
                valid = support >= float(config["minimum_pixel_support_fraction"])
                current = metrics(predicted, observed, valid, float(smoothing_sigma))
                score = min(current["gain_g_by_band"]) + float(config["correlation_score_weight"]) * min(
                    current["correlation_by_band"]
                ) * float(torch.std(observed).item())
                search_records.append(
                    {
                        "smoothing_sigma_pixels": float(smoothing_sigma),
                        "registration_east_m": absolute_east,
                        "registration_south_m": absolute_south,
                        "registration_east_offset_m": float(east),
                        "registration_south_offset_m": float(south),
                        "score": float(score),
                        "mean_gain_g": current["mean_gain_g"],
                        "gain_g_by_band": current["gain_g_by_band"],
                        "mean_correlation": current["mean_correlation"],
                        "correlation_by_band": current["correlation_by_band"],
                    }
                )
    selected = max(search_records, key=lambda item: item["score"])
    nominal_prediction, nominal_support = project(
        reference_grid,
        psf_sigma,
        float(selected["registration_east_m"]),
        float(selected["registration_south_m"]),
    )
    valid = nominal_support >= float(config["minimum_pixel_support_fraction"])
    nominal = metrics(nominal_prediction, observed, valid, float(selected["smoothing_sigma_pixels"]))

    production_grid = int(config["production_grid"])
    production_prediction, production_support = project(
        production_grid,
        psf_sigma,
        float(selected["registration_east_m"]),
        float(selected["registration_south_m"]),
    )
    common = valid & (production_support >= float(config["minimum_pixel_support_fraction"]))
    gain = torch.tensor(nominal["gain"], dtype=torch.float64, device=device)
    offset = torch.tensor(nominal["offset"], dtype=torch.float64, device=device)
    production_calibrated = (gain[:, None] * production_prediction + offset[:, None]).reshape(observed.shape)
    production_local = local_field(production_calibrated, float(selected["smoothing_sigma_pixels"]))
    reference_local = nominal["g_model"]
    u_num = float(torch.sqrt(torch.mean((production_local.reshape(production_local.shape[0], -1)[:, common] - reference_local.reshape(reference_local.shape[0], -1)[:, common]) ** 2)))

    perturbation_gain: list[float] = []
    perturbation_records: list[dict[str, float]] = []
    for sigma_offset in config["psf_sigma_perturbations_m"]:
        for south_offset in config["local_registration_perturbations_m"]:
            for east_offset in config["local_registration_perturbations_m"]:
                sigma = psf_sigma + float(sigma_offset)
                east = float(selected["registration_east_m"]) + float(east_offset)
                south = float(selected["registration_south_m"]) + float(south_offset)
                prediction, support = project(reference_grid, sigma, east, south)
                current_valid = support >= float(config["minimum_pixel_support_fraction"])
                current = metrics(prediction, observed, current_valid, float(selected["smoothing_sigma_pixels"]))
                perturbation_gain.append(float(current["mean_gain_g"]))
                perturbation_records.append(
                    {
                        "psf_sigma_m": sigma,
                        "registration_east_m": east,
                        "registration_south_m": south,
                        "mean_gain_g": float(current["mean_gain_g"]),
                        "mean_correlation": float(current["mean_correlation"]),
                    }
                )
    nominal_gain = float(nominal["mean_gain_g"])
    u_sys = float(np.quantile(np.abs(np.asarray(perturbation_gain) - nominal_gain), 0.95))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "design_local_fields.npz",
        observed=observed.detach().cpu().numpy(),
        calibrated_b3=nominal["calibrated"].detach().cpu().numpy(),
        g_observed=nominal["g_observed"].detach().cpu().numpy(),
        g_b3=nominal["g_model"].detach().cpu().numpy(),
        valid=valid.detach().cpu().numpy().reshape(side, side),
    )
    result = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": "PASS" if nominal_gain > 0.0 and min(nominal["correlation_by_band"]) > 0.0 else "FAIL",
        "selected_design_operator": selected,
        "nominal": {key: value for key, value in nominal.items() if not isinstance(value, torch.Tensor)},
        "u_num_radiance_rmse": u_num,
        "u_sys_gain_p95": u_sys,
        "u_val_conservative": u_num + u_sys,
        "local_signal_p95": float(torch.quantile(torch.abs(nominal["g_model"][:, valid.reshape(side, side)]), 0.95)),
        "local_signal_exceeds_u_num_plus_u_sys": float(torch.quantile(torch.abs(nominal["g_model"][:, valid.reshape(side, side)]), 0.95)) > u_num + u_sys,
        "search_records": search_records,
        "perturbation_records": perturbation_records,
        "inputs": {
            "observed_product_id": config["observed_product_id"],
            "observed_cube_path": config["observed_cube_path"],
            "surface_band_indices_zero_based": config["surface_band_indices_zero_based"],
            "model_levels": {str(key): value["state_paths"] for key, value in levels.items()},
        },
        "elapsed_s": time.time() - started,
        "claim_boundary": config["claim_boundary"],
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
