#!/usr/bin/env python3
"""Evaluate frozen B3 local-pattern reconstruction on Gasa cross-year strips."""

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

from run_e4_design_local_visibility import (
    dense_radiance,
    event_grid_coordinate,
    gaussian_kernel,
    load_dtm,
    load_final_temperature,
    masked_gaussian,
    sample_patch,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_patch(item: dict[str, object], band_indices: list[int], radius: int) -> torch.Tensor:
    cube = np.memmap(
        item["cube_path"],
        mode="r",
        dtype="<f4",
        offset=int(item["cube_offset_bytes"]),
        shape=tuple(int(value) for value in item["cube_shape"]),
        order="C",
    )
    line, sample = (int(value) for value in item["event_pixel_line_sample_zero_based"])
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


def local_field(image: torch.Tensor, sigma_pixels: float) -> torch.Tensor:
    kernel = gaussian_kernel(sigma_pixels, image.device)
    radius = kernel.shape[-1] // 2
    weight = kernel.reshape(1, 1, *kernel.shape).expand(image.shape[0], 1, -1, -1)
    padded = F.pad(image[None], (radius, radius, radius, radius), mode="reflect")
    return image - F.conv2d(padded, weight, groups=image.shape[0])[0]


def field_metrics(g_model: torch.Tensor, g_observed: torch.Tensor, valid: torch.Tensor) -> dict[str, object]:
    gm = g_model.reshape(g_model.shape[0], -1)[:, valid]
    go = g_observed.reshape(g_observed.shape[0], -1)[:, valid]
    e0 = torch.sqrt(torch.mean(go.square(), dim=1))
    e3 = torch.sqrt(torch.mean((gm - go).square(), dim=1))
    correlations = []
    for band in range(gm.shape[0]):
        a = gm[band] - gm[band].mean()
        b = go[band] - go[band].mean()
        correlations.append(float((a * b).sum() / torch.sqrt(a.square().sum() * b.square().sum()).clamp_min(1.0e-24)))
    return {
        "e0_by_band": e0.tolist(),
        "e3_by_band": e3.tolist(),
        "gain_g_by_band": (e0 - e3).tolist(),
        "mean_gain_g": float((e0 - e3).mean()),
        "correlation_by_band": correlations,
        "mean_correlation": float(np.mean(correlations)),
    }


def moving_block_bootstrap(
    g_model: np.ndarray,
    g_observed: np.ndarray,
    valid: np.ndarray,
    *,
    block_side: int,
    resamples: int,
    seed: int,
) -> dict[str, list[float]]:
    bands, rows, columns = g_model.shape
    starts = []
    for row in range(rows - block_side + 1):
        for column in range(columns - block_side + 1):
            block_valid = valid[row : row + block_side, column : column + block_side]
            if bool(block_valid.all()):
                starts.append((row, column))
    require(bool(starts), "no complete moving blocks are available")
    block_size = bands * block_side * block_side
    gm_blocks = np.stack(
        [g_model[:, r : r + block_side, c : c + block_side].reshape(block_size) for r, c in starts]
    )
    go_blocks = np.stack(
        [g_observed[:, r : r + block_side, c : c + block_side].reshape(block_size) for r, c in starts]
    )
    draws_per_sample = max(1, math.ceil(valid.sum() * bands / block_size))
    rng = np.random.default_rng(seed)
    gains = np.empty(resamples, dtype=np.float64)
    correlations = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = rng.integers(0, len(starts), size=draws_per_sample)
        gm = gm_blocks[chosen].reshape(-1)
        go = go_blocks[chosen].reshape(-1)
        e0 = np.sqrt(np.mean(go**2))
        e3 = np.sqrt(np.mean((gm - go) ** 2))
        gains[index] = e0 - e3
        correlations[index] = np.corrcoef(gm, go)[0, 1]
    return {
        "gain_g_interval_95": [float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))],
        "gain_g_lower_95_one_sided": float(np.quantile(gains, 0.05)),
        "correlation_interval_95": [
            float(np.quantile(correlations, 0.025)),
            float(np.quantile(correlations, 0.975)),
        ],
        "correlation_lower_95_one_sided": float(np.quantile(correlations, 0.05)),
        "moving_block_count": len(starts),
        "resamples": resamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(4)
    project_root = Path(config["project_root"])
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "src"))
    from mars_frost.formal_themis_observation import interpolate_band_radiance, read_temperature_radiance_table

    started = time.time()
    device = torch.device(args.device)
    band_indices = [int(value) for value in config["surface_band_indices_zero_based"]]
    lookup = read_temperature_radiance_table(Path(config["pds_temperature_radiance_table"]))
    temperature_grid = torch.tensor(lookup["temperature_k"], dtype=torch.float64, device=device)
    radiance_grid = torch.tensor(lookup["band_radiance_w_cm2_sr_um"], dtype=torch.float64, device=device)
    valid_dtm, metadata = load_dtm(Path(config["dtm_path"]))
    spacing = float(metadata["engineering_spacing_m"])
    event = event_grid_coordinate(metadata)
    gain = torch.tensor(config["frozen_gain"], dtype=torch.float64, device=device)
    offset = torch.tensor(config["frozen_offset"], dtype=torch.float64, device=device)
    by_product: dict[str, object] = {}
    gain_draws_for_mean: list[np.ndarray] = []

    for product_index, item in enumerate(config["validation_products"]):
        paths = [Path(value) for pattern in item["state_globs"] for value in sorted(glob.glob(pattern))]
        require(bool(paths), f"states are missing for {item['product_id']}")
        temperature = load_final_temperature(paths, int(valid_dtm.sum()))
        dense, mask = dense_radiance(
            temperature,
            valid_dtm,
            temperature_grid_k=temperature_grid,
            band_radiance_w_cm2_sr_um=radiance_grid,
            band_indices=band_indices,
            emissivity=float(config["emissivity"]),
            interpolate_band_radiance=interpolate_band_radiance,
            device=device,
        )
        blurred, blurred_mask = masked_gaussian(
            dense,
            mask,
            float(config["psf_sigma_m"]) / spacing,
        )
        east = float(item["base_registration_east_m"]) + float(config["frozen_registration_east_offset_m"])
        south = float(item["base_registration_south_m"]) + float(config["frozen_registration_south_offset_m"])
        prediction, support = sample_patch(
            blurred * blurred_mask,
            blurred_mask,
            event_line=event[0],
            event_sample=event[1],
            spacing_m=spacing,
            pixel_scale_m=float(config["pixel_scale_m"]),
            patch_radius=int(config["patch_radius_pixels"]),
            quadrature_axis_count=int(config["quadrature_axis_count"]),
            registration_east_m=east,
            registration_south_m=south,
        )
        valid = support >= float(config["minimum_pixel_support_fraction"])
        observed = load_patch(item, band_indices, int(config["patch_radius_pixels"])).to(device)
        calibrated = (gain[:, None] * prediction + offset[:, None]).reshape(observed.shape)
        g_model = local_field(calibrated, float(config["smoothing_sigma_pixels"]))
        g_observed = local_field(observed, float(config["smoothing_sigma_pixels"]))
        current = field_metrics(g_model, g_observed, valid)
        bootstrap = moving_block_bootstrap(
            g_model.detach().cpu().numpy(),
            g_observed.detach().cpu().numpy(),
            valid.detach().cpu().numpy().reshape(observed.shape[-2:]),
            block_side=int(config["moving_block_side_pixels"]),
            resamples=int(config["bootstrap_resamples"]),
            seed=int(config["bootstrap_seed"]) + product_index,
        )
        if (
            current["mean_gain_g"] > 0.0
            and bootstrap["gain_g_lower_95_one_sided"] > float(config["u_val_conservative"])
            and bootstrap["correlation_lower_95_one_sided"] > 0.0
        ):
            status = "SUPPORTED_BEYOND_UNCERTAINTY"
        elif current["mean_gain_g"] > 0.0 and bootstrap["correlation_lower_95_one_sided"] > 0.0:
            status = "POSITIVE_BUT_NOT_SEPARABLE_FROM_UNCERTAINTY"
        else:
            status = "NO_POSITIVE_RECONSTRUCTION_EVIDENCE"
        by_product[item["product_id"]] = {
            **current,
            **bootstrap,
            "status": status,
            "registration_east_m": east,
            "registration_south_m": south,
            "state_paths": [str(value) for value in paths],
        }
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / f"{item['product_id']}_local_fields.npz",
            observed=observed.detach().cpu().numpy(),
            calibrated_b3=calibrated.detach().cpu().numpy(),
            g_observed=g_observed.detach().cpu().numpy(),
            g_b3=g_model.detach().cpu().numpy(),
            valid=valid.detach().cpu().numpy().reshape(observed.shape[-2:]),
        )

    product_values = list(by_product.values())
    all_positive = all(float(item["mean_gain_g"]) > 0.0 for item in product_values)
    all_correlated = all(float(item["correlation_lower_95_one_sided"]) > 0.0 for item in product_values)
    mean_gain = float(np.mean([item["mean_gain_g"] for item in product_values]))
    # A positive point estimate is not enough. Every held-out strip must retain
    # a positive spatial relation and its one-sided gain lower bound must clear
    # the frozen validation floor. This is conservative when several strips are
    # available, but prevents an optimistic mean from hiding a weak holdout.
    beyond_uncertainty = all(
        float(item["gain_g_lower_95_one_sided"]) > float(config["u_val_conservative"])
        and float(item["correlation_lower_95_one_sided"]) > 0.0
        for item in product_values
    )
    positive_cross_year = all_positive and all_correlated
    overall_status = (
        "SUPPORTED_BEYOND_UNCERTAINTY"
        if beyond_uncertainty
        else "POSITIVE_BUT_NOT_SEPARABLE_FROM_UNCERTAINTY"
        if positive_cross_year
        else "OBSERVATION_CONDITION_DEPENDENT_OR_UNSUPPORTED"
    )
    result = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": overall_status,
        "by_product": by_product,
        "strip_equal_mean_gain_g": mean_gain,
        "all_product_gains_positive": all_positive,
        "all_product_correlation_lower_bounds_positive": all_correlated,
        "u_val_conservative": config["u_val_conservative"],
        "frozen_design_operator": {
            "smoothing_sigma_pixels": config["smoothing_sigma_pixels"],
            "frozen_gain": config["frozen_gain"],
            "frozen_offset": config["frozen_offset"],
            "registration_east_offset_m": config["frozen_registration_east_offset_m"],
            "registration_south_offset_m": config["frozen_registration_south_offset_m"],
            "psf_sigma_m": config["psf_sigma_m"],
        },
        "elapsed_s": time.time() - started,
        "claim_boundary": config["claim_boundary"],
    }
    output_dir = Path(config["output_dir"])
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
