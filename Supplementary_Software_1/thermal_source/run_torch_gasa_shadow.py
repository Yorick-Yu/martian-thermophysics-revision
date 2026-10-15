#!/usr/bin/env python3
"""Run paired local-slope and horizon-shadow B1 columns on a Gasa DTM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from mars_frost.mars_forcing import (
    MARS_SOL_SECONDS,
    solar_declination_rad,
    solar_flux_w_m2,
    solar_longitude_after_elapsed_s,
)
from mars_frost.mcd_forcing import (
    interpolate_mcd_variable,
    load_mcd_forcing,
)
from mars_frost.radiation import STEFAN_BOLTZMANN_W_M2_K4
from mars_frost.torch_b1 import (
    advance_b1_columns,
    prepare_b1_material,
    total_b1_absolute_energy_j_m2,
)
from mars_frost.torch_topographic_shadow import (
    horizon_angles_rad,
    surface_normals_from_dem,
)


AXIS_NAMES = (
    "thermal_inertia_j_m2_k_sqrt_s",
    "substrate_albedo",
    "surface_pressure_pa",
    "initial_solar_longitude_deg",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _neumaier_add(
    total: torch.Tensor,
    compensation: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    updated = total + value
    correction = torch.where(
        total.abs() >= value.abs(),
        (total - updated) + value,
        (value - updated) + total,
    )
    return updated, compensation + correction


def _axis_values(config: dict[str, Any], name: str) -> list[float]:
    values = [float(value) for value in config["parameter_axes"][name]]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite values")
    return values


def _shard_scenario_count(
    total_scenarios: int,
    shard_index: int,
    shard_count: int,
) -> int:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard index lies outside the frozen shard count")
    if shard_index >= total_scenarios:
        return 0
    return (total_scenarios - 1 - shard_index) // shard_count + 1


def _scenario_ids_for_chunk(
    *,
    total_scenarios: int,
    shard_index: int,
    shard_count: int,
    local_offset: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    shard_total = _shard_scenario_count(
        total_scenarios,
        shard_index,
        shard_count,
    )
    if local_offset < 0 or local_offset + count > shard_total:
        raise ValueError("chunk range lies outside the shard")
    local_ids = torch.arange(
        local_offset,
        local_offset + count,
        dtype=torch.int64,
        device=device,
    )
    return shard_index + shard_count * local_ids


def _decode_scenario_ids(
    scenario_ids: torch.Tensor,
    *,
    valid_cell_count: int,
    config: dict[str, Any],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cell_index = torch.remainder(scenario_ids, valid_cell_count)
    remainder = torch.div(
        scenario_ids,
        valid_cell_count,
        rounding_mode="floor",
    )
    decoded: dict[str, torch.Tensor] = {}
    for name in AXIS_NAMES:
        values = torch.tensor(
            _axis_values(config, name),
            dtype=dtype,
            device=scenario_ids.device,
        )
        index = torch.remainder(remainder, values.numel())
        remainder = torch.div(
            remainder,
            values.numel(),
            rounding_mode="floor",
        )
        decoded[name] = values[index]
    if bool((remainder != 0).any()):
        raise ValueError("scenario ID exceeds the configured grid")
    return cell_index, decoded


def _parameter_combination_count(config: dict[str, Any]) -> int:
    count = 1
    for name in AXIS_NAMES:
        count *= len(_axis_values(config, name))
    return count


def _formal_step_count(config: dict[str, Any]) -> int:
    annual = config.get("annual_cycle_design")
    time_step_s = float(config["time_step_s"])
    sol_seconds = float(config.get("mars_sol_seconds", MARS_SOL_SECONDS))
    if annual is None:
        return math.ceil(float(config["sol_count"]) * sol_seconds / time_step_s)
    year_duration_s = float(annual["sols_per_mars_year"]) * sol_seconds
    steps_per_year = math.ceil(year_duration_s / time_step_s)
    return int(annual["mars_year_count"]) * steps_per_year


def _step_schedule(
    config: dict[str, Any],
    *,
    step_count: int,
) -> list[tuple[float, float, int, bool]]:
    time_step_s = float(config["time_step_s"])
    sol_seconds = float(config.get("mars_sol_seconds", MARS_SOL_SECONDS))
    annual = config.get("annual_cycle_design")
    if annual is None:
        duration_s = float(config["sol_count"]) * sol_seconds
        schedule = []
        for step_index in range(step_count):
            start_s = step_index * time_step_s
            duration = min(time_step_s, duration_s - start_s)
            if duration <= 0:
                raise ValueError("forcing table extends beyond the duration")
            schedule.append(
                (start_s, duration, 0, step_index + 1 == step_count)
            )
        return schedule

    year_duration_s = float(annual["sols_per_mars_year"]) * sol_seconds
    steps_per_year = math.ceil(year_duration_s / time_step_s)
    formal_steps = int(annual["mars_year_count"]) * steps_per_year
    if not 1 <= step_count <= formal_steps:
        raise ValueError("annual step count lies outside the frozen design")
    schedule = []
    for step_index in range(step_count):
        year_index, step_in_year = divmod(step_index, steps_per_year)
        start_in_year_s = step_in_year * time_step_s
        duration = min(time_step_s, year_duration_s - start_in_year_s)
        if duration <= 0:
            raise ValueError("annual forcing step extends beyond its year")
        schedule.append(
            (
                year_index * year_duration_s + start_in_year_s,
                duration,
                year_index,
                step_in_year + 1 == steps_per_year,
            )
        )
    return schedule


def _forcing_tables(
    config: dict[str, Any],
    *,
    step_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    sol_seconds = float(config.get("mars_sol_seconds", MARS_SOL_SECONDS))
    initial_longitudes = _axis_values(
        config,
        "initial_solar_longitude_deg",
    )
    solar_flux_rows: list[list[float]] = []
    declination_rows: list[list[float]] = []
    hour_angle_rows: list[float] = []
    duration_rows: list[float] = []
    mcd_rows: dict[str, list[list[float]]] = {
        "surface_pressure_pa": [],
        "air_temperature_k": [],
        "dust_optical_depth": [],
        "downwelling_longwave_w_m2": [],
        "horizontal_shortwave_w_m2": [],
    }
    schedule = _step_schedule(config, step_count=step_count)
    year_indices: list[int] = []
    year_end_flags: list[bool] = []
    mcd_config = config.get("mcd_forcing")
    mcd_payload = None
    if mcd_config is not None:
        mcd_payload = load_mcd_forcing(
            Path(mcd_config["path"]),
            expected_sha256=mcd_config.get("sha256"),
        )
    for step_start, step_duration, year_index, year_end in schedule:
        midpoint = step_start + 0.5 * step_duration
        local_fraction = (midpoint % sol_seconds) / sol_seconds
        hour_angle_rows.append(2.0 * math.pi * (local_fraction - 0.5))
        duration_rows.append(step_duration)
        year_indices.append(year_index)
        year_end_flags.append(year_end)
        flux_row: list[float] = []
        declination_row: list[float] = []
        step_mcd = {name: [] for name in mcd_rows}
        for initial_longitude in initial_longitudes:
            longitude = solar_longitude_after_elapsed_s(
                initial_longitude,
                midpoint,
            )
            flux_row.append(solar_flux_w_m2(longitude))
            declination_row.append(solar_declination_rad(longitude))
            if mcd_payload is not None:
                local_time_hours = 24.0 * local_fraction
                step_mcd["surface_pressure_pa"].append(
                    interpolate_mcd_variable(
                        mcd_payload,
                        "ps",
                        solar_longitude_deg=longitude,
                        local_time_hours=local_time_hours,
                    )
                )
                step_mcd["air_temperature_k"].append(
                    interpolate_mcd_variable(
                        mcd_payload,
                        "t",
                        solar_longitude_deg=longitude,
                        local_time_hours=local_time_hours,
                    )
                )
                step_mcd["dust_optical_depth"].append(
                    interpolate_mcd_variable(
                        mcd_payload,
                        "tauref",
                        solar_longitude_deg=longitude,
                        local_time_hours=local_time_hours,
                    )
                )
                step_mcd["downwelling_longwave_w_m2"].append(
                    interpolate_mcd_variable(
                        mcd_payload,
                        "fluxsurf_lw",
                        solar_longitude_deg=longitude,
                        local_time_hours=local_time_hours,
                    )
                )
                step_mcd["horizontal_shortwave_w_m2"].append(
                    interpolate_mcd_variable(
                        mcd_payload,
                        "fluxsurf_dn_sw",
                        solar_longitude_deg=longitude,
                        local_time_hours=local_time_hours,
                    )
                )
        solar_flux_rows.append(flux_row)
        declination_rows.append(declination_row)
        if mcd_payload is not None:
            for name in mcd_rows:
                mcd_rows[name].append(step_mcd[name])
    result = {
        "solar_flux_w_m2": torch.tensor(
            solar_flux_rows,
            dtype=dtype,
            device=device,
        ),
        "declination_rad": torch.tensor(
            declination_rows,
            dtype=dtype,
            device=device,
        ),
        "hour_angle_rad": torch.tensor(
            hour_angle_rows,
            dtype=dtype,
            device=device,
        ),
        "step_duration_s": torch.tensor(
            duration_rows,
            dtype=dtype,
            device=device,
        ),
        "mars_year_index": torch.tensor(
            year_indices,
            dtype=torch.int64,
            device=device,
        ),
        "mars_year_end": torch.tensor(
            year_end_flags,
            dtype=torch.bool,
            device=device,
        ),
    }
    if mcd_payload is not None:
        for name, rows in mcd_rows.items():
            result[name] = torch.tensor(
                rows,
                dtype=dtype,
                device=device,
            )
    return result


def _forcing_summary(
    config: dict[str, Any],
    forcing: dict[str, torch.Tensor],
) -> dict[str, Any]:
    mcd_config = config.get("mcd_forcing")
    if mcd_config is None:
        return {"mode": "synthetic_constant_pressure_and_longwave"}
    mcd_payload = load_mcd_forcing(
        Path(mcd_config["path"]),
        expected_sha256=mcd_config.get("sha256"),
    )
    variables = {}
    for name in (
        "surface_pressure_pa",
        "air_temperature_k",
        "dust_optical_depth",
        "downwelling_longwave_w_m2",
        "horizontal_shortwave_w_m2",
    ):
        tensor = forcing[name]
        variables[name] = {
            "minimum": float(tensor.min()),
            "maximum": float(tensor.max()),
        }
    summary = {
        "mode": "MCD_v6.1_climatology_time_varying",
        "path": str(mcd_config["path"]),
        "size_bytes": Path(mcd_config["path"]).stat().st_size,
        "scenario": str(mcd_payload["scenario"]),
        "scenario_code": mcd_payload.get("scenario_code", 1),
        "variables": variables,
    }
    if "sha256" in mcd_config:
        summary["sha256"] = str(mcd_config["sha256"])
    return summary


def _sun_vector_enu(
    latitude_rad: float,
    declination_rad: torch.Tensor,
    hour_angle_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sin_declination = torch.sin(declination_rad)
    cos_declination = torch.cos(declination_rad)
    sin_latitude = math.sin(latitude_rad)
    cos_latitude = math.cos(latitude_rad)
    east = -cos_declination * torch.sin(hour_angle_rad)
    north = (
        sin_declination * cos_latitude
        - cos_declination * torch.cos(hour_angle_rad) * sin_latitude
    )
    up = (
        sin_declination * sin_latitude
        + cos_declination * torch.cos(hour_angle_rad) * cos_latitude
    )
    return east, north, up


def _fingerprint_controls(
    scenario_ids: torch.Tensor,
    tensors: tuple[torch.Tensor, ...],
) -> str:
    digest = hashlib.sha256()
    for value in scenario_ids.detach().cpu().tolist():
        digest.update(struct.pack("<q", int(value)))
    for tensor in tensors:
        for value in tensor.detach().cpu().reshape(-1).tolist():
            digest.update(struct.pack("<d", float(value)))
    return digest.hexdigest()


def _write_torch_payload(
    path: Path,
    payload: dict[str, Any],
    *,
    record_sha256: bool = True,
) -> str | None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite state file: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(
            f"refusing to overwrite temporary state file: {temporary}"
        )
    torch.save(payload, temporary)
    temporary.replace(path)
    return _sha256(path) if record_sha256 else None


def _append_json_line(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _acquisition_alignment_summary(config: dict[str, Any]) -> dict[str, Any]:
    alignment = config.get("acquisition_alignment")
    if alignment is None:
        return {}

    target_local_time_hours = float(
        alignment["target_local_true_solar_time_hours"]
    )
    spinup_complete_sols = int(alignment["spinup_complete_sols"])
    sol_seconds = float(config.get("mars_sol_seconds", MARS_SOL_SECONDS))
    sol_count = float(config["sol_count"])
    elapsed_s = sol_count * sol_seconds
    expected_sol_count = (
        spinup_complete_sols + target_local_time_hours / 24.0
    )
    initial_longitudes = _axis_values(
        config,
        "initial_solar_longitude_deg",
    )
    attained_solar_longitude_deg = solar_longitude_after_elapsed_s(
        initial_longitudes[0],
        elapsed_s,
    )
    target_solar_longitude_deg = float(
        alignment["target_solar_longitude_deg"]
    )
    solar_longitude_error_deg = (
        (
            attained_solar_longitude_deg
            - target_solar_longitude_deg
            + 180.0
        )
        % 360.0
        - 180.0
    )
    attained_local_time_hours = ((elapsed_s % sol_seconds) / sol_seconds) * 24.0
    local_time_error_hours = (
        (
            attained_local_time_hours
            - target_local_time_hours
            + 12.0
        )
        % 24.0
        - 12.0
    )
    return {
        "product_id": str(alignment["product_id"]),
        "target_observation_utc": str(
            alignment["target_observation_utc"]
        ),
        "spinup_complete_sols": spinup_complete_sols,
        "expected_sol_count": expected_sol_count,
        "elapsed_s": elapsed_s,
        "initial_solar_longitude_deg": initial_longitudes[0],
        "target_solar_longitude_deg": target_solar_longitude_deg,
        "attained_solar_longitude_deg": attained_solar_longitude_deg,
        "solar_longitude_error_deg": solar_longitude_error_deg,
        "target_local_true_solar_time_hours": target_local_time_hours,
        "attained_local_true_solar_time_hours": attained_local_time_hours,
        "local_true_solar_time_error_s": local_time_error_hours * 3600.0,
    }


def _formal_run_mode(config: dict[str, Any]) -> str:
    if config.get("annual_cycle_design") is not None:
        return "formal_multi_mars_year"
    if (
        config.get("mcd_forcing") is not None
        and config.get("acquisition_alignment") is not None
    ):
        return "formal_mcd_acquisition_aligned"
    if config.get("mcd_forcing") is not None:
        return "formal_mars_year"
    if config.get("acquisition_alignment") is not None:
        return "formal_acquisition_aligned"
    return "formal_30_sol"


def _validate_config(config: dict[str, Any]) -> None:
    shard_count = int(config["shard_count"])
    scenario_family = config.get("scenario_design", {}).get("family")
    annual_family = config.get("annual_cycle_design", {}).get("family")
    is_rb_s22_scenario = (
        scenario_family == "RB-S22-GASA-MCD61-SCENARIO-ROBUSTNESS"
    )
    is_rb_s23_annual = (
        annual_family == "RB-S23-GASA-MCD61-MULTIYEAR-CONVERGENCE"
    )
    is_formal_e5_annual = (
        annual_family == "FORMAL-E5-HETEROGENEOUS-TWIN-V1"
    )
    is_formal_e6_annual = annual_family in {
        "FORMAL-E6-OBSERVATION-SCHEDULE-V1",
        "FORMAL-E6-OBSERVATION-SCHEDULE-V2",
        "FORMAL-E6-OBSERVATION-SCHEDULE-V3",
    }
    is_registered_annual = (
        is_rb_s23_annual or is_formal_e5_annual or is_formal_e6_annual
    )
    if shard_count != 4 and not (
        shard_count == 1 and (is_rb_s22_scenario or is_registered_annual)
    ):
        raise ValueError(
            "the frozen design requires four shards, or one shard for RB-S22 "
            "or a registered annual family"
        )
    if str(config["dtype"]) != "float64":
        raise ValueError("the frozen design requires float64")
    if not isinstance(
        config.get("conserve_stored_state_energy", False),
        bool,
    ):
        raise ValueError("conserve_stored_state_energy must be boolean")
    if not isinstance(config.get("record_sha256", True), bool):
        raise ValueError("record_sha256 must be boolean")
    if tuple(config["parameter_axes"]) != AXIS_NAMES:
        raise ValueError(f"parameter axes must be ordered as {AXIS_NAMES}")
    for name in AXIS_NAMES:
        _axis_values(config, name)
    if not all(
        value > 0
        for value in _axis_values(
            config,
            "thermal_inertia_j_m2_k_sqrt_s",
        )
    ):
        raise ValueError("thermal inertia values must be positive")
    if not all(
        0 <= value <= 1
        for value in _axis_values(config, "substrate_albedo")
    ):
        raise ValueError("albedo values must lie in [0,1]")
    if int(config["layer_count"]) < 2:
        raise ValueError("at least two subsurface layers are required")
    if int(config["chunk_size_per_gpu"]) < 1:
        raise ValueError("chunk size must be positive")
    if int(config["horizon_azimuth_count"]) < 4:
        raise ValueError("at least four horizon azimuths are required")
    if int(config["horizon_maximum_distance_cells"]) < 1:
        raise ValueError("horizon distance must be positive")
    snapshot_stride = config.get("snapshot_stride_steps")
    explicit_snapshot_indices = config.get(
        "snapshot_step_indices_zero_based"
    )
    if snapshot_stride is not None and explicit_snapshot_indices is not None:
        raise ValueError(
            "snapshot stride and explicit snapshot indices are mutually exclusive"
        )
    if snapshot_stride is not None and int(snapshot_stride) < 1:
        raise ValueError("snapshot stride must be positive")
    if explicit_snapshot_indices is not None:
        if not isinstance(explicit_snapshot_indices, list) or not explicit_snapshot_indices:
            raise ValueError("explicit snapshot indices must be a non-empty list")
        if not all(type(value) is int for value in explicit_snapshot_indices):
            raise ValueError("explicit snapshot indices must be integers")
        if len(explicit_snapshot_indices) != len(set(explicit_snapshot_indices)):
            raise ValueError("explicit snapshot indices must be unique")
        if explicit_snapshot_indices != sorted(explicit_snapshot_indices):
            raise ValueError("explicit snapshot indices must be ascending")
    annual = config.get("annual_cycle_design")
    if annual is not None:
        if not is_registered_annual:
            raise ValueError("annual-cycle execution is not registered")
        year_count = int(annual["mars_year_count"])
        sols_per_year = float(annual["sols_per_mars_year"])
        if year_count < 1 or sols_per_year <= 0:
            raise ValueError("annual-cycle design requires at least one year")
        if year_count == 1 and config.get("initial_state") is None:
            raise ValueError(
                "a one-year annual cycle requires a periodic initial state"
            )
        expected_sols = year_count * sols_per_year
        if not math.isclose(
            float(config["sol_count"]),
            expected_sols,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("sol_count does not match the annual-cycle design")
        if not str(annual.get("initial_state_id", "")).strip():
            raise ValueError("annual-cycle design requires an initial-state ID")
    if explicit_snapshot_indices is not None:
        formal_step_count = _formal_step_count(config)
        if (
            explicit_snapshot_indices[0] < 0
            or explicit_snapshot_indices[-1] >= formal_step_count
        ):
            raise ValueError("explicit snapshot index lies outside the formal run")
    alignment = _acquisition_alignment_summary(config)
    if alignment:
        if len(_axis_values(config, "initial_solar_longitude_deg")) != 1:
            raise ValueError(
                "acquisition alignment requires one initial solar longitude"
            )
        if not 0 <= alignment["target_local_true_solar_time_hours"] < 24:
            raise ValueError("target local true solar time must lie in [0,24)")
        if alignment["spinup_complete_sols"] < 0:
            raise ValueError("spin-up sol count must be non-negative")
        if not math.isclose(
            float(config["sol_count"]),
            alignment["expected_sol_count"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "sol_count does not end at the target local solar time"
            )
        if abs(alignment["solar_longitude_error_deg"]) > float(
            config["maximum_solar_longitude_alignment_error_deg"]
        ):
            raise ValueError("initial solar longitude misses the target Ls")
        if abs(alignment["local_true_solar_time_error_s"]) > float(
            config["maximum_local_time_alignment_error_s"]
        ):
            raise ValueError("run duration misses the target local time")
    mcd_config = config.get("mcd_forcing")
    if mcd_config is not None:
        load_mcd_forcing(
            Path(mcd_config["path"]),
            expected_sha256=mcd_config.get("sha256"),
        )
        if len(_axis_values(config, "surface_pressure_pa")) != 1:
            raise ValueError("MCD forcing requires one legacy pressure axis")


def _load_geometry(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    config: dict[str, Any],
) -> dict[str, Any]:
    if "dtm_sha256" in config:
        observed_sha256 = _sha256(path)
        if observed_sha256 != str(config["dtm_sha256"]):
            raise ValueError("DTM SHA256 does not match the frozen config")
    if "dtm_size_bytes" in config:
        if path.stat().st_size != int(config["dtm_size_bytes"]):
            raise ValueError("DTM size does not match the frozen config")
    with np.load(path, allow_pickle=False) as payload:
        elevation_numpy = np.asarray(payload["elevation_m"], dtype=np.float64)
        valid_numpy = np.asarray(payload["fully_valid"], dtype=bool)
        metadata = json.loads(str(payload["metadata_json"]))
    if metadata["product_id"] != str(config["dtm_product_id"]):
        raise ValueError("DTM product ID does not match the frozen config")
    if elevation_numpy.ndim != 2 or valid_numpy.shape != elevation_numpy.shape:
        raise ValueError("DTM payload has inconsistent array shapes")
    if not bool(valid_numpy.any()):
        raise ValueError("DTM payload has no fully valid cell")
    if not np.isfinite(elevation_numpy[valid_numpy]).all():
        raise ValueError("valid DTM cells must have finite elevations")
    fill_value = float(np.median(elevation_numpy[valid_numpy]))
    elevation_numpy = np.where(valid_numpy, elevation_numpy, fill_value)
    elevation = torch.tensor(
        elevation_numpy,
        dtype=dtype,
        device=device,
    )
    valid = torch.tensor(valid_numpy, dtype=torch.bool, device=device)
    normals = surface_normals_from_dem(
        elevation,
        spacing_m=float(metadata["engineering_spacing_m"]),
    )
    _, horizons = horizon_angles_rad(
        elevation,
        spacing_m=float(metadata["engineering_spacing_m"]),
        azimuth_count=int(config["horizon_azimuth_count"]),
        maximum_distance_cells=int(
            config["horizon_maximum_distance_cells"]
        ),
        valid_mask=valid,
    )
    valid_flat_indices = torch.nonzero(
        valid.reshape(-1),
        as_tuple=False,
    )[:, 0]
    return {
        "metadata": metadata,
        "shape": tuple(int(value) for value in elevation.shape),
        "valid_flat_indices": valid_flat_indices,
        "normals": normals.reshape(-1, 3)[valid_flat_indices],
        "horizons": horizons.reshape(horizons.shape[0], -1)[
            :,
            valid_flat_indices,
        ],
    }


def _load_initial_state(
    config: dict[str, Any],
    *,
    total_scenarios: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    state_config = config.get("initial_state")
    if state_config is None:
        return None
    path = Path(state_config["path"])
    if "size_bytes" in state_config:
        if path.stat().st_size != int(state_config["size_bytes"]):
            raise ValueError("initial-state size does not match config")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "scenario_id",
        "baseline_surface_temperature_k",
        "shadow_surface_temperature_k",
        "baseline_frost_mass_kg_m2",
        "shadow_frost_mass_kg_m2",
        "baseline_subsurface_temperature_k",
        "shadow_subsurface_temperature_k",
    }
    if not required.issubset(payload):
        raise ValueError("initial-state payload is incomplete")
    scenario_id = payload["scenario_id"].to(torch.int64).reshape(-1)
    if not torch.equal(
        scenario_id,
        torch.arange(total_scenarios, dtype=torch.int64),
    ):
        raise ValueError("initial-state scenario IDs do not match the run")
    loaded: dict[str, torch.Tensor] = {"scenario_id": scenario_id.to(device)}
    for name in required - {"scenario_id"}:
        value = payload[name]
        if value.shape[0] != total_scenarios:
            raise ValueError(f"initial-state field {name} has wrong length")
        loaded[name] = value.to(device=device, dtype=dtype)
    return loaded


def run(
    config: dict[str, Any],
    *,
    device_name: str,
    shard_index: int,
    state_dir: Path | None,
    chunks_manifest: Path | None,
    pilot_steps: int | None = None,
    pilot_chunks: int = 1,
) -> dict[str, Any]:
    _validate_config(config)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = torch.float64
    dtm_path = Path(config["dtm_path"])
    geometry = _load_geometry(
        dtm_path,
        device=device,
        dtype=dtype,
        config=config,
    )
    valid_cell_count = int(geometry["valid_flat_indices"].numel())
    total_scenarios = (
        valid_cell_count * _parameter_combination_count(config)
    )
    initial_state = _load_initial_state(
        config,
        total_scenarios=total_scenarios,
        device=device,
        dtype=dtype,
    )
    controls = [int(value) for value in config["control_scenario_ids"]]
    if len(controls) != len(set(controls)):
        raise ValueError("control scenario IDs must be unique")
    if not all(0 <= value < total_scenarios for value in controls):
        raise ValueError("control scenario ID lies outside the grid")

    shard_count = int(config["shard_count"])
    shard_scenarios = _shard_scenario_count(
        total_scenarios,
        shard_index,
        shard_count,
    )
    formal_step_count = _formal_step_count(config)
    if pilot_steps is not None:
        if not 1 <= pilot_steps <= formal_step_count:
            raise ValueError("pilot steps lie outside the formal run")
        if pilot_chunks < 1:
            raise ValueError("pilot chunk count must be positive")
        step_count = pilot_steps
        max_chunks = pilot_chunks
        mode = "bounded_pilot"
    else:
        step_count = formal_step_count
        max_chunks = None
        mode = _formal_run_mode(config)
        if state_dir is None or chunks_manifest is None:
            raise ValueError("formal runs require state and manifest paths")

    chunk_size = int(config["chunk_size_per_gpu"])
    total_chunks = math.ceil(shard_scenarios / chunk_size)
    chunks_to_run = (
        total_chunks if max_chunks is None else min(total_chunks, max_chunks)
    )
    if state_dir is not None:
        if state_dir.exists() and any(state_dir.iterdir()):
            raise FileExistsError(
                f"refusing to reuse non-empty state directory: {state_dir}"
            )
        state_dir.mkdir(parents=True, exist_ok=True)
    if chunks_manifest is not None:
        if chunks_manifest.exists():
            raise FileExistsError(
                f"refusing to overwrite chunk manifest: {chunks_manifest}"
            )
        chunks_manifest.parent.mkdir(parents=True, exist_ok=True)
        chunks_manifest.touch()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forcing = _forcing_tables(
        config,
        step_count=step_count,
        device=device,
        dtype=dtype,
    )
    initial_longitudes = _axis_values(
        config,
        "initial_solar_longitude_deg",
    )
    longitude_count = len(initial_longitudes)
    latitude_rad = math.radians(
        float(geometry["metadata"]["event_center_latitude_deg"])
    )
    horizon_bin_width = (
        2.0 * math.pi / int(config["horizon_azimuth_count"])
    )
    frost_epsilon = float(config["frost_mass_epsilon_kg_m2"])
    surface_heat_capacity = float(
        config["surface_heat_capacity_j_m2_k"]
    )
    layer_thickness = float(config["layer_thickness_m"])
    volumetric_heat_capacity = float(
        config["volumetric_heat_capacity_j_m3_k"]
    )
    latent_heat = float(config["latent_heat_j_kg"])
    bottom_flux = float(config["bottom_downward_heat_flux_w_m2"])
    progress_every = int(config["progress_every_steps"])
    annual_design = config.get("annual_cycle_design")
    record_sha256 = bool(config.get("record_sha256", True))
    conserve_stored_state_energy = config.get(
        "conserve_stored_state_energy",
        False,
    )
    annual_year_count = (
        int(annual_design["mars_year_count"])
        if annual_design is not None
        else 0
    )
    annual_steps_per_year = (
        formal_step_count // annual_year_count
        if annual_year_count
        else formal_step_count
    )
    snapshot_stride = config.get("snapshot_stride_steps")
    explicit_snapshot_indices = config.get(
        "snapshot_step_indices_zero_based"
    )
    snapshot_step_indices: list[int] = []
    snapshot_mars_year_indices: list[int] = []
    snapshot_step_in_year: list[int] = []
    if explicit_snapshot_indices is not None:
        snapshot_step_indices = [
            int(step_index)
            for step_index in explicit_snapshot_indices
            if int(step_index) < step_count
        ]
    elif snapshot_stride is not None:
        snapshot_year_index = int(
            config.get(
                "snapshot_year_index",
                annual_year_count - 1 if annual_year_count else 0,
            )
        )
        maximum_year_index = annual_year_count - 1 if annual_year_count else 0
        if not 0 <= snapshot_year_index <= maximum_year_index:
            raise ValueError("snapshot year index lies outside the run")
        for step_index in range(step_count):
            year_index = step_index // annual_steps_per_year
            step_in_year = step_index % annual_steps_per_year + 1
            if year_index != snapshot_year_index:
                continue
            if (
                step_in_year % int(snapshot_stride) == 0
                or step_in_year == annual_steps_per_year
            ):
                snapshot_step_indices.append(step_index)
    snapshot_mars_year_indices = [
        step_index // annual_steps_per_year
        for step_index in snapshot_step_indices
    ]
    snapshot_step_in_year = [
        step_index % annual_steps_per_year + 1
        for step_index in snapshot_step_indices
    ]
    snapshot_row_by_step = {
        step_index: row for row, step_index in enumerate(snapshot_step_indices)
    }
    control_ids = torch.tensor(
        controls,
        dtype=torch.int64,
        device=device,
    )

    aggregate = {
        "processed_primary_scenarios": 0,
        "baseline_frosting_scenarios": 0,
        "shadow_frosting_scenarios": 0,
        "paired_changed_scenarios": 0,
        "maximum_abs_baseline_energy_residual_j_m2": 0.0,
        "maximum_abs_shadow_energy_residual_j_m2": 0.0,
        "maximum_abs_baseline_endpoint_energy_residual_j_m2": 0.0,
        "maximum_abs_shadow_endpoint_energy_residual_j_m2": 0.0,
        "maximum_abs_baseline_energy_roundoff_projection_j_m2": 0.0,
        "maximum_abs_shadow_energy_roundoff_projection_j_m2": 0.0,
        "maximum_baseline_mass_balance_error_kg_m2": 0.0,
        "maximum_shadow_mass_balance_error_kg_m2": 0.0,
        "maximum_abs_final_temperature_delta_k": 0.0,
        "maximum_abs_final_frost_delta_kg_m2": 0.0,
        "sum_final_temperature_delta_k": 0.0,
        "sum_final_frost_delta_kg_m2": 0.0,
        "baseline_direct_energy_j_m2_sum": 0.0,
        "blocked_direct_energy_j_m2_sum": 0.0,
        "daylight_samples": 0,
        "blocked_daylight_samples": 0,
    }
    maximum_stability_number = 0.0
    control_fingerprint: str | None = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    with torch.inference_mode():
        for chunk_index in range(chunks_to_run):
            local_offset = chunk_index * chunk_size
            primary_count = min(
                chunk_size,
                shard_scenarios - local_offset,
            )
            primary_ids = _scenario_ids_for_chunk(
                total_scenarios=total_scenarios,
                shard_index=shard_index,
                shard_count=shard_count,
                local_offset=local_offset,
                count=primary_count,
                device=device,
            )
            include_controls = chunk_index == 0
            scenario_ids = (
                torch.cat((primary_ids, control_ids))
                if include_controls
                else primary_ids
            )
            cell_index, decoded = _decode_scenario_ids(
                scenario_ids,
                valid_cell_count=valid_cell_count,
                config=config,
                dtype=dtype,
            )
            batch_size = scenario_ids.numel()
            normals = geometry["normals"][cell_index]
            horizons = geometry["horizons"][:, cell_index]
            scenario_index = torch.arange(batch_size, device=device)
            longitude_index = torch.div(
                scenario_ids,
                valid_cell_count,
                rounding_mode="floor",
            )
            for name in AXIS_NAMES[:-1]:
                longitude_index = torch.div(
                    longitude_index,
                    len(_axis_values(config, name)),
                    rounding_mode="floor",
                )
            longitude_index = torch.remainder(
                longitude_index,
                longitude_count,
            )
            material = prepare_b1_material(
                decoded["thermal_inertia_j_m2_k_sqrt_s"],
                volumetric_heat_capacity_j_m3_k=(
                    volumetric_heat_capacity
                ),
                layer_thickness_m=layer_thickness,
                surface_heat_capacity_j_m2_k=surface_heat_capacity,
                time_step_s=float(config["time_step_s"]),
            )
            maximum_stability_number = max(
                maximum_stability_number,
                material.maximum_stability_number,
            )
            initial_surface = float(
                config["initial_surface_temperature_k"]
            )
            initial_frost = float(
                config["initial_surface_frost_mass_kg_m2"]
            )
            initial_subsurface = float(
                config["initial_subsurface_temperature_k"]
            )
            if initial_state is None:
                baseline_surface = torch.full(
                    (batch_size,),
                    initial_surface,
                    dtype=dtype,
                    device=device,
                )
                shadow_surface = baseline_surface.clone()
                baseline_frost = torch.full(
                    (batch_size,),
                    initial_frost,
                    dtype=dtype,
                    device=device,
                )
                shadow_frost = baseline_frost.clone()
                baseline_subsurface = torch.full(
                    (batch_size, int(config["layer_count"])),
                    initial_subsurface,
                    dtype=dtype,
                    device=device,
                )
                shadow_subsurface = baseline_subsurface.clone()
            else:
                baseline_surface = initial_state[
                    "baseline_surface_temperature_k"
                ][scenario_ids].clone()
                shadow_surface = initial_state[
                    "shadow_surface_temperature_k"
                ][scenario_ids].clone()
                baseline_frost = initial_state[
                    "baseline_frost_mass_kg_m2"
                ][scenario_ids].clone()
                shadow_frost = initial_state[
                    "shadow_frost_mass_kg_m2"
                ][scenario_ids].clone()
                baseline_subsurface = initial_state[
                    "baseline_subsurface_temperature_k"
                ][scenario_ids].clone()
                shadow_subsurface = initial_state[
                    "shadow_subsurface_temperature_k"
                ][scenario_ids].clone()
            baseline_initial_frost = baseline_frost.clone()
            shadow_initial_frost = shadow_frost.clone()
            baseline_initial_energy = total_b1_absolute_energy_j_m2(
                baseline_surface,
                baseline_frost,
                baseline_subsurface,
                surface_heat_capacity_j_m2_k=surface_heat_capacity,
                latent_heat_j_kg=latent_heat,
                cell_heat_capacity_j_m2_k=(
                    material.cell_heat_capacity_j_m2_k
                ),
            )
            shadow_initial_energy = total_b1_absolute_energy_j_m2(
                shadow_surface,
                shadow_frost,
                shadow_subsurface,
                surface_heat_capacity_j_m2_k=surface_heat_capacity,
                latent_heat_j_kg=latent_heat,
                cell_heat_capacity_j_m2_k=(
                    material.cell_heat_capacity_j_m2_k
                ),
            )
            baseline_boundary_energy = torch.zeros_like(baseline_frost)
            shadow_boundary_energy = torch.zeros_like(shadow_frost)
            baseline_boundary_energy_compensation = torch.zeros_like(
                baseline_frost
            )
            shadow_boundary_energy_compensation = torch.zeros_like(
                shadow_frost
            )
            baseline_step_energy_error = torch.zeros_like(baseline_frost)
            shadow_step_energy_error = torch.zeros_like(shadow_frost)
            baseline_step_energy_error_compensation = torch.zeros_like(
                baseline_frost
            )
            shadow_step_energy_error_compensation = torch.zeros_like(
                shadow_frost
            )
            baseline_max_abs_energy_roundoff_projection = torch.zeros_like(
                baseline_frost
            )
            shadow_max_abs_energy_roundoff_projection = torch.zeros_like(
                shadow_frost
            )
            baseline_condensed = torch.zeros_like(baseline_frost)
            baseline_sublimated = torch.zeros_like(baseline_frost)
            shadow_condensed = torch.zeros_like(shadow_frost)
            shadow_sublimated = torch.zeros_like(shadow_frost)
            baseline_condensed_compensation = torch.zeros_like(
                baseline_frost
            )
            baseline_sublimated_compensation = torch.zeros_like(
                baseline_frost
            )
            shadow_condensed_compensation = torch.zeros_like(shadow_frost)
            shadow_sublimated_compensation = torch.zeros_like(shadow_frost)
            baseline_maximum_frost = baseline_frost.clone()
            shadow_maximum_frost = shadow_frost.clone()
            baseline_frost_steps = torch.zeros(
                (batch_size,),
                dtype=torch.int32,
                device=device,
            )
            shadow_frost_steps = baseline_frost_steps.clone()
            baseline_first_frost_step = torch.full(
                (batch_size,),
                -1,
                dtype=torch.int64,
                device=device,
            )
            shadow_first_frost_step = baseline_first_frost_step.clone()
            baseline_last_frost_step = baseline_first_frost_step.clone()
            shadow_last_frost_step = baseline_first_frost_step.clone()
            baseline_peak_sublimation_flux = torch.zeros_like(
                baseline_frost
            )
            shadow_peak_sublimation_flux = torch.zeros_like(shadow_frost)
            baseline_maximum_warming_rate_during_frost_loss = (
                torch.zeros_like(baseline_frost)
            )
            shadow_maximum_warming_rate_during_frost_loss = (
                torch.zeros_like(shadow_frost)
            )
            baseline_minimum_surface_temperature = baseline_surface.clone()
            shadow_minimum_surface_temperature = shadow_surface.clone()
            baseline_maximum_surface_temperature = baseline_surface.clone()
            shadow_maximum_surface_temperature = shadow_surface.clone()
            baseline_direct_energy = torch.zeros_like(baseline_frost)
            blocked_direct_energy = torch.zeros_like(baseline_frost)
            daylight_count = torch.zeros(
                (batch_size,),
                dtype=torch.int64,
                device=device,
            )
            blocked_daylight_count = daylight_count.clone()
            annual_state: dict[str, torch.Tensor] = {}
            snapshot_state: dict[str, torch.Tensor] = {}
            if snapshot_step_indices:
                baseline_surface_snapshots = torch.empty(
                    (len(snapshot_step_indices), batch_size),
                    dtype=dtype,
                    device=device,
                )
                shadow_surface_snapshots = torch.empty_like(
                    baseline_surface_snapshots
                )
            if annual_design is not None:
                baseline_year_maximum_frost = baseline_frost.clone()
                shadow_year_maximum_frost = shadow_frost.clone()
                baseline_year_frost_duration_s = torch.zeros_like(
                    baseline_frost
                )
                shadow_year_frost_duration_s = torch.zeros_like(shadow_frost)
                baseline_year_first_frost_step = (
                    baseline_first_frost_step.clone()
                )
                shadow_year_first_frost_step = (
                    shadow_first_frost_step.clone()
                )
                baseline_year_last_frost_step = (
                    baseline_last_frost_step.clone()
                )
                shadow_year_last_frost_step = shadow_last_frost_step.clone()
                baseline_year_peak_sublimation_flux = torch.zeros_like(
                    baseline_frost
                )
                shadow_year_peak_sublimation_flux = torch.zeros_like(
                    shadow_frost
                )
                baseline_year_cumulative_sublimation = torch.zeros_like(
                    baseline_frost
                )
                shadow_year_cumulative_sublimation = torch.zeros_like(
                    shadow_frost
                )
                baseline_year_maximum_warming_rate = torch.zeros_like(
                    baseline_frost
                )
                shadow_year_maximum_warming_rate = torch.zeros_like(
                    shadow_frost
                )
                baseline_year_minimum_surface_temperature = (
                    baseline_surface.clone()
                )
                shadow_year_minimum_surface_temperature = (
                    shadow_surface.clone()
                )
                baseline_year_maximum_surface_temperature = (
                    baseline_surface.clone()
                )
                shadow_year_maximum_surface_temperature = (
                    shadow_surface.clone()
                )
                annual_snapshots: dict[str, list[torch.Tensor]] = {
                    "baseline_annual_maximum_frost_mass_kg_m2": [],
                    "shadow_annual_maximum_frost_mass_kg_m2": [],
                    "baseline_annual_frost_duration_s": [],
                    "shadow_annual_frost_duration_s": [],
                    "baseline_year_end_surface_temperature_k": [],
                    "shadow_year_end_surface_temperature_k": [],
                    "baseline_year_end_frost_mass_kg_m2": [],
                    "shadow_year_end_frost_mass_kg_m2": [],
                    "baseline_year_end_subsurface_temperature_k": [],
                    "shadow_year_end_subsurface_temperature_k": [],
                    "baseline_annual_first_frost_step": [],
                    "shadow_annual_first_frost_step": [],
                    "baseline_annual_last_frost_step": [],
                    "shadow_annual_last_frost_step": [],
                    "baseline_annual_peak_sublimation_flux_kg_m2_s": [],
                    "shadow_annual_peak_sublimation_flux_kg_m2_s": [],
                    "baseline_annual_cumulative_sublimation_kg_m2": [],
                    "shadow_annual_cumulative_sublimation_kg_m2": [],
                    "baseline_annual_maximum_warming_rate_during_frost_loss_k_s": [],
                    "shadow_annual_maximum_warming_rate_during_frost_loss_k_s": [],
                    "baseline_annual_minimum_surface_temperature_k": [],
                    "shadow_annual_minimum_surface_temperature_k": [],
                    "baseline_annual_maximum_surface_temperature_k": [],
                    "shadow_annual_maximum_surface_temperature_k": [],
                }

            for step_index in range(step_count):
                surface_pressure = (
                    forcing["surface_pressure_pa"][
                        step_index,
                        longitude_index,
                    ]
                    if "surface_pressure_pa" in forcing
                    else decoded["surface_pressure_pa"]
                )
                frost_temperature = (
                    149.2
                    + 6.48
                    * torch.log(0.00135 * surface_pressure)
                )
                declination = forcing["declination_rad"][
                    step_index,
                    longitude_index,
                ]
                hour_angle = forcing["hour_angle_rad"][step_index]
                east, north, up = _sun_vector_enu(
                    latitude_rad,
                    declination,
                    hour_angle,
                )
                sun_up = up > 0
                incidence = torch.clamp(
                    normals[:, 0] * east
                    + normals[:, 1] * north
                    + normals[:, 2] * up,
                    min=0.0,
                    max=1.0,
                )
                incidence = torch.where(
                    sun_up,
                    incidence,
                    torch.zeros_like(incidence),
                )
                azimuth = torch.remainder(
                    torch.atan2(east, north),
                    2.0 * math.pi,
                )
                altitude = torch.asin(torch.clamp(up, -1.0, 1.0))
                horizon_index = torch.floor(
                    torch.remainder(
                        azimuth + 0.5 * horizon_bin_width,
                        2.0 * math.pi,
                    )
                    / horizon_bin_width
                ).to(torch.int64)
                selected_horizon = horizons[
                    horizon_index,
                    scenario_index,
                ]
                visible = sun_up & (altitude > selected_horizon)
                shadow_incidence = torch.where(
                    visible,
                    incidence,
                    torch.zeros_like(incidence),
                )
                clear_sky_solar_flux = forcing["solar_flux_w_m2"][
                    step_index,
                    longitude_index,
                ]
                if "horizontal_shortwave_w_m2" in forcing:
                    horizontal_shortwave = forcing[
                        "horizontal_shortwave_w_m2"
                    ][step_index, longitude_index]
                    clear_horizontal = (
                        clear_sky_solar_flux * torch.clamp(up, min=0.0)
                    )
                    attenuation = torch.where(
                        clear_horizontal > 0,
                        torch.clamp(
                            horizontal_shortwave
                            / torch.clamp(clear_horizontal, min=1e-12),
                            min=0.0,
                            max=1.0,
                        ),
                        torch.zeros_like(clear_horizontal),
                    )
                    solar_flux = clear_sky_solar_flux * attenuation
                else:
                    solar_flux = clear_sky_solar_flux
                step_duration = float(
                    forcing["step_duration_s"][step_index]
                )
                baseline_albedo = torch.where(
                    baseline_frost > frost_epsilon,
                    torch.full_like(
                        baseline_frost,
                        float(config["frost_albedo"]),
                    ),
                    decoded["substrate_albedo"],
                )
                shadow_albedo = torch.where(
                    shadow_frost > frost_epsilon,
                    torch.full_like(
                        shadow_frost,
                        float(config["frost_albedo"]),
                    ),
                    decoded["substrate_albedo"],
                )
                downwelling_longwave = (
                    forcing["downwelling_longwave_w_m2"][
                        step_index,
                        longitude_index,
                    ]
                    if "downwelling_longwave_w_m2" in forcing
                    else float(config["downwelling_longwave_w_m2"])
                )
                baseline_flux = (
                    (1.0 - baseline_albedo) * solar_flux * incidence
                    + downwelling_longwave
                    - float(config["emissivity"])
                    * STEFAN_BOLTZMANN_W_M2_K4
                    * baseline_surface**4
                )
                shadow_flux = (
                    (1.0 - shadow_albedo)
                    * solar_flux
                    * shadow_incidence
                    + downwelling_longwave
                    - float(config["emissivity"])
                    * STEFAN_BOLTZMANN_W_M2_K4
                    * shadow_surface**4
                )
                baseline_previous_surface = baseline_surface
                shadow_previous_surface = shadow_surface
                baseline_previous_frost = baseline_frost
                shadow_previous_frost = shadow_frost
                baseline_step = advance_b1_columns(
                    baseline_surface,
                    baseline_frost,
                    baseline_subsurface,
                    material=material,
                    time_step_s=step_duration,
                    layer_thickness_m=layer_thickness,
                    surface_heat_capacity_j_m2_k=surface_heat_capacity,
                    frost_temperature_k=frost_temperature,
                    latent_heat_j_kg=latent_heat,
                    external_surface_heat_flux_w_m2=baseline_flux,
                    bottom_downward_heat_flux_w_m2=bottom_flux,
                    conserve_stored_state_energy=(
                        conserve_stored_state_energy
                    ),
                )
                shadow_step = advance_b1_columns(
                    shadow_surface,
                    shadow_frost,
                    shadow_subsurface,
                    material=material,
                    time_step_s=step_duration,
                    layer_thickness_m=layer_thickness,
                    surface_heat_capacity_j_m2_k=surface_heat_capacity,
                    frost_temperature_k=frost_temperature,
                    latent_heat_j_kg=latent_heat,
                    external_surface_heat_flux_w_m2=shadow_flux,
                    bottom_downward_heat_flux_w_m2=bottom_flux,
                    conserve_stored_state_energy=(
                        conserve_stored_state_energy
                    ),
                )
                if annual_design is not None:
                    baseline_step_error = (
                        baseline_step.stored_state_energy_residual_j_m2
                    )
                    shadow_step_error = (
                        shadow_step.stored_state_energy_residual_j_m2
                    )
                    (
                        baseline_step_energy_error,
                        baseline_step_energy_error_compensation,
                    ) = _neumaier_add(
                        baseline_step_energy_error,
                        baseline_step_energy_error_compensation,
                        baseline_step_error,
                    )
                    (
                        shadow_step_energy_error,
                        shadow_step_energy_error_compensation,
                    ) = _neumaier_add(
                        shadow_step_energy_error,
                        shadow_step_energy_error_compensation,
                        shadow_step_error,
                    )
                    baseline_max_abs_energy_roundoff_projection = (
                        torch.maximum(
                            baseline_max_abs_energy_roundoff_projection,
                            baseline_step.energy_roundoff_projection_j_m2.abs(),
                        )
                    )
                    shadow_max_abs_energy_roundoff_projection = torch.maximum(
                        shadow_max_abs_energy_roundoff_projection,
                        shadow_step.energy_roundoff_projection_j_m2.abs(),
                    )
                baseline_surface = baseline_step.surface_temperature_k
                baseline_frost = baseline_step.frost_mass_kg_m2
                baseline_subsurface = (
                    baseline_step.subsurface_temperature_k
                )
                shadow_surface = shadow_step.surface_temperature_k
                shadow_frost = shadow_step.frost_mass_kg_m2
                shadow_subsurface = shadow_step.subsurface_temperature_k
                step_number = step_index + 1
                step_in_year = step_index % annual_steps_per_year + 1
                baseline_frost_present = baseline_frost > frost_epsilon
                shadow_frost_present = shadow_frost > frost_epsilon
                baseline_first_frost_step = torch.where(
                    (baseline_first_frost_step < 0) & baseline_frost_present,
                    torch.full_like(baseline_first_frost_step, step_number),
                    baseline_first_frost_step,
                )
                shadow_first_frost_step = torch.where(
                    (shadow_first_frost_step < 0) & shadow_frost_present,
                    torch.full_like(shadow_first_frost_step, step_number),
                    shadow_first_frost_step,
                )
                baseline_last_frost_step = torch.where(
                    baseline_frost_present,
                    torch.full_like(baseline_last_frost_step, step_number),
                    baseline_last_frost_step,
                )
                shadow_last_frost_step = torch.where(
                    shadow_frost_present,
                    torch.full_like(shadow_last_frost_step, step_number),
                    shadow_last_frost_step,
                )
                baseline_sublimation_flux = (
                    baseline_step.sublimated_kg_m2 / step_duration
                )
                shadow_sublimation_flux = (
                    shadow_step.sublimated_kg_m2 / step_duration
                )
                baseline_peak_sublimation_flux = torch.maximum(
                    baseline_peak_sublimation_flux,
                    baseline_sublimation_flux,
                )
                shadow_peak_sublimation_flux = torch.maximum(
                    shadow_peak_sublimation_flux,
                    shadow_sublimation_flux,
                )
                baseline_warming_rate = torch.clamp(
                    (baseline_surface - baseline_previous_surface)
                    / step_duration,
                    min=0.0,
                )
                shadow_warming_rate = torch.clamp(
                    (shadow_surface - shadow_previous_surface) / step_duration,
                    min=0.0,
                )
                baseline_frost_loss = (
                    baseline_previous_frost > baseline_frost
                ) & (baseline_previous_frost > frost_epsilon)
                shadow_frost_loss = (
                    shadow_previous_frost > shadow_frost
                ) & (shadow_previous_frost > frost_epsilon)
                baseline_maximum_warming_rate_during_frost_loss = (
                    torch.maximum(
                        baseline_maximum_warming_rate_during_frost_loss,
                        torch.where(
                            baseline_frost_loss,
                            baseline_warming_rate,
                            torch.zeros_like(baseline_warming_rate),
                        ),
                    )
                )
                shadow_maximum_warming_rate_during_frost_loss = torch.maximum(
                    shadow_maximum_warming_rate_during_frost_loss,
                    torch.where(
                        shadow_frost_loss,
                        shadow_warming_rate,
                        torch.zeros_like(shadow_warming_rate),
                    ),
                )
                baseline_minimum_surface_temperature = torch.minimum(
                    baseline_minimum_surface_temperature,
                    baseline_surface,
                )
                shadow_minimum_surface_temperature = torch.minimum(
                    shadow_minimum_surface_temperature,
                    shadow_surface,
                )
                baseline_maximum_surface_temperature = torch.maximum(
                    baseline_maximum_surface_temperature,
                    baseline_surface,
                )
                shadow_maximum_surface_temperature = torch.maximum(
                    shadow_maximum_surface_temperature,
                    shadow_surface,
                )
                snapshot_row = snapshot_row_by_step.get(step_index)
                if snapshot_row is not None:
                    baseline_surface_snapshots[snapshot_row] = baseline_surface
                    shadow_surface_snapshots[snapshot_row] = shadow_surface
                (
                    baseline_boundary_energy,
                    baseline_boundary_energy_compensation,
                ) = _neumaier_add(
                    baseline_boundary_energy,
                    baseline_boundary_energy_compensation,
                    step_duration * (baseline_flux - bottom_flux),
                )
                (
                    shadow_boundary_energy,
                    shadow_boundary_energy_compensation,
                ) = _neumaier_add(
                    shadow_boundary_energy,
                    shadow_boundary_energy_compensation,
                    step_duration * (shadow_flux - bottom_flux),
                )
                if annual_design is not None:
                    baseline_year_first_frost_step = torch.where(
                        (baseline_year_first_frost_step < 0)
                        & baseline_frost_present,
                        torch.full_like(
                            baseline_year_first_frost_step,
                            step_in_year,
                        ),
                        baseline_year_first_frost_step,
                    )
                    shadow_year_first_frost_step = torch.where(
                        (shadow_year_first_frost_step < 0)
                        & shadow_frost_present,
                        torch.full_like(
                            shadow_year_first_frost_step,
                            step_in_year,
                        ),
                        shadow_year_first_frost_step,
                    )
                    baseline_year_last_frost_step = torch.where(
                        baseline_frost_present,
                        torch.full_like(
                            baseline_year_last_frost_step,
                            step_in_year,
                        ),
                        baseline_year_last_frost_step,
                    )
                    shadow_year_last_frost_step = torch.where(
                        shadow_frost_present,
                        torch.full_like(
                            shadow_year_last_frost_step,
                            step_in_year,
                        ),
                        shadow_year_last_frost_step,
                    )
                    baseline_year_peak_sublimation_flux = torch.maximum(
                        baseline_year_peak_sublimation_flux,
                        baseline_sublimation_flux,
                    )
                    shadow_year_peak_sublimation_flux = torch.maximum(
                        shadow_year_peak_sublimation_flux,
                        shadow_sublimation_flux,
                    )
                    baseline_year_cumulative_sublimation += (
                        baseline_step.sublimated_kg_m2
                    )
                    shadow_year_cumulative_sublimation += (
                        shadow_step.sublimated_kg_m2
                    )
                    baseline_year_maximum_warming_rate = torch.maximum(
                        baseline_year_maximum_warming_rate,
                        torch.where(
                            baseline_frost_loss,
                            baseline_warming_rate,
                            torch.zeros_like(baseline_warming_rate),
                        ),
                    )
                    shadow_year_maximum_warming_rate = torch.maximum(
                        shadow_year_maximum_warming_rate,
                        torch.where(
                            shadow_frost_loss,
                            shadow_warming_rate,
                            torch.zeros_like(shadow_warming_rate),
                        ),
                    )
                    baseline_year_minimum_surface_temperature = torch.minimum(
                        baseline_year_minimum_surface_temperature,
                        baseline_surface,
                    )
                    shadow_year_minimum_surface_temperature = torch.minimum(
                        shadow_year_minimum_surface_temperature,
                        shadow_surface,
                    )
                    baseline_year_maximum_surface_temperature = torch.maximum(
                        baseline_year_maximum_surface_temperature,
                        baseline_surface,
                    )
                    shadow_year_maximum_surface_temperature = torch.maximum(
                        shadow_year_maximum_surface_temperature,
                        shadow_surface,
                    )
                    (
                        baseline_condensed,
                        baseline_condensed_compensation,
                    ) = _neumaier_add(
                        baseline_condensed,
                        baseline_condensed_compensation,
                        baseline_step.condensed_kg_m2,
                    )
                    (
                        baseline_sublimated,
                        baseline_sublimated_compensation,
                    ) = _neumaier_add(
                        baseline_sublimated,
                        baseline_sublimated_compensation,
                        baseline_step.sublimated_kg_m2,
                    )
                    (
                        shadow_condensed,
                        shadow_condensed_compensation,
                    ) = _neumaier_add(
                        shadow_condensed,
                        shadow_condensed_compensation,
                        shadow_step.condensed_kg_m2,
                    )
                    (
                        shadow_sublimated,
                        shadow_sublimated_compensation,
                    ) = _neumaier_add(
                        shadow_sublimated,
                        shadow_sublimated_compensation,
                        shadow_step.sublimated_kg_m2,
                    )
                else:
                    baseline_condensed += baseline_step.condensed_kg_m2
                    baseline_sublimated += baseline_step.sublimated_kg_m2
                    shadow_condensed += shadow_step.condensed_kg_m2
                    shadow_sublimated += shadow_step.sublimated_kg_m2
                baseline_maximum_frost = torch.maximum(
                    baseline_maximum_frost,
                    baseline_frost,
                )
                shadow_maximum_frost = torch.maximum(
                    shadow_maximum_frost,
                    shadow_frost,
                )
                baseline_frost_steps += (
                    baseline_frost > frost_epsilon
                ).to(torch.int32)
                shadow_frost_steps += (
                    shadow_frost > frost_epsilon
                ).to(torch.int32)
                if annual_design is not None:
                    baseline_year_maximum_frost = torch.maximum(
                        baseline_year_maximum_frost,
                        baseline_frost,
                    )
                    shadow_year_maximum_frost = torch.maximum(
                        shadow_year_maximum_frost,
                        shadow_frost,
                    )
                    baseline_year_frost_duration_s += (
                        step_duration
                        * (baseline_frost > frost_epsilon).to(dtype)
                    )
                    shadow_year_frost_duration_s += (
                        step_duration
                        * (shadow_frost > frost_epsilon).to(dtype)
                    )
                    if (step_index + 1) % annual_steps_per_year == 0:
                        annual_snapshots[
                            "baseline_annual_maximum_frost_mass_kg_m2"
                        ].append(baseline_year_maximum_frost.clone())
                        annual_snapshots[
                            "shadow_annual_maximum_frost_mass_kg_m2"
                        ].append(shadow_year_maximum_frost.clone())
                        annual_snapshots[
                            "baseline_annual_frost_duration_s"
                        ].append(baseline_year_frost_duration_s.clone())
                        annual_snapshots[
                            "shadow_annual_frost_duration_s"
                        ].append(shadow_year_frost_duration_s.clone())
                        annual_snapshots[
                            "baseline_year_end_surface_temperature_k"
                        ].append(baseline_surface.clone())
                        annual_snapshots[
                            "shadow_year_end_surface_temperature_k"
                        ].append(shadow_surface.clone())
                        annual_snapshots[
                            "baseline_year_end_frost_mass_kg_m2"
                        ].append(baseline_frost.clone())
                        annual_snapshots[
                            "shadow_year_end_frost_mass_kg_m2"
                        ].append(shadow_frost.clone())
                        annual_snapshots[
                            "baseline_year_end_subsurface_temperature_k"
                        ].append(baseline_subsurface.clone())
                        annual_snapshots[
                            "shadow_year_end_subsurface_temperature_k"
                        ].append(shadow_subsurface.clone())
                        annual_snapshots[
                            "baseline_annual_first_frost_step"
                        ].append(baseline_year_first_frost_step.clone())
                        annual_snapshots[
                            "shadow_annual_first_frost_step"
                        ].append(shadow_year_first_frost_step.clone())
                        annual_snapshots[
                            "baseline_annual_last_frost_step"
                        ].append(baseline_year_last_frost_step.clone())
                        annual_snapshots[
                            "shadow_annual_last_frost_step"
                        ].append(shadow_year_last_frost_step.clone())
                        annual_snapshots[
                            "baseline_annual_peak_sublimation_flux_kg_m2_s"
                        ].append(baseline_year_peak_sublimation_flux.clone())
                        annual_snapshots[
                            "shadow_annual_peak_sublimation_flux_kg_m2_s"
                        ].append(shadow_year_peak_sublimation_flux.clone())
                        annual_snapshots[
                            "baseline_annual_cumulative_sublimation_kg_m2"
                        ].append(baseline_year_cumulative_sublimation.clone())
                        annual_snapshots[
                            "shadow_annual_cumulative_sublimation_kg_m2"
                        ].append(shadow_year_cumulative_sublimation.clone())
                        annual_snapshots[
                            "baseline_annual_maximum_warming_rate_during_frost_loss_k_s"
                        ].append(baseline_year_maximum_warming_rate.clone())
                        annual_snapshots[
                            "shadow_annual_maximum_warming_rate_during_frost_loss_k_s"
                        ].append(shadow_year_maximum_warming_rate.clone())
                        annual_snapshots[
                            "baseline_annual_minimum_surface_temperature_k"
                        ].append(
                            baseline_year_minimum_surface_temperature.clone()
                        )
                        annual_snapshots[
                            "shadow_annual_minimum_surface_temperature_k"
                        ].append(
                            shadow_year_minimum_surface_temperature.clone()
                        )
                        annual_snapshots[
                            "baseline_annual_maximum_surface_temperature_k"
                        ].append(
                            baseline_year_maximum_surface_temperature.clone()
                        )
                        annual_snapshots[
                            "shadow_annual_maximum_surface_temperature_k"
                        ].append(
                            shadow_year_maximum_surface_temperature.clone()
                        )
                        baseline_year_maximum_frost = baseline_frost.clone()
                        shadow_year_maximum_frost = shadow_frost.clone()
                        baseline_year_frost_duration_s.zero_()
                        shadow_year_frost_duration_s.zero_()
                        baseline_year_first_frost_step.fill_(-1)
                        shadow_year_first_frost_step.fill_(-1)
                        baseline_year_last_frost_step.fill_(-1)
                        shadow_year_last_frost_step.fill_(-1)
                        baseline_year_peak_sublimation_flux.zero_()
                        shadow_year_peak_sublimation_flux.zero_()
                        baseline_year_cumulative_sublimation.zero_()
                        shadow_year_cumulative_sublimation.zero_()
                        baseline_year_maximum_warming_rate.zero_()
                        shadow_year_maximum_warming_rate.zero_()
                        baseline_year_minimum_surface_temperature = (
                            baseline_surface.clone()
                        )
                        shadow_year_minimum_surface_temperature = (
                            shadow_surface.clone()
                        )
                        baseline_year_maximum_surface_temperature = (
                            baseline_surface.clone()
                        )
                        shadow_year_maximum_surface_temperature = (
                            shadow_surface.clone()
                        )
                baseline_direct_energy += (
                    step_duration * solar_flux * incidence
                )
                blocked_direct_energy += (
                    step_duration
                    * solar_flux
                    * (incidence - shadow_incidence)
                )
                daylight_count += sun_up.to(torch.int64)
                blocked_daylight_count += (
                    sun_up & (incidence > 0) & ~visible
                ).to(torch.int64)

                if (
                    step_number % progress_every == 0
                    or step_number == step_count
                ):
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    completed_units = (
                        local_offset * step_count
                        + primary_count * step_number
                    )
                    target_units = (
                        shard_scenarios * formal_step_count
                        if pilot_steps is None
                        else min(
                            shard_scenarios,
                            chunks_to_run * chunk_size,
                        )
                        * step_count
                    )
                    fraction = min(1.0, completed_units / target_units)
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "mode": mode,
                                "shard_index": shard_index,
                                "chunk_index": chunk_index,
                                "step": step_number,
                                "steps_per_chunk": step_count,
                                "mars_year_index": (
                                    step_index // annual_steps_per_year
                                )
                                + 1,
                                "step_in_mars_year": (
                                    step_index % annual_steps_per_year
                                )
                                + 1,
                                "overall_fraction": fraction,
                                "elapsed_s": elapsed,
                                "eta_s": (
                                    elapsed * (1.0 - fraction) / fraction
                                    if fraction > 0
                                    else None
                                ),
                                "shadow_frost_fraction_current_batch": float(
                                    (shadow_frost > frost_epsilon)
                                    .to(dtype)
                                    .mean()
                                ),
                                "peak_memory_bytes": (
                                    torch.cuda.max_memory_allocated(device)
                                    if device.type == "cuda"
                                    else 0
                                ),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

            if annual_design is not None:
                completed_years = len(
                    annual_snapshots[
                        "baseline_annual_maximum_frost_mass_kg_m2"
                    ]
                )
                expected_years = (
                    annual_year_count if pilot_steps is None else completed_years
                )
                if pilot_steps is None and completed_years != expected_years:
                    raise RuntimeError(
                        "formal annual run did not produce every checkpoint"
                    )
                if completed_years:
                    annual_state = {
                        name: torch.stack(values, dim=0)
                        for name, values in annual_snapshots.items()
                    }
            if snapshot_step_indices:
                snapshot_state = {
                    "snapshot_step_index_zero_based": torch.tensor(
                        snapshot_step_indices,
                        dtype=torch.int64,
                    ),
                    "snapshot_mars_year_index_zero_based": torch.tensor(
                        snapshot_mars_year_indices,
                        dtype=torch.int64,
                    ),
                    "snapshot_step_in_year": torch.tensor(
                        snapshot_step_in_year,
                        dtype=torch.int64,
                    ),
                    "baseline_surface_temperature_snapshots_k": (
                        baseline_surface_snapshots
                    ),
                    "shadow_surface_temperature_snapshots_k": (
                        shadow_surface_snapshots
                    ),
                }

            baseline_final_energy = total_b1_absolute_energy_j_m2(
                baseline_surface,
                baseline_frost,
                baseline_subsurface,
                surface_heat_capacity_j_m2_k=surface_heat_capacity,
                latent_heat_j_kg=latent_heat,
                cell_heat_capacity_j_m2_k=(
                    material.cell_heat_capacity_j_m2_k
                ),
            )
            shadow_final_energy = total_b1_absolute_energy_j_m2(
                shadow_surface,
                shadow_frost,
                shadow_subsurface,
                surface_heat_capacity_j_m2_k=surface_heat_capacity,
                latent_heat_j_kg=latent_heat,
                cell_heat_capacity_j_m2_k=(
                    material.cell_heat_capacity_j_m2_k
                ),
            )
            baseline_endpoint_energy_error = (
                baseline_final_energy
                - baseline_initial_energy
                - (
                    baseline_boundary_energy
                    + baseline_boundary_energy_compensation
                )
            )
            shadow_endpoint_energy_error = (
                shadow_final_energy
                - shadow_initial_energy
                - (
                    shadow_boundary_energy
                    + shadow_boundary_energy_compensation
                )
            )
            baseline_energy_error = (
                baseline_step_energy_error
                + baseline_step_energy_error_compensation
                if annual_design is not None
                else baseline_endpoint_energy_error
            )
            shadow_energy_error = (
                shadow_step_energy_error
                + shadow_step_energy_error_compensation
                if annual_design is not None
                else shadow_endpoint_energy_error
            )
            baseline_mass_error = (
                baseline_condensed
                + baseline_condensed_compensation
                - baseline_sublimated
                - baseline_sublimated_compensation
                - baseline_frost
                + baseline_initial_frost
            )
            shadow_mass_error = (
                shadow_condensed
                + shadow_condensed_compensation
                - shadow_sublimated
                - shadow_sublimated_compensation
                - shadow_frost
                + shadow_initial_frost
            )
            primary = slice(0, primary_count)
            temperature_delta = (
                shadow_surface[primary] - baseline_surface[primary]
            )
            frost_delta = shadow_frost[primary] - baseline_frost[primary]
            changed = (
                temperature_delta.abs()
                > float(config["paired_change_temperature_epsilon_k"])
            ) | (
                frost_delta.abs()
                > float(config["paired_change_frost_epsilon_kg_m2"])
            )
            aggregate["processed_primary_scenarios"] += primary_count
            aggregate["baseline_frosting_scenarios"] += int(
                (baseline_maximum_frost[primary] > frost_epsilon).sum()
            )
            aggregate["shadow_frosting_scenarios"] += int(
                (shadow_maximum_frost[primary] > frost_epsilon).sum()
            )
            aggregate["paired_changed_scenarios"] += int(changed.sum())
            aggregate[
                "maximum_abs_baseline_energy_residual_j_m2"
            ] = max(
                aggregate[
                    "maximum_abs_baseline_energy_residual_j_m2"
                ],
                float(baseline_energy_error[primary].abs().max()),
            )
            aggregate["maximum_abs_shadow_energy_residual_j_m2"] = max(
                aggregate["maximum_abs_shadow_energy_residual_j_m2"],
                float(shadow_energy_error[primary].abs().max()),
            )
            aggregate[
                "maximum_abs_baseline_endpoint_energy_residual_j_m2"
            ] = max(
                aggregate[
                    "maximum_abs_baseline_endpoint_energy_residual_j_m2"
                ],
                float(
                    baseline_endpoint_energy_error[primary].abs().max()
                ),
            )
            aggregate[
                "maximum_abs_shadow_endpoint_energy_residual_j_m2"
            ] = max(
                aggregate[
                    "maximum_abs_shadow_endpoint_energy_residual_j_m2"
                ],
                float(shadow_endpoint_energy_error[primary].abs().max()),
            )
            aggregate[
                "maximum_abs_baseline_energy_roundoff_projection_j_m2"
            ] = max(
                aggregate[
                    "maximum_abs_baseline_energy_roundoff_projection_j_m2"
                ],
                float(
                    baseline_max_abs_energy_roundoff_projection[
                        primary
                    ].max()
                ),
            )
            aggregate[
                "maximum_abs_shadow_energy_roundoff_projection_j_m2"
            ] = max(
                aggregate[
                    "maximum_abs_shadow_energy_roundoff_projection_j_m2"
                ],
                float(
                    shadow_max_abs_energy_roundoff_projection[primary].max()
                ),
            )
            aggregate[
                "maximum_baseline_mass_balance_error_kg_m2"
            ] = max(
                aggregate[
                    "maximum_baseline_mass_balance_error_kg_m2"
                ],
                float(baseline_mass_error[primary].abs().max()),
            )
            aggregate["maximum_shadow_mass_balance_error_kg_m2"] = max(
                aggregate["maximum_shadow_mass_balance_error_kg_m2"],
                float(shadow_mass_error[primary].abs().max()),
            )
            aggregate["maximum_abs_final_temperature_delta_k"] = max(
                aggregate["maximum_abs_final_temperature_delta_k"],
                float(temperature_delta.abs().max()),
            )
            aggregate["maximum_abs_final_frost_delta_kg_m2"] = max(
                aggregate["maximum_abs_final_frost_delta_kg_m2"],
                float(frost_delta.abs().max()),
            )
            aggregate["sum_final_temperature_delta_k"] += float(
                temperature_delta.sum()
            )
            aggregate["sum_final_frost_delta_kg_m2"] += float(
                frost_delta.sum()
            )
            aggregate["baseline_direct_energy_j_m2_sum"] += float(
                baseline_direct_energy[primary].sum()
            )
            aggregate["blocked_direct_energy_j_m2_sum"] += float(
                blocked_direct_energy[primary].sum()
            )
            aggregate["daylight_samples"] += int(
                daylight_count[primary].sum()
            )
            aggregate["blocked_daylight_samples"] += int(
                blocked_daylight_count[primary].sum()
            )

            if include_controls:
                control_slice = slice(primary_count, batch_size)
                control_fingerprint = _fingerprint_controls(
                    scenario_ids[control_slice],
                    (
                        baseline_surface[control_slice],
                        shadow_surface[control_slice],
                        baseline_frost[control_slice],
                        shadow_frost[control_slice],
                        baseline_maximum_frost[control_slice],
                        shadow_maximum_frost[control_slice],
                    ),
                )
                if state_dir is not None:
                    control_annual_state = {
                        name: value[:, control_slice].cpu()
                        for name, value in annual_state.items()
                    }
                    control_snapshot_state = {
                        name: (
                            value[:, control_slice].cpu()
                            if value.ndim == 2
                            and value.shape[1] == batch_size
                            else value.cpu()
                        )
                        for name, value in snapshot_state.items()
                    }
                    _write_torch_payload(
                        state_dir / "controls.pt",
                        {
                            "schema_version": 1,
                            "scenario_id": scenario_ids[
                                control_slice
                            ].cpu(),
                            "baseline_final_surface_temperature_k": (
                                baseline_surface[control_slice].cpu()
                            ),
                            "shadow_final_surface_temperature_k": (
                                shadow_surface[control_slice].cpu()
                            ),
                            "baseline_final_frost_mass_kg_m2": (
                                baseline_frost[control_slice].cpu()
                            ),
                            "shadow_final_frost_mass_kg_m2": (
                                shadow_frost[control_slice].cpu()
                            ),
                            "numerical_fingerprint_sha256": (
                                control_fingerprint
                            ),
                            **control_annual_state,
                            **control_snapshot_state,
                        },
                        record_sha256=record_sha256,
                    )
            if state_dir is not None:
                state_path = state_dir / f"chunk-{chunk_index:05d}.pt"
                primary_annual_state = {
                    name: value[:, primary].cpu()
                    for name, value in annual_state.items()
                }
                primary_snapshot_state = {
                    name: (
                        value[:, primary].cpu()
                        if value.ndim == 2 and value.shape[1] == batch_size
                        else value.cpu()
                    )
                    for name, value in snapshot_state.items()
                }
                state_sha = _write_torch_payload(
                    state_path,
                    {
                        "schema_version": 1,
                        "scenario_id": primary_ids.cpu(),
                        "baseline_final_surface_temperature_k": (
                            baseline_surface[primary].cpu()
                        ),
                        "shadow_final_surface_temperature_k": (
                            shadow_surface[primary].cpu()
                        ),
                        "baseline_final_frost_mass_kg_m2": (
                            baseline_frost[primary].cpu()
                        ),
                        "shadow_final_frost_mass_kg_m2": (
                            shadow_frost[primary].cpu()
                        ),
                        "baseline_maximum_frost_mass_kg_m2": (
                            baseline_maximum_frost[primary].cpu()
                        ),
                        "shadow_maximum_frost_mass_kg_m2": (
                            shadow_maximum_frost[primary].cpu()
                        ),
                        "baseline_frost_present_steps": (
                            baseline_frost_steps[primary].cpu()
                        ),
                        "shadow_frost_present_steps": (
                            shadow_frost_steps[primary].cpu()
                        ),
                        "baseline_first_frost_step": (
                            baseline_first_frost_step[primary].cpu()
                        ),
                        "shadow_first_frost_step": (
                            shadow_first_frost_step[primary].cpu()
                        ),
                        "baseline_last_frost_step": (
                            baseline_last_frost_step[primary].cpu()
                        ),
                        "shadow_last_frost_step": (
                            shadow_last_frost_step[primary].cpu()
                        ),
                        "baseline_peak_sublimation_flux_kg_m2_s": (
                            baseline_peak_sublimation_flux[primary].cpu()
                        ),
                        "shadow_peak_sublimation_flux_kg_m2_s": (
                            shadow_peak_sublimation_flux[primary].cpu()
                        ),
                        "baseline_cumulative_sublimation_kg_m2": (
                            (
                                baseline_sublimated
                                + baseline_sublimated_compensation
                            )[primary].cpu()
                        ),
                        "shadow_cumulative_sublimation_kg_m2": (
                            (
                                shadow_sublimated
                                + shadow_sublimated_compensation
                            )[primary].cpu()
                        ),
                        "baseline_maximum_warming_rate_during_frost_loss_k_s": (
                            baseline_maximum_warming_rate_during_frost_loss[
                                primary
                            ].cpu()
                        ),
                        "shadow_maximum_warming_rate_during_frost_loss_k_s": (
                            shadow_maximum_warming_rate_during_frost_loss[
                                primary
                            ].cpu()
                        ),
                        "baseline_minimum_surface_temperature_k": (
                            baseline_minimum_surface_temperature[primary].cpu()
                        ),
                        "shadow_minimum_surface_temperature_k": (
                            shadow_minimum_surface_temperature[primary].cpu()
                        ),
                        "baseline_maximum_surface_temperature_k": (
                            baseline_maximum_surface_temperature[primary].cpu()
                        ),
                        "shadow_maximum_surface_temperature_k": (
                            shadow_maximum_surface_temperature[primary].cpu()
                        ),
                        "baseline_direct_energy_j_m2": (
                            baseline_direct_energy[primary].cpu()
                        ),
                        "blocked_direct_energy_j_m2": (
                            blocked_direct_energy[primary].cpu()
                        ),
                        "daylight_samples": daylight_count[primary].cpu(),
                        "blocked_daylight_samples": (
                            blocked_daylight_count[primary].cpu()
                        ),
                        "formal_feature_schema_version": 1,
                        **primary_annual_state,
                        **primary_snapshot_state,
                    },
                    record_sha256=record_sha256,
                )
                if chunks_manifest is None:
                    raise RuntimeError("chunk manifest disappeared")
                chunk_record = {
                    "chunk_index": chunk_index,
                    "path": str(state_path),
                    "size_bytes": state_path.stat().st_size,
                    "primary_scenarios": primary_count,
                    "scenario_id_min": int(primary_ids.min()),
                    "scenario_id_max": int(primary_ids.max()),
                }
                if state_sha is not None:
                    chunk_record["sha256"] = state_sha
                _append_json_line(chunks_manifest, chunk_record)
            print(
                json.dumps(
                    {
                        "event": "chunk_complete",
                        "mode": mode,
                        "shard_index": shard_index,
                        "chunk_index": chunk_index,
                        "primary_scenarios": primary_count,
                        "control_fingerprint_sha256": (
                            control_fingerprint if include_controls else None
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - started
    processed = int(aggregate["processed_primary_scenarios"])
    baseline_direct_energy_sum = float(
        aggregate["baseline_direct_energy_j_m2_sum"]
    )
    blocked_direct_energy_fraction = (
        float(aggregate["blocked_direct_energy_j_m2_sum"])
        / baseline_direct_energy_sum
        if baseline_direct_energy_sum > 0
        else 0.0
    )
    finite_summary = all(
        math.isfinite(float(value)) for value in aggregate.values()
    )
    passed = (
        maximum_stability_number <= 1.0
        and finite_summary
        and aggregate["maximum_abs_baseline_energy_residual_j_m2"]
        <= float(config["maximum_energy_residual_tolerance_j_m2"])
        and aggregate["maximum_abs_shadow_energy_residual_j_m2"]
        <= float(config["maximum_energy_residual_tolerance_j_m2"])
        and aggregate["maximum_baseline_mass_balance_error_kg_m2"]
        <= float(config["maximum_mass_balance_tolerance_kg_m2"])
        and aggregate["maximum_shadow_mass_balance_error_kg_m2"]
        <= float(config["maximum_mass_balance_tolerance_kg_m2"])
        and control_fingerprint is not None
        and blocked_direct_energy_fraction
        >= float(config["minimum_blocked_direct_energy_fraction"])
    )
    if pilot_steps is None:
        passed = (
            passed
            and processed == shard_scenarios
            and aggregate["paired_changed_scenarios"] > 0
        )
    scenario_steps_per_s = processed * step_count / elapsed_s
    projected_formal_elapsed_s = (
        shard_scenarios * formal_step_count / scenario_steps_per_s
    )
    peak_memory_bytes = 0
    device_metadata: dict[str, Any] = {
        "requested_device": device_name,
        "logical_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        device_metadata.update(
            {
                "name": properties.name,
                "uuid": str(properties.uuid),
                "total_memory_bytes": properties.total_memory,
                "peak_memory_bytes": peak_memory_bytes,
            }
        )
    summary = {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "dtm_grid_shape": list(geometry["shape"]),
        "valid_dtm_cells": valid_cell_count,
        "parameter_combinations": _parameter_combination_count(config),
        "total_unique_scenarios": total_scenarios,
        "formal_primary_scenarios_in_shard": shard_scenarios,
        "processed_primary_scenarios": processed,
        "formal_steps": formal_step_count,
        "executed_steps_per_chunk": step_count,
        "formal_sol_count": float(config["sol_count"]),
        "annual_cycle_design": annual_design,
        "annual_steps_per_year": annual_steps_per_year,
        "snapshot_stride_steps": snapshot_stride,
        "snapshot_step_indices_zero_based": snapshot_step_indices,
        "snapshot_count": len(snapshot_step_indices),
        "snapshot_state_timing": "post_step",
        "energy_residual_accounting": (
            "compensated_sum_of_post_projection_per_step_stored_state_energy_residuals"
            if annual_design is not None and conserve_stored_state_energy
            else "compensated_sum_of_per_step_state_energy_changes"
            if annual_design is not None
            else "endpoint_state_change_minus_compensated_boundary_energy"
        ),
        "conserve_stored_state_energy": conserve_stored_state_energy,
        "energy_roundoff_projection_target": (
            "deepest_subsurface_control_volume"
            if conserve_stored_state_energy
            else None
        ),
        "mass_balance_accounting": (
            "compensated_condensation_and_sublimation_sums"
            if annual_design is not None
            else "direct_condensation_and_sublimation_sums"
        ),
        "time_step_s": float(config["time_step_s"]),
        "horizon_azimuth_count": int(config["horizon_azimuth_count"]),
        "horizon_maximum_distance_cells": int(
            config["horizon_maximum_distance_cells"]
        ),
        "chunks_in_formal_shard": total_chunks,
        "executed_chunks": chunks_to_run,
        "maximum_stability_number": maximum_stability_number,
        "blocked_direct_energy_fraction": blocked_direct_energy_fraction,
        "paired_changed_scenario_fraction": (
            aggregate["paired_changed_scenarios"] / processed
            if processed
            else 0.0
        ),
        "mean_final_temperature_delta_k": (
            aggregate["sum_final_temperature_delta_k"] / processed
            if processed
            else 0.0
        ),
        "mean_final_frost_delta_kg_m2": (
            aggregate["sum_final_frost_delta_kg_m2"] / processed
            if processed
            else 0.0
        ),
        **aggregate,
        "control_scenario_ids": controls,
        "control_numerical_fingerprint_sha256": control_fingerprint,
        "elapsed_s": elapsed_s,
        "scenario_steps_per_s": scenario_steps_per_s,
        "projected_formal_elapsed_s": projected_formal_elapsed_s,
        "projected_formal_gpu_h": projected_formal_elapsed_s / 3600.0,
        "peak_memory_bytes": peak_memory_bytes,
        "acquisition_alignment": _acquisition_alignment_summary(config),
        "forcing": _forcing_summary(config, forcing),
    }
    dtm_record = {
        "path": str(dtm_path),
        "size_bytes": dtm_path.stat().st_size,
        "product_id": geometry["metadata"]["product_id"],
        "source_url": geometry["metadata"]["source_url"],
        "coordinate_system": geometry["metadata"]["coordinate_system"],
    }
    if record_sha256:
        dtm_record["sha256"] = _sha256(dtm_path)
    initial_state_record = None
    if config.get("initial_state") is not None:
        initial_state_path = Path(config["initial_state"]["path"])
        initial_state_record = {
            "id": str(config["initial_state"].get("id", "UNSPECIFIED")),
            "path": str(initial_state_path),
            "size_bytes": initial_state_path.stat().st_size,
        }
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "mode": mode,
        "model_scope": config.get(
            "model_scope",
            "official_Gasa_DTM_local_normals_and_projected_shadow_paired_B1",
        ),
        "claim_boundary": config.get(
            "claim_boundary",
            (
                "real Gasa DTM geometry engineering evidence only; synthetic "
                "radiative, material, pressure and seasonal forcing; no frozen "
                "THEMIS or frost product, atmosphere retrieval, activity label, "
                "event causality or gully-activity inference"
            ),
        ),
        "dtm": dtm_record,
        "initial_state": initial_state_record,
        "device": device_metadata,
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--chunks-manifest", type=Path)
    parser.add_argument("--pilot-steps", type=int)
    parser.add_argument("--pilot-chunks", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {args.output}"
        )
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    result = run(
        config,
        device_name=args.device,
        shard_index=args.shard_index,
        state_dir=args.state_dir,
        chunks_manifest=args.chunks_manifest,
        pilot_steps=args.pilot_steps,
        pilot_chunks=args.pilot_chunks,
    )
    if bool(config.get("record_sha256", True)):
        result["config_sha256"] = _sha256(args.config)
    else:
        result["config"] = {
            "path": str(args.config),
            "size_bytes": args.config.stat().st_size,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], sort_keys=True), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
