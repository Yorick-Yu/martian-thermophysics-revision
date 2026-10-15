"""Read and interpolate frozen Mars Climate Database web outputs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence


_ROW_PATTERN = re.compile(r"^\+?([0-9]+(?:\.[0-9]+)?)\s+\|\|\s+(.+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_mcd_ascii(
    text: str,
    *,
    variable_codes: Sequence[str],
) -> dict[str, Any]:
    """Parse a point ``Ls x local-time`` MCD ASCII response."""

    lines = [line.rstrip() for line in text.splitlines()]
    description_indices = [
        index
        for index, line in enumerate(lines)
        if line.startswith("### Columns 2+ are ")
    ]
    blocks: list[dict[str, Any]] = []
    for block_number, description_index in enumerate(description_indices):
        description_line = lines[description_index]
        description = description_line.removeprefix(
            "### Columns 2+ are "
        )
        header_index = next(
            (
                index
                for index, line in enumerate(
                    lines[description_index + 1 :],
                    start=description_index + 1,
                )
                if line.startswith("---- ||")
            ),
            None,
        )
        if header_index is None:
            raise ValueError("MCD block is missing the local-time header")
        local_times = [
            float(value)
            for value in lines[header_index].split("||", 1)[1].split()
        ]
        solar_longitudes: list[float] = []
        rows: list[list[float]] = []
        next_description_index = (
            description_indices[block_number + 1]
            if block_number + 1 < len(description_indices)
            else len(lines)
        )
        for line in lines[header_index + 1 : next_description_index]:
            match = _ROW_PATTERN.match(line)
            if match is None:
                continue
            solar_longitudes.append(float(match.group(1)))
            rows.append(
                [float(value) for value in match.group(2).split()]
            )
        if not rows:
            raise ValueError("MCD block contains no numeric rows")
        if any(len(row) != len(local_times) for row in rows):
            raise ValueError("MCD block has inconsistent row lengths")
        blocks.append(
            {
                "description": description,
                "solar_longitude_deg": solar_longitudes,
                "local_time_hours": local_times,
                "values": rows,
            }
        )

    if len(blocks) != len(variable_codes):
        raise ValueError(
            "MCD response block count does not match requested variables"
        )
    reference_ls = blocks[0]["solar_longitude_deg"]
    reference_lt = blocks[0]["local_time_hours"]
    if reference_ls != [float(value) for value in range(0, 361, 15)]:
        raise ValueError("MCD solar-longitude grid is not 0..360 by 15")
    if reference_lt != [float(value) for value in range(25)]:
        raise ValueError("MCD local-time grid is not 0..24 by 1")

    variables: dict[str, Any] = {}
    for code, block in zip(variable_codes, blocks):
        if block["solar_longitude_deg"] != reference_ls:
            raise ValueError("MCD variables use different Ls grids")
        if block["local_time_hours"] != reference_lt:
            raise ValueError("MCD variables use different local-time grids")
        values = block["values"]
        for row in values:
            if not math.isclose(
                row[0],
                row[-1],
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError("MCD LT=0 and LT=24 columns differ")
        variables[code] = {
            "description": block["description"],
            "values": [row[:24] for row in values],
        }
    return {
        "solar_longitude_deg": reference_ls,
        "local_time_hours": reference_lt[:24],
        "variables": variables,
    }


def load_mcd_forcing(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_sha256 is not None:
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise ValueError("MCD forcing SHA256 does not match config")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported MCD forcing schema")
    if payload.get("mcd_version") != "6.1":
        raise ValueError("the frozen forcing must use MCD version 6.1")
    grid = payload.get("grid", {})
    if grid.get("solar_longitude_deg") != [
        float(value) for value in range(0, 361, 15)
    ]:
        raise ValueError("MCD forcing has an unexpected Ls grid")
    if grid.get("local_time_hours") != [
        float(value) for value in range(24)
    ]:
        raise ValueError("MCD forcing has an unexpected local-time grid")
    required = {
        "ps",
        "t",
        "tauref",
        "fluxsurf_lw",
        "fluxsurf_dn_sw",
        "thermal_inertia",
        "ground_albedo",
    }
    if not required.issubset(payload.get("variables", {})):
        raise ValueError("MCD forcing is missing a required variable")
    return payload


def interpolate_mcd_variable(
    payload: dict[str, Any],
    variable_code: str,
    *,
    solar_longitude_deg: float,
    local_time_hours: float,
) -> float:
    """Periodic bilinear interpolation on the frozen 15-degree/hour grid."""

    if not math.isfinite(solar_longitude_deg) or not math.isfinite(
        local_time_hours
    ):
        raise ValueError("MCD interpolation coordinates must be finite")
    try:
        values = payload["variables"][variable_code]["values"]
    except KeyError as exc:
        raise KeyError(f"unknown MCD variable: {variable_code}") from exc
    if len(values) != 25 or any(len(row) != 24 for row in values):
        raise ValueError("MCD variable must have a 25x24 seasonal grid")

    ls_position = (solar_longitude_deg % 360.0) / 15.0
    lt_position = local_time_hours % 24.0
    ls0 = math.floor(ls_position)
    lt0 = math.floor(lt_position)
    ls1 = ls0 + 1
    lt1 = (lt0 + 1) % 24
    ls_weight = ls_position - ls0
    lt_weight = lt_position - lt0
    low = (
        (1.0 - lt_weight) * float(values[ls0][lt0])
        + lt_weight * float(values[ls0][lt1])
    )
    high = (
        (1.0 - lt_weight) * float(values[ls1][lt0])
        + lt_weight * float(values[ls1][lt1])
    )
    return (1.0 - ls_weight) * low + ls_weight * high
