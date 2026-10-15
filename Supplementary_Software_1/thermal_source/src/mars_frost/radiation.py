"""Minimal radiative surface-energy operator with analytic equilibrium."""

from __future__ import annotations

import math


STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8


def _validate(
    *,
    surface_temperature_k: float,
    albedo: float,
    solar_flux_w_m2: float,
    incidence_cosine: float,
    downwelling_longwave_w_m2: float,
    emissivity: float,
) -> None:
    values = (
        surface_temperature_k,
        albedo,
        solar_flux_w_m2,
        incidence_cosine,
        downwelling_longwave_w_m2,
        emissivity,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all radiative inputs must be finite")
    if surface_temperature_k <= 0:
        raise ValueError("surface temperature must be positive")
    if not 0 <= albedo <= 1:
        raise ValueError("albedo must lie in [0, 1]")
    if solar_flux_w_m2 < 0:
        raise ValueError("solar flux must be non-negative")
    if not 0 <= incidence_cosine <= 1:
        raise ValueError("incidence cosine must lie in [0, 1]")
    if downwelling_longwave_w_m2 < 0:
        raise ValueError("downwelling longwave must be non-negative")
    if not 0 < emissivity <= 1:
        raise ValueError("emissivity must lie in (0, 1]")


def absorbed_shortwave_w_m2(
    *,
    albedo: float,
    solar_flux_w_m2: float,
    incidence_cosine: float,
) -> float:
    _validate(
        surface_temperature_k=1.0,
        albedo=albedo,
        solar_flux_w_m2=solar_flux_w_m2,
        incidence_cosine=incidence_cosine,
        downwelling_longwave_w_m2=0.0,
        emissivity=1.0,
    )
    return (1.0 - albedo) * solar_flux_w_m2 * incidence_cosine


def emitted_longwave_w_m2(
    *,
    surface_temperature_k: float,
    emissivity: float,
) -> float:
    _validate(
        surface_temperature_k=surface_temperature_k,
        albedo=0.0,
        solar_flux_w_m2=0.0,
        incidence_cosine=0.0,
        downwelling_longwave_w_m2=0.0,
        emissivity=emissivity,
    )
    return (
        emissivity
        * STEFAN_BOLTZMANN_W_M2_K4
        * surface_temperature_k**4
    )


def net_radiative_heat_flux_w_m2(
    *,
    surface_temperature_k: float,
    albedo: float,
    solar_flux_w_m2: float,
    incidence_cosine: float,
    downwelling_longwave_w_m2: float,
    emissivity: float,
) -> float:
    """Return positive-downward net radiative energy entering the surface."""

    _validate(
        surface_temperature_k=surface_temperature_k,
        albedo=albedo,
        solar_flux_w_m2=solar_flux_w_m2,
        incidence_cosine=incidence_cosine,
        downwelling_longwave_w_m2=downwelling_longwave_w_m2,
        emissivity=emissivity,
    )
    return (
        (1.0 - albedo) * solar_flux_w_m2 * incidence_cosine
        + downwelling_longwave_w_m2
        - emissivity
        * STEFAN_BOLTZMANN_W_M2_K4
        * surface_temperature_k**4
    )


def radiative_equilibrium_temperature_k(
    *,
    albedo: float,
    solar_flux_w_m2: float,
    incidence_cosine: float,
    downwelling_longwave_w_m2: float,
    emissivity: float,
) -> float:
    """Return the temperature where incoming and emitted radiation balance."""

    _validate(
        surface_temperature_k=1.0,
        albedo=albedo,
        solar_flux_w_m2=solar_flux_w_m2,
        incidence_cosine=incidence_cosine,
        downwelling_longwave_w_m2=downwelling_longwave_w_m2,
        emissivity=emissivity,
    )
    incoming = (
        (1.0 - albedo) * solar_flux_w_m2 * incidence_cosine
        + downwelling_longwave_w_m2
    )
    if incoming <= 0:
        raise ValueError("radiative equilibrium requires positive incoming flux")
    return (
        incoming / (emissivity * STEFAN_BOLTZMANN_W_M2_K4)
    ) ** 0.25
