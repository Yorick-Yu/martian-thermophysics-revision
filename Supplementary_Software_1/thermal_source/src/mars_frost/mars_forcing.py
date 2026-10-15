"""Deterministic Mars orbital forcing and CO2 frost-point utilities.

The functions in this module are intentionally limited to clear-sky geometric
forcing.  They do not represent dust, atmospheric scattering, topographic
shadowing, or a Gasa-specific atmosphere.
"""

from __future__ import annotations

import math


SOLAR_CONSTANT_1_AU_W_M2 = 1361.0
MARS_SEMIMAJOR_AXIS_AU = 1.523679
MARS_ECCENTRICITY = 0.0934
MARS_OBLIQUITY_DEG = 25.19
MARS_PERIHELION_SOLAR_LONGITUDE_DEG = 251.0
MARS_SOL_SECONDS = 88775.244
MARS_YEAR_SOLS = 668.5991
MARS_ORBITAL_PERIOD_S = MARS_SOL_SECONDS * MARS_YEAR_SOLS


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def co2_frost_temperature_k(surface_pressure_pa: float) -> float:
    """Return the empirical CO2 frost temperature for a surface pressure.

    The approximation is ``149.2 + 6.48 ln(0.00135 p)`` with pressure in Pa.
    It is restricted here to 100--2000 Pa, which comfortably brackets the
    synthetic Mars surface-pressure grid used by the rebuild benchmark.
    """

    _require_finite("surface pressure", surface_pressure_pa)
    if not 100.0 <= surface_pressure_pa <= 2000.0:
        raise ValueError("surface pressure must lie in [100, 2000] Pa")
    return 149.2 + 6.48 * math.log(0.00135 * surface_pressure_pa)


def heliocentric_distance_au(solar_longitude_deg: float) -> float:
    """Return Mars--Sun distance for areocentric solar longitude ``Ls``."""

    _require_finite("solar longitude", solar_longitude_deg)
    true_anomaly = math.radians(
        solar_longitude_deg - MARS_PERIHELION_SOLAR_LONGITUDE_DEG
    )
    return (
        MARS_SEMIMAJOR_AXIS_AU
        * (1.0 - MARS_ECCENTRICITY**2)
        / (1.0 + MARS_ECCENTRICITY * math.cos(true_anomaly))
    )


def solar_flux_w_m2(solar_longitude_deg: float) -> float:
    """Return top-of-atmosphere direct solar flux at Mars."""

    distance_au = heliocentric_distance_au(solar_longitude_deg)
    return SOLAR_CONSTANT_1_AU_W_M2 / distance_au**2


def solar_declination_rad(solar_longitude_deg: float) -> float:
    """Return solar declination for a fixed present-day Mars obliquity."""

    _require_finite("solar longitude", solar_longitude_deg)
    longitude = math.radians(solar_longitude_deg)
    obliquity = math.radians(MARS_OBLIQUITY_DEG)
    return math.asin(math.sin(obliquity) * math.sin(longitude))


def surface_incidence_cosine(
    *,
    latitude_deg: float,
    slope_deg: float,
    aspect_deg: float,
    local_solar_time_hours: float,
    solar_longitude_deg: float,
) -> float:
    """Return the non-negative direct-beam cosine on a planar slope.

    Aspect is clockwise from north and denotes the direction faced by the
    slope.  Local solar time is in the half-open interval ``[0, 24)``.
    """

    values = (
        latitude_deg,
        slope_deg,
        aspect_deg,
        local_solar_time_hours,
        solar_longitude_deg,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all incidence inputs must be finite")
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude must lie in [-90, 90] degrees")
    if not 0.0 <= slope_deg <= 90.0:
        raise ValueError("slope must lie in [0, 90] degrees")
    if not 0.0 <= local_solar_time_hours < 24.0:
        raise ValueError("local solar time must lie in [0, 24) hours")

    latitude = math.radians(latitude_deg)
    slope = math.radians(slope_deg)
    aspect = math.radians(aspect_deg)
    declination = solar_declination_rad(solar_longitude_deg)
    hour_angle = math.radians(15.0 * (local_solar_time_hours - 12.0))

    sun_east = -math.cos(declination) * math.sin(hour_angle)
    sun_north = (
        math.cos(latitude) * math.sin(declination)
        - math.sin(latitude)
        * math.cos(declination)
        * math.cos(hour_angle)
    )
    sun_up = (
        math.sin(latitude) * math.sin(declination)
        + math.cos(latitude)
        * math.cos(declination)
        * math.cos(hour_angle)
    )
    normal_east = math.sin(slope) * math.sin(aspect)
    normal_north = math.sin(slope) * math.cos(aspect)
    normal_up = math.cos(slope)
    cosine = (
        normal_east * sun_east
        + normal_north * sun_north
        + normal_up * sun_up
    )
    return min(1.0, max(0.0, cosine))


def solar_longitude_after_elapsed_s(
    initial_solar_longitude_deg: float,
    elapsed_s: float,
) -> float:
    """Advance ``Ls`` with a Keplerian present-day Mars orbit."""

    values = (initial_solar_longitude_deg, elapsed_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("solar longitude and elapsed time must be finite")
    if elapsed_s < 0:
        raise ValueError("elapsed time must be non-negative")

    eccentricity = MARS_ECCENTRICITY
    true_anomaly = math.radians(
        initial_solar_longitude_deg
        - MARS_PERIHELION_SOLAR_LONGITUDE_DEG
    )
    eccentric_anomaly = math.atan2(
        math.sqrt(1.0 - eccentricity**2) * math.sin(true_anomaly),
        eccentricity + math.cos(true_anomaly),
    )
    mean_anomaly = (
        eccentric_anomaly
        - eccentricity * math.sin(eccentric_anomaly)
        + 2.0 * math.pi * elapsed_s / MARS_ORBITAL_PERIOD_S
    )
    eccentric_anomaly = mean_anomaly
    for _ in range(8):
        residual = (
            eccentric_anomaly
            - eccentricity * math.sin(eccentric_anomaly)
            - mean_anomaly
        )
        derivative = 1.0 - eccentricity * math.cos(eccentric_anomaly)
        eccentric_anomaly -= residual / derivative

    true_anomaly = math.atan2(
        math.sqrt(1.0 - eccentricity**2)
        * math.sin(eccentric_anomaly),
        math.cos(eccentric_anomaly) - eccentricity,
    )
    return (
        math.degrees(true_anomaly)
        + MARS_PERIHELION_SOLAR_LONGITUDE_DEG
    ) % 360.0


def initial_solar_longitude_for_target(
    target_solar_longitude_deg: float,
    elapsed_s: float,
) -> float:
    """Return the initial ``Ls`` that reaches a target after ``elapsed_s``."""

    values = (target_solar_longitude_deg, elapsed_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("solar longitude and elapsed time must be finite")
    if elapsed_s < 0:
        raise ValueError("elapsed time must be non-negative")
    reverse_elapsed_s = (-elapsed_s) % MARS_ORBITAL_PERIOD_S
    return solar_longitude_after_elapsed_s(
        target_solar_longitude_deg,
        reverse_elapsed_s,
    )
