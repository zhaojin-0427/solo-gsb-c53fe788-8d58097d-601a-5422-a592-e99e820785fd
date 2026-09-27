"""P-wave first-motion focal mechanism grid search (pure math).

A double-couple mechanism is described by fault strike / dip / rake in the
Aki & Richards convention.  For a candidate mechanism the P radiation
amplitude at a station is

    A_P = (u . r) (n . r)

with ``n`` the fault normal, ``u`` the slip vector and ``r`` the unit ray
direction from the source towards the receiver, all expressed in a local
(north, east, down) frame.  The sign of ``A_P`` is the predicted first
motion: positive -> compression ("positive" polarity), negative ->
dilatation ("negative").  A (numerically) zero amplitude sits on a nodal
plane and always counts as a misfit, whatever the observed polarity.

Strike / dip / rake space is searched exhaustively on a regular grid
(default 10 degrees).  Two grid mechanisms that differ only by swapping
the two nodal planes are the same physical object; they are identified
and de-duplicated through the canonical moment-tensor key (the moment
tensor is invariant under both the plane swap and the (n, u) -> (-n, -u)
ambiguity, but NOT under slip reversal, which flips every polarity).
All de-duplicated mechanisms attaining the minimum weighted misfit are
returned in a stable, fully deterministic order.
"""
from __future__ import annotations

import math

from .association import distance_km
from .location import KM_PER_DEG_LAT

GRID_STEP_DEG = 10.0
ZERO_AMPLITUDE_EPS = 1e-9


# --------------------------------------------------------------------------- #
# fault geometry
# --------------------------------------------------------------------------- #
def fault_vectors(strike_deg: float, dip_deg: float, rake_deg: float):
    """(normal, slip) unit vectors of the fault in (north, east, down)."""
    phi = math.radians(strike_deg)
    delta = math.radians(dip_deg)
    lam = math.radians(rake_deg)
    s = (math.cos(phi), math.sin(phi), 0.0)  # along strike
    d = (  # down-dip
        -math.sin(phi) * math.cos(delta),
        math.cos(phi) * math.cos(delta),
        math.sin(delta),
    )
    n = (
        math.sin(phi) * math.sin(delta),
        -math.cos(phi) * math.sin(delta),
        math.cos(delta),
    )
    u = (
        math.cos(lam) * s[0] + math.sin(lam) * d[0],
        math.cos(lam) * s[1] + math.sin(lam) * d[1],
        math.cos(lam) * s[2] + math.sin(lam) * d[2],
    )
    return n, u


def vectors_to_plane(n, u) -> tuple[float, float, float]:
    """Canonical (strike, dip, rake) of the plane given by normal/slip.

    The (n, u) -> (-n, -u) ambiguity is resolved by choosing the normal
    with a non-negative down component (dip in [0, 90]); rake is returned
    in [-180, 180).
    """
    nx, ny, nz = n
    ux, uy, uz = u
    if nz < 0.0:
        nx, ny, nz = -nx, -ny, -nz
        ux, uy, uz = -ux, -uy, -uz
    horizontal = math.hypot(nx, ny)
    dip = math.degrees(math.atan2(horizontal, nz))
    if horizontal < 1e-12:
        strike = 0.0  # horizontal plane: strike is undefined, fix it
    else:
        strike = math.degrees(math.atan2(nx, -ny)) % 360.0
    phi = math.radians(strike)
    delta = math.radians(dip)
    s = (math.cos(phi), math.sin(phi), 0.0)
    d = (
        -math.sin(phi) * math.cos(delta),
        math.cos(phi) * math.cos(delta),
        math.sin(delta),
    )
    rake = math.degrees(
        math.atan2(
            ux * d[0] + uy * d[1] + uz * d[2],
            ux * s[0] + uy * s[1] + uz * s[2],
        )
    )
    rake = (rake + 180.0) % 360.0 - 180.0
    return strike, dip, rake


def auxiliary_plane(
    strike_deg: float, dip_deg: float, rake_deg: float
) -> tuple[float, float, float]:
    """The second nodal plane of a double-couple mechanism."""
    n, u = fault_vectors(strike_deg, dip_deg, rake_deg)
    return vectors_to_plane(u, n)


def moment_key(strike_deg: float, dip_deg: float, rake_deg: float) -> tuple:
    """Canonical key of the double-couple moment tensor.

    Invariant under nodal-plane swap and the (n, u) -> (-n, -u) ambiguity,
    so equivalent grid mechanisms share a key; slip reversal (rake + 180)
    yields the negated tensor and a different key.
    """
    n, u = fault_vectors(strike_deg, dip_deg, rake_deg)
    m = (
        2.0 * u[0] * n[0],
        2.0 * u[1] * n[1],
        2.0 * u[2] * n[2],
        u[0] * n[1] + u[1] * n[0],
        u[0] * n[2] + u[2] * n[0],
        u[1] * n[2] + u[2] * n[1],
    )
    scale = max(abs(c) for c in m)  # a double couple never vanishes
    return tuple(round(c / scale, 6) for c in m)


# --------------------------------------------------------------------------- #
# station geometry (azimuth / takeoff from the layered model)
# --------------------------------------------------------------------------- #
def ray_vector(azimuth_deg: float, takeoff_deg: float):
    """Unit ray from source to receiver in (north, east, down).

    ``takeoff_deg`` is measured from the downward vertical, so upgoing
    rays to surface stations have takeoff angles in [90, 180].
    """
    az = math.radians(azimuth_deg)
    to = math.radians(takeoff_deg)
    return (
        math.sin(to) * math.cos(az),
        math.sin(to) * math.sin(az),
        math.cos(to),
    )


def station_geometry(
    ev_lat: float,
    ev_lon: float,
    depth_km: float,
    stations: dict[str, tuple[float, float, float]],
    model,
) -> dict[str, tuple[float, float, float]]:
    """Per-station (azimuth_deg, takeoff_deg, distance_km) from the event.

    Azimuth is measured clockwise from north with the same equirectangular
    approximation as the locator.  The takeoff angle comes from the ray
    parameter of the run's layered velocity model (station elevation
    shifts the effective source depth, exactly as in the locator).
    """
    geometry = {}
    for code in sorted(stations):
        lat, lon, elev_m = stations[code]
        dist = distance_km(ev_lat, ev_lon, lat, lon)
        km_per_rad = KM_PER_DEG_LAT * 180.0 / math.pi
        dx = (
            math.radians(lon - ev_lon)
            * math.cos(math.radians(0.5 * (ev_lat + lat)))
            * km_per_rad
        )
        dy = math.radians(lat - ev_lat) * km_per_rad
        azimuth = math.degrees(math.atan2(dx, dy)) % 360.0
        z_eff = depth_km + elev_m / 1000.0
        _, p = model.travel_time(z_eff, dist, "P")
        v_src = model.velocity_at_depth(z_eff, "P")
        sin_theta = min(1.0, max(0.0, p * v_src))
        takeoff = 180.0 - math.degrees(math.asin(sin_theta))
        geometry[code] = (azimuth, takeoff, dist)
    return geometry


def max_azimuthal_gap(azimuths: list[float]) -> float:
    """Largest gap (degrees) between consecutive station azimuths."""
    if len(azimuths) < 2:
        return 360.0
    ordered = sorted(a % 360.0 for a in azimuths)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    gaps.append(ordered[0] + 360.0 - ordered[-1])
    return max(gaps)


# --------------------------------------------------------------------------- #
# polarity misfit evaluation
# --------------------------------------------------------------------------- #
def p_amplitude(n, u, ray) -> float:
    """P first-motion amplitude (u.r)(n.r); its sign is the polarity."""
    un = u[0] * ray[0] + u[1] * ray[1] + u[2] * ray[2]
    nn = n[0] * ray[0] + n[1] * ray[1] + n[2] * ray[2]
    return un * nn


def predicted_sign(amplitude: float, eps: float = ZERO_AMPLITUDE_EPS) -> int:
    """+1 / -1 for a clear polarity, 0 on (numerically) a nodal plane."""
    if amplitude > eps:
        return 1
    if amplitude < -eps:
        return -1
    return 0


def evaluate(
    strike_deg: float,
    dip_deg: float,
    rake_deg: float,
    observations: list[dict],
    eps: float = ZERO_AMPLITUDE_EPS,
) -> dict:
    """Weighted polarity misfit of one mechanism against the observations.

    Each observation carries ``azimuth_deg``, ``takeoff_deg``, ``sign``
    (+1/-1) and ``weight``.  A zero predicted amplitude always counts as a
    misfit.  Returns the candidate record with both nodal planes and the
    per-station predictions.
    """
    n, u = fault_vectors(strike_deg, dip_deg, rake_deg)
    rays = [ray_vector(o["azimuth_deg"], o["takeoff_deg"]) for o in observations]
    return _candidate(
        (strike_deg, dip_deg, rake_deg), n, u, rays, observations, eps
    )


def _candidate(plane1, n, u, rays, observations, eps) -> dict:
    misfit_weight = 0.0
    misfit_count = 0
    predictions = []
    amplitudes = []
    for ray, obs in zip(rays, observations):
        amp = p_amplitude(n, u, ray)
        sign = predicted_sign(amp, eps)
        predictions.append(sign)
        amplitudes.append(amp)
        if sign != obs["sign"]:  # a zero amplitude never matches +/-1
            misfit_weight += obs["weight"]
            misfit_count += 1
    plane2 = vectors_to_plane(u, n)
    planes = sorted([plane1, plane2])
    return {
        "planes": planes,
        "misfit_weight": misfit_weight,
        "misfit_count": misfit_count,
        "predictions": predictions,
        "amplitudes": amplitudes,
    }


# --------------------------------------------------------------------------- #
# exhaustive grid search
# --------------------------------------------------------------------------- #
def grid_points(step_deg: float = GRID_STEP_DEG):
    """All (strike, dip, rake) grid nodes: strike [0,360), dip [0,90],
    rake [-180,180), each at ``step_deg`` spacing."""
    n_turn = max(1, round(360.0 / step_deg))
    n_dip = max(1, round(90.0 / step_deg))
    for i in range(n_turn):
        strike = i * step_deg
        for j in range(n_dip + 1):
            dip = j * step_deg
            for k in range(n_turn):
                rake = -180.0 + k * step_deg
                yield strike, dip, rake


def grid_search(
    observations: list[dict],
    step_deg: float = GRID_STEP_DEG,
    eps: float = ZERO_AMPLITUDE_EPS,
) -> tuple[float, list[dict], int]:
    """Exhaustive strike/dip/rake grid search over unique mechanisms.

    Returns (best_misfit_weight, solutions, evaluated_count) where
    ``solutions`` are the de-duplicated mechanisms tied at the minimum
    weighted misfit, stably ordered by their canonical plane pair, and
    ``evaluated_count`` is the number of distinct mechanisms evaluated.
    """
    rays = [ray_vector(o["azimuth_deg"], o["takeoff_deg"]) for o in observations]
    unique: dict[tuple, dict] = {}
    for strike, dip, rake in grid_points(step_deg):
        key = moment_key(strike, dip, rake)
        if key in unique:
            continue  # same mechanism already evaluated via its other plane
        n, u = fault_vectors(strike, dip, rake)
        candidate = _candidate((strike, dip, rake), n, u, rays, observations, eps)
        candidate["_key"] = key
        unique[key] = candidate
    best = min(c["misfit_weight"] for c in unique.values())
    # Ties are decided with a small tolerance: equal nominal misfits built
    # from different weight subsets may differ in the last float bit.
    solutions = [
        c for c in unique.values() if c["misfit_weight"] <= best + 1e-9
    ]
    solutions.sort(
        key=lambda c: (
            round(c["misfit_weight"], 9),
            *c["planes"][0],
            *c["planes"][1],
            c["_key"],
        )
    )
    return best, solutions, len(unique)
