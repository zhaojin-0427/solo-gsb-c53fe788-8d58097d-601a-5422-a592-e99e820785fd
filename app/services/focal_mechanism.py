"""P-wave first-motion focal mechanism inversion (analysis layer).

For one event of a completed run the client submits the event's associated
P picks with observed polarities (positive/negative) and weights.  The
service freezes the event hypocenter, the run's layered velocity model and
the station geometry, computes ray azimuths and takeoff angles, then
grid-searches double-couple strike/dip/rake on a 10-degree grid for the
mechanisms with the minimum weighted polarity misfit (a zero radiation
amplitude counts as a misfit).  Mechanisms equivalent under nodal-plane
swap are de-duplicated; every tied best solution reports both nodal
planes, its misfit weight and the per-station observed/predicted
polarities, and the tied set is stably paginated.

Requests are explicitly rejected (422, nothing persisted) when a pick is
not an associated P pick of the event, when one station carries
contradictory polarities, when fewer than six independent stations remain,
or when the largest azimuthal gap is not below 180 degrees.

Like the station impact audit this is a read-only analysis layer: it never
writes runs/events/picks and never touches catalog content.  Inversions
are immutable and idempotent per (event, model version, normalized
polarity set); concurrent identical requests collapse onto one report.
"""
from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import config
from ..models import (
    Event,
    EventPick,
    FocalMechanism,
    FocalMechanismSolution,
    Pick,
    Run,
    StationCalibration,
    VelocityModel,
    Waveform,
)
from ..schemas import FocalMechanismCreate
from . import focal
from .pipeline import ApiError, canonical_layers, iso_z
from .traveltime import LayeredModel

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 200


# --------------------------------------------------------------------------- #
# inversion creation (validate -> freeze -> grid search -> persist, idempotent)
# --------------------------------------------------------------------------- #
def create_mechanism(db: Session, body: FocalMechanismCreate) -> tuple[dict, bool]:
    event = db.get(Event, body.event_id)
    if event is None:
        raise ApiError(404, "event_not_found", f"event {body.event_id} not found")
    run = db.get(Run, event.run_id)
    if run is None:  # pragma: no cover - FK integrity makes this unreachable
        raise ApiError(404, "run_not_found", f"run {event.run_id} not found")
    model_row = db.get(VelocityModel, run.model_id)
    model = LayeredModel.from_dicts(canonical_layers(model_row.layers))

    # The event's associated P picks (pick_id -> station).
    p_rows = (
        db.query(EventPick, Pick)
        .join(Pick, EventPick.pick_id == Pick.id)
        .filter(EventPick.event_id == event.id, EventPick.phase == "P")
        .all()
    )
    p_pick_station = {p.id: p.station for ep, p in p_rows}

    observations = _normalize_observations(db, body, p_pick_station)
    if len(observations) < config.FM_MIN_STATIONS:
        raise ApiError(
            422,
            "insufficient_polarity_stations",
            f"focal mechanism inversion needs at least "
            f"{config.FM_MIN_STATIONS} independent stations, got "
            f"{len(observations)}",
        )

    waveforms = db.query(Waveform).filter_by(ingestion_id=run.ingestion_id).all()
    stations = {w.station: (w.latitude, w.longitude, w.elevation_m) for w in waveforms}
    geometry = focal.station_geometry(
        event.latitude, event.longitude, event.depth_km, stations, model
    )
    gap = focal.max_azimuthal_gap(
        [geometry[o["station"]][0] for o in observations]
    )
    if gap >= config.FM_MAX_AZIMUTHAL_GAP_DEG:
        raise ApiError(
            422,
            "azimuthal_gap_too_large",
            f"largest azimuthal gap {gap:.1f} deg is not below "
            f"{config.FM_MAX_AZIMUTHAL_GAP_DEG:.0f} deg; the focal sphere is "
            "not covered well enough for a first-motion inversion",
        )

    # Idempotency: event + model version + normalized polarity set.
    canon_obs = [
        {"station": o["station"], "polarity": o["polarity"], "weight": o["weight"]}
        for o in observations
    ]
    canon_request = {
        "event_id": event.id,
        "model": {"name": model_row.name, "version": model_row.version},
        "observations": canon_obs,
    }
    request_hash = hashlib.sha256(
        json.dumps(canon_request, sort_keys=True).encode()
    ).hexdigest()
    existing = (
        db.query(FocalMechanism).filter_by(request_hash=request_hash).one_or_none()
    )
    if existing is not None:
        db.commit()
        return mechanism_payload(existing, created=False), False

    search_obs = [
        {
            "station": o["station"],
            "azimuth_deg": geometry[o["station"]][0],
            "takeoff_deg": geometry[o["station"]][1],
            "sign": 1 if o["polarity"] == "positive" else -1,
            "weight": o["weight"],
        }
        for o in observations
    ]
    best_weight, solutions, evaluated_count = focal.grid_search(
        search_obs,
        step_deg=config.FM_GRID_STEP_DEG,
        eps=config.FM_ZERO_AMPLITUDE_EPS,
    )

    calibration = None
    if run.calibration_id is not None:
        cal = db.get(StationCalibration, run.calibration_id)
        calibration = {"version": cal.calibration_version, "status": cal.status}

    frozen = _frozen_json(
        event, run, model_row, calibration, observations, stations, geometry
    )
    params_hash = _params_hash(run, model_row, calibration, canon_obs)

    mech = FocalMechanism(
        request_hash=request_hash,
        request=canon_request,
        params_hash=params_hash,
        event_id=event.id,
        run_id=run.id,
        model_id=model_row.id,
        frozen=frozen,
        station_count=len(observations),
        observation_count=len(body.observations),
        max_azimuthal_gap_deg=gap,
        evaluated_count=evaluated_count,
        min_misfit_weight=best_weight,
        min_misfit_count=min(s["misfit_count"] for s in solutions),
        solution_count=len(solutions),
    )
    db.add(mech)
    try:
        db.flush()
    except IntegrityError:
        # Concurrent identical request: collapse onto the surviving report.
        db.rollback()
        winner = db.query(FocalMechanism).filter_by(request_hash=request_hash).one()
        return mechanism_payload(winner, created=False), False

    for idx, sol in enumerate(solutions):
        planes = [_plane_json(p) for p in sol["planes"]]
        db.add(
            FocalMechanismSolution(
                mechanism_id=mech.id,
                sort_index=idx,
                strike_deg=planes[0]["strike_deg"],
                dip_deg=planes[0]["dip_deg"],
                rake_deg=planes[0]["rake_deg"],
                planes=planes,
                misfit_weight=sol["misfit_weight"],
                misfit_count=sol["misfit_count"],
                stations=_solution_stations_json(sol, search_obs),
            )
        )
    db.commit()
    db.refresh(mech)
    return mechanism_payload(mech, created=True), True


def _normalize_observations(
    db: Session, body: FocalMechanismCreate, p_pick_station: dict[str, str]
) -> list[dict]:
    """Validate the submitted picks and merge them per station.

    Every pick must be an associated P pick of the event; one station's
    picks must agree in polarity (their weights add up).  Returns the
    normalized per-station observations sorted by station code.
    """
    seen: set[str] = set()
    per_station: dict[str, dict] = {}
    for obs in body.observations:
        if obs.pick_id in seen:
            raise ApiError(
                422,
                "duplicate_pick",
                f"pick {obs.pick_id} is submitted more than once",
            )
        seen.add(obs.pick_id)
        station = p_pick_station.get(obs.pick_id)
        if station is None:
            if db.get(Pick, obs.pick_id) is None:
                raise ApiError(
                    404, "pick_not_found", f"pick {obs.pick_id} not found"
                )
            raise ApiError(
                422,
                "pick_not_associated",
                f"pick {obs.pick_id} is not an associated P pick of event "
                f"{body.event_id}",
            )
        entry = per_station.get(station)
        if entry is None:
            entry = {
                "station": station,
                "polarity": obs.polarity,
                "weight": 0.0,
                "pick_ids": [],
            }
            per_station[station] = entry
        elif entry["polarity"] != obs.polarity:
            raise ApiError(
                422,
                "contradictory_polarity",
                f"station {station} is submitted with both positive and "
                "negative polarity",
            )
        entry["weight"] += obs.weight
        entry["pick_ids"].append(obs.pick_id)
    return [per_station[s] for s in sorted(per_station)]


def _params_hash(
    run: Run, model_row: VelocityModel, calibration: dict | None, canon_obs: list
) -> str:
    """Canonical hash binding the report to run, versions and polarity set."""
    blob = {
        "run_id": run.id,
        "model": {"name": model_row.name, "version": model_row.version},
        "calibration": None if calibration is None else calibration["version"],
        "grid_step_deg": config.FM_GRID_STEP_DEG,
        "min_stations": config.FM_MIN_STATIONS,
        "max_azimuthal_gap_deg": config.FM_MAX_AZIMUTHAL_GAP_DEG,
        "observations": canon_obs,
    }
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# JSON snapshots (frozen context is stored verbatim with the report)
# --------------------------------------------------------------------------- #
def _r4(x: float) -> float:
    """Round to 4 decimals, normalizing negative zero."""
    v = round(float(x), 4)
    return 0.0 if v == 0 else v


def _plane_json(plane) -> dict:
    strike = _r4(plane[0])
    if strike >= 360.0:  # 359.9999... rounds onto the wrap boundary
        strike = 0.0
    return {
        "strike_deg": strike,
        "dip_deg": _r4(plane[1]),
        "rake_deg": _r4(plane[2]),
    }


def _frozen_json(
    event: Event,
    run: Run,
    model_row: VelocityModel,
    calibration: dict | None,
    observations: list[dict],
    stations: dict[str, tuple[float, float, float]],
    geometry: dict[str, tuple[float, float, float]],
) -> dict:
    return {
        "ingestion_id": run.ingestion_id,
        "model": {
            "name": model_row.name,
            "version": model_row.version,
            "layers": canonical_layers(model_row.layers),
        },
        "calibration": calibration,
        "event": {
            "event_id": event.id,
            "run_id": run.id,
            "origin_time": iso_z(event.origin_time),
            "latitude": event.latitude,
            "longitude": event.longitude,
            "depth_km": event.depth_km,
            "rms_residual_s": event.rms_s,
            "station_count": event.station_count,
            "quality": event.quality,
        },
        "grid": {
            "step_deg": config.FM_GRID_STEP_DEG,
            "strike_deg": {"min": 0.0, "max": 360.0, "upper_exclusive": True},
            "dip_deg": {"min": 0.0, "max": 90.0, "upper_exclusive": False},
            "rake_deg": {"min": -180.0, "max": 180.0, "upper_exclusive": True},
        },
        "stations": [
            {
                "station": o["station"],
                "latitude": stations[o["station"]][0],
                "longitude": stations[o["station"]][1],
                "elevation_m": stations[o["station"]][2],
                "distance_km": _r4(geometry[o["station"]][2]),
                "azimuth_deg": _r4(geometry[o["station"]][0]),
                "takeoff_deg": _r4(geometry[o["station"]][1]),
                "polarity": o["polarity"],
                "weight": o["weight"],
                "pick_ids": sorted(o["pick_ids"]),
            }
            for o in observations
        ],
    }


def _solution_stations_json(sol: dict, search_obs: list[dict]) -> list[dict]:
    rows = []
    for obs, sign, amp in zip(search_obs, sol["predictions"], sol["amplitudes"]):
        rows.append(
            {
                "station": obs["station"],
                "azimuth_deg": _r4(obs["azimuth_deg"]),
                "takeoff_deg": _r4(obs["takeoff_deg"]),
                "observed": "positive" if obs["sign"] > 0 else "negative",
                "predicted": {1: "positive", 0: "zero", -1: "negative"}[sign],
                "weight": round(obs["weight"], 3),
                "amplitude": round(amp, 6),
                "misfit": sign != obs["sign"],
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #
def get_mechanism(db: Session, mechanism_id: str) -> dict:
    row = db.get(FocalMechanism, mechanism_id)
    if row is None:
        raise ApiError(
            404,
            "focal_mechanism_not_found",
            f"focal mechanism {mechanism_id} not found",
        )
    return mechanism_payload(row, created=False)


def list_mechanisms(
    db: Session,
    event_id: str | None,
    run_id: str | None,
    limit: int,
    cursor: str | None,
) -> dict:
    if event_id is not None and db.get(Event, event_id) is None:
        raise ApiError(404, "event_not_found", f"event {event_id} not found")
    if run_id is not None and db.get(Run, run_id) is None:
        raise ApiError(404, "run_not_found", f"run {run_id} not found")

    filters = {"event_id": event_id, "run_id": run_id, "limit": limit}
    fingerprint = hashlib.sha256(
        json.dumps(filters, sort_keys=True).encode()
    ).hexdigest()[:16]

    cursor_at, cursor_id = None, None
    if cursor is not None:
        cursor_at, cursor_id = _decode_list_cursor(cursor, fingerprint)

    q = db.query(FocalMechanism)
    if event_id is not None:
        q = q.filter(FocalMechanism.event_id == event_id)
    if run_id is not None:
        q = q.filter(FocalMechanism.run_id == run_id)
    if cursor_at is not None:
        q = q.filter(
            (FocalMechanism.created_at > cursor_at)
            | (
                (FocalMechanism.created_at == cursor_at)
                & (FocalMechanism.id > cursor_id)
            )
        )
    q = q.order_by(FocalMechanism.created_at, FocalMechanism.id).limit(limit + 1)
    rows = q.all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        last = page[-1]
        next_cursor = _encode_list_cursor(last.created_at, last.id, fingerprint)
    return {
        "mechanisms": [mechanism_summary(m) for m in page],
        "next_cursor": next_cursor,
    }


def list_solutions(
    db: Session, mechanism_id: str, limit: int, cursor: str | None
) -> dict:
    mech = db.get(FocalMechanism, mechanism_id)
    if mech is None:
        raise ApiError(
            404,
            "focal_mechanism_not_found",
            f"focal mechanism {mechanism_id} not found",
        )
    fingerprint = hashlib.sha256(mechanism_id.encode()).hexdigest()[:16]
    cursor_index = None
    if cursor is not None:
        cursor_index = _decode_solution_cursor(cursor, fingerprint)

    q = db.query(FocalMechanismSolution).filter_by(mechanism_id=mech.id)
    if cursor_index is not None:
        q = q.filter(FocalMechanismSolution.sort_index > cursor_index)
    q = q.order_by(FocalMechanismSolution.sort_index).limit(limit + 1)
    rows = q.all()

    page = rows[:limit]
    next_cursor = None
    if len(rows) > limit:
        next_cursor = _encode_solution_cursor(page[-1].sort_index, fingerprint)
    return {
        "mechanism_id": mech.id,
        "event_id": mech.event_id,
        "run_id": mech.run_id,
        "solution_count": mech.solution_count,
        "min_misfit_weight": round(mech.min_misfit_weight, 6),
        "solutions": [solution_payload(s) for s in page],
        "next_cursor": next_cursor,
    }


# --------------------------------------------------------------------------- #
# cursors
# --------------------------------------------------------------------------- #
def _encode_list_cursor(created_at: datetime, mechanism_id: str, fingerprint: str) -> str:
    raw = json.dumps(
        {
            "t": created_at.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            "id": mechanism_id,
            "fp": fingerprint,
        }
    ).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_list_cursor(cursor: str, fingerprint: str) -> tuple[datetime, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        token_at = datetime.strptime(payload["t"], "%Y-%m-%dT%H:%M:%S.%f")
        token_id = payload["id"]
        token_fp = payload["fp"]
    except Exception as exc:
        raise ApiError(422, "invalid_cursor", "malformed pagination cursor") from exc
    if token_fp != fingerprint:
        raise ApiError(
            422,
            "invalid_cursor",
            "cursor was issued for a different focal mechanism query "
            "(event/run filter mismatch)",
        )
    return token_at, token_id


def _encode_solution_cursor(sort_index: int, fingerprint: str) -> str:
    raw = json.dumps({"i": sort_index, "fp": fingerprint}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_solution_cursor(cursor: str, fingerprint: str) -> int:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        token_i = int(payload["i"])
        token_fp = payload["fp"]
    except Exception as exc:
        raise ApiError(422, "invalid_cursor", "malformed pagination cursor") from exc
    if token_fp != fingerprint:
        raise ApiError(
            422,
            "invalid_cursor",
            "cursor was issued for a different focal mechanism inversion",
        )
    return token_i


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #
def solution_payload(row: FocalMechanismSolution) -> dict:
    return {
        "solution_id": row.id,
        "sort_index": row.sort_index,
        "strike_deg": row.strike_deg,
        "dip_deg": row.dip_deg,
        "rake_deg": row.rake_deg,
        "planes": row.planes,
        "misfit_weight": round(row.misfit_weight, 6),
        "misfit_count": row.misfit_count,
        "stations": row.stations,
    }


def mechanism_summary(mech: FocalMechanism) -> dict:
    return {
        "mechanism_id": mech.id,
        "event_id": mech.event_id,
        "run_id": mech.run_id,
        "model": {
            "name": mech.frozen["model"]["name"],
            "version": mech.frozen["model"]["version"],
        },
        "params_hash": mech.params_hash,
        "request": mech.request,
        "station_count": mech.station_count,
        "observation_count": mech.observation_count,
        "max_azimuthal_gap_deg": round(mech.max_azimuthal_gap_deg, 4),
        "evaluated_count": mech.evaluated_count,
        "min_misfit_weight": round(mech.min_misfit_weight, 6),
        "min_misfit_count": mech.min_misfit_count,
        "solution_count": mech.solution_count,
        "created_at": iso_z(mech.created_at),
    }


def mechanism_payload(mech: FocalMechanism, created: bool) -> dict:
    payload = mechanism_summary(mech)
    payload.update(
        {
            "created": created,
            "ingestion_id": mech.frozen["ingestion_id"],
            "frozen": mech.frozen,
        }
    )
    return payload
