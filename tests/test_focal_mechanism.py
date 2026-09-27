"""Tests for the P-wave first-motion focal mechanism inversion."""
from __future__ import annotations

import threading

import pytest

from app.config import DEFAULT_MODEL_LAYERS
from app.services import focal
from app.services.traveltime import LayeredModel
from tests.conftest import (
    BASE_TIME,
    make_waveform_b64,
    synthetic_event_payload,
)

MODEL = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
TRUE_MECH = (30.0, 60.0, -90.0)  # on the 10-degree grid
REVERSE_MECH = (30.0, 60.0, 90.0)  # slip reversed: flips every polarity


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _process(client, prefix, **kwargs):
    body, truth = synthetic_event_payload(prefix=prefix, **kwargs)
    resp = client.post("/v1/process", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json(), body


def _best(data):
    return max(data["events"], key=lambda e: e["station_count"])


def _stations_of(body):
    return {w["station"]: (w["latitude"], w["longitude"], w["elevation_m"])
            for w in body["waveforms"]}


def _p_picks(client, event_id):
    event = client.get(f"/v1/events/{event_id}").json()
    return event, [p for p in event["picks"] if p["phase"] == "P"]


def _pick_station_map(client, event_id):
    """pick_id -> station for the event's associated P picks."""
    _, p_picks = _p_picks(client, event_id)
    return {p["pick_id"]: p["station"] for p in p_picks}


def _polarity_observations(client, event_id, stations, mechanism=TRUE_MECH,
                           weight=1.0, flips=()):
    """Observations generated from a known mechanism via the service geometry."""
    event, p_picks = _p_picks(client, event_id)
    geometry = focal.station_geometry(
        event["latitude"], event["longitude"], event["depth_km"], stations, MODEL
    )
    n, u = focal.fault_vectors(*mechanism)
    observations = []
    for p in p_picks:
        az, takeoff, _ = geometry[p["station"]]
        amp = focal.p_amplitude(n, u, focal.ray_vector(az, takeoff))
        assert abs(amp) > 1e-6, "test mechanism must not be nodal at any station"
        polarity = "positive" if amp > 0 else "negative"
        if p["station"] in flips:
            polarity = "negative" if polarity == "positive" else "positive"
        observations.append(
            {"pick_id": p["pick_id"], "polarity": polarity, "weight": weight}
        )
    return observations


def _create(client, event_id, observations):
    return client.post(
        "/v1/focal-mechanisms",
        json={"event_id": event_id, "observations": observations},
    )


def _all_solutions(client, mechanism_id, limit=3):
    """Walk the paginated solutions endpoint to the full tied set."""
    solutions, cursor = [], None
    for _ in range(10000):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        resp = client.get(
            f"/v1/focal-mechanisms/{mechanism_id}/solutions", params=params
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        solutions.extend(data["solutions"])
        cursor = data["next_cursor"]
        if cursor is None:
            return solutions, data
    raise AssertionError("pagination did not terminate")


def _planes_close(planes, expected, tol=1e-3):
    """Each expected plane is present in ``planes`` (angle-wise)."""
    for exp in expected:
        if not any(
            all(abs(a - b) < tol for a, b in zip(p, exp)) for p in planes
        ):
            return False
    return True


def _solution_plane_tuples(solution):
    return [
        (p["strike_deg"], p["dip_deg"], p["rake_deg"]) for p in solution["planes"]
    ]


def _moment_keys_close(plane_a, plane_b, tol=1e-3):
    """Same mechanism up to output rounding (planes are stored at 4 decimals)."""
    ka = focal.moment_key(*plane_a)
    kb = focal.moment_key(*plane_b)
    return all(abs(x - y) < tol for x, y in zip(ka, kb))


def _add_extra_p_pick(event_id, station):
    """White-box: attach a second associated P pick to one station."""
    from app.database import SessionLocal
    from app.models import Event, EventPick, Pick, Run

    db = SessionLocal()
    try:
        event = db.get(Event, event_id)
        run = db.get(Run, event.run_id)
        pick = Pick(
            ingestion_id=run.ingestion_id,
            station=station,
            phase="P",
            time_utc=event.origin_time,
            source="manual",
            weight=1.0,
        )
        db.add(pick)
        db.flush()
        db.add(
            EventPick(
                event_id=event.id,
                pick_id=pick.id,
                phase="P",
                residual_s=0.0,
                weight=1.0,
            )
        )
        db.commit()
        return pick.id
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# unit: fault geometry
# --------------------------------------------------------------------------- #
def test_fault_vectors_orthonormal():
    for angles in [(0, 90, 0), (30, 60, -90), (123, 45, 77), (350, 10, -170)]:
        n, u = focal.fault_vectors(*angles)
        assert pytest.approx(1.0) == sum(c * c for c in n)
        assert pytest.approx(1.0) == sum(c * c for c in u)
        assert abs(sum(a * b for a, b in zip(n, u))) < 1e-12


def test_auxiliary_plane_known_cases():
    # vertical strike-slip: conjugate vertical plane, reversed rake
    assert focal.auxiliary_plane(0.0, 90.0, 0.0) == pytest.approx(
        (90.0, 90.0, -180.0)
    )
    # 60-degree normal fault: 30-degree normal fault the other way
    assert focal.auxiliary_plane(0.0, 60.0, -90.0) == pytest.approx(
        (180.0, 30.0, -90.0)
    )


def test_auxiliary_plane_is_involution():
    for angles in [(30, 60, -90), (0, 90, 0), (200, 45, 130), (90, 90, 45)]:
        back = focal.auxiliary_plane(*focal.auxiliary_plane(*angles))
        assert focal.moment_key(*back) == focal.moment_key(*angles)


def test_moment_key_plane_swap_invariant_but_not_slip_reversal():
    key = focal.moment_key(*TRUE_MECH)
    assert focal.moment_key(*focal.auxiliary_plane(*TRUE_MECH)) == key
    # slip reversal (rake + 180) negates the moment tensor: different mechanism
    assert focal.moment_key(*REVERSE_MECH) != key


def test_vectors_to_plane_roundtrip():
    for angles in [(30, 60, -90), (200, 45, 130), (90, 90, 45), (10, 20, 30)]:
        n, u = focal.fault_vectors(*angles)
        plane = focal.vectors_to_plane(n, u)
        assert focal.moment_key(*plane) == focal.moment_key(*angles)


# --------------------------------------------------------------------------- #
# unit: radiation pattern / misfit evaluation
# --------------------------------------------------------------------------- #
def test_radiation_quadrant_pattern_vertical_strike_slip():
    # (strike 0, dip 90, rake 0): nodal along strike and dip directions,
    # alternating quadrants in between (hand-computed).
    n, u = focal.fault_vectors(0.0, 90.0, 0.0)
    expected = {45: -1, 135: 1, 225: -1, 315: 1, 0: 0, 90: 0, 180: 0, 270: 0}
    for az, sign in expected.items():
        amp = focal.p_amplitude(n, u, focal.ray_vector(az, 135.0))
        assert focal.predicted_sign(amp) == sign, f"azimuth {az}"


def test_zero_amplitude_counts_as_misfit():
    # station exactly on a nodal plane of the candidate mechanism
    observations = [
        {"azimuth_deg": 0.0, "takeoff_deg": 135.0, "sign": 1, "weight": 2.0},
        {"azimuth_deg": 135.0, "takeoff_deg": 135.0, "sign": 1, "weight": 1.0},
    ]
    cand = focal.evaluate(0.0, 90.0, 0.0, observations)
    # station 1 is nodal (amplitude 0) -> misfit despite matching quadrant
    assert cand["predictions"][0] == 0
    assert cand["misfit_weight"] == pytest.approx(2.0)
    assert cand["misfit_count"] == 1
    assert cand["predictions"][1] == 1


def test_evaluate_weighted_misfit_sums_weights():
    observations = [
        {"azimuth_deg": 45.0, "takeoff_deg": 135.0, "sign": 1, "weight": 1.5},
        {"azimuth_deg": 135.0, "takeoff_deg": 135.0, "sign": -1, "weight": 2.5},
        {"azimuth_deg": 225.0, "takeoff_deg": 135.0, "sign": -1, "weight": 0.5},
    ]
    # (0,90,0) predicts -1/+1/-1 at 45/135/225: the first two mismatch
    cand = focal.evaluate(0.0, 90.0, 0.0, observations)
    assert cand["misfit_weight"] == pytest.approx(4.0)
    assert cand["misfit_count"] == 2


# --------------------------------------------------------------------------- #
# unit: station geometry / azimuthal gap
# --------------------------------------------------------------------------- #
def test_station_geometry_azimuth_and_takeoff():
    stations = {
        "N": (35.5, -118.0, 0.0),   # due north
        "E": (35.0, -117.5, 0.0),   # due east
        "TOP": (35.0, -118.0, 0.0),  # directly overhead
    }
    geometry = focal.station_geometry(35.0, -118.0, 12.0, stations, MODEL)
    assert geometry["N"][0] == pytest.approx(0.0, abs=1e-6)
    assert geometry["E"][0] == pytest.approx(90.0, abs=1e-6)
    # upgoing rays: takeoff in (90, 180]; overhead station goes straight up
    assert 90.0 < geometry["N"][1] < 180.0
    assert geometry["TOP"][1] == pytest.approx(180.0)
    assert geometry["TOP"][2] == pytest.approx(0.0, abs=1e-9)


def test_max_azimuthal_gap():
    assert focal.max_azimuthal_gap([0, 90, 180, 270]) == pytest.approx(90.0)
    assert focal.max_azimuthal_gap([10, 50, 200]) == pytest.approx(170.0)
    assert focal.max_azimuthal_gap([350, 10, 180]) == pytest.approx(170.0)
    assert focal.max_azimuthal_gap([42.0]) == 360.0


def test_grid_search_recovers_true_mechanism_and_dedupes():
    observations = []
    n, u = focal.fault_vectors(*TRUE_MECH)
    for az, toff in [(10, 140), (70, 150), (130, 145), (190, 155), (250, 148), (310, 142)]:
        amp = focal.p_amplitude(n, u, focal.ray_vector(az, toff))
        observations.append(
            {"azimuth_deg": az, "takeoff_deg": toff,
             "sign": 1 if amp > 0 else -1, "weight": 1.0}
        )
    best, solutions, evaluated = focal.grid_search(observations)
    assert best == 0.0
    # de-duplication: fewer unique mechanisms than raw grid nodes
    assert 0 < evaluated < 36 * 10 * 36
    # the generating mechanism is among the tied best solutions
    true_planes = [TRUE_MECH, focal.auxiliary_plane(*TRUE_MECH)]
    assert any(_planes_close(s["planes"], true_planes, tol=1e-6) for s in solutions)
    # solutions are unique (no plane-swap duplicates) and stably ordered
    keys = [tuple(tuple(p) for p in s["planes"]) for s in solutions]
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys)


def test_grid_search_all_flipped_finds_reverse_mechanism():
    observations = []
    n, u = focal.fault_vectors(*TRUE_MECH)
    for az, toff in [(10, 140), (70, 150), (130, 145), (190, 155), (250, 148), (310, 142)]:
        amp = focal.p_amplitude(n, u, focal.ray_vector(az, toff))
        observations.append(
            {"azimuth_deg": az, "takeoff_deg": toff,
             "sign": -1 if amp > 0 else 1, "weight": 1.0}  # all flipped
        )
    best, solutions, _ = focal.grid_search(observations)
    assert best == 0.0
    reverse_planes = [REVERSE_MECH, focal.auxiliary_plane(*REVERSE_MECH)]
    true_planes = [TRUE_MECH, focal.auxiliary_plane(*TRUE_MECH)]
    assert any(_planes_close(s["planes"], reverse_planes, tol=1e-6) for s in solutions)
    assert not any(
        _planes_close(s["planes"], true_planes, tol=1e-6) for s in solutions
    )


# --------------------------------------------------------------------------- #
# e2e: successful inversion
# --------------------------------------------------------------------------- #
def test_create_mechanism_recovers_true_mechanism(client):
    data, body = _process(client, "FM", n_stations=8)
    event = _best(data)
    observations = _polarity_observations(
        client, event["event_id"], _stations_of(body)
    )
    resp = _create(client, event["event_id"], observations)
    assert resp.status_code == 201, resp.text
    rep = resp.json()

    assert rep["created"] is True
    assert rep["event_id"] == event["event_id"]
    assert rep["run_id"] == data["run_id"]
    assert rep["ingestion_id"] == data["ingestion_id"]
    assert rep["model"] == {"name": "default", "version": "1.0"}
    assert rep["station_count"] == 8
    assert rep["observation_count"] == 8
    assert rep["min_misfit_weight"] == 0.0
    assert rep["min_misfit_count"] == 0
    assert rep["solution_count"] >= 1
    assert rep["evaluated_count"] > rep["solution_count"]
    assert 0.0 < rep["max_azimuthal_gap_deg"] < 180.0
    assert len(rep["params_hash"]) == 64

    # frozen context: hypocenter, model layers, grid, per-station geometry
    frozen = rep["frozen"]
    assert frozen["model"]["name"] == "default"
    assert frozen["model"]["layers"]
    assert frozen["calibration"] is None
    assert frozen["event"]["event_id"] == event["event_id"]
    assert frozen["grid"]["step_deg"] == 10.0
    assert len(frozen["stations"]) == 8
    for row in frozen["stations"]:
        assert 0.0 <= row["azimuth_deg"] < 360.0
        assert 90.0 < row["takeoff_deg"] <= 180.0
        assert row["polarity"] in ("positive", "negative")
        assert len(row["pick_ids"]) == 1

    # canonical request echoes the normalized polarity set
    assert rep["request"]["event_id"] == event["event_id"]
    assert rep["request"]["model"] == {"name": "default", "version": "1.0"}
    assert rep["request"]["observations"] == sorted(
        rep["request"]["observations"], key=lambda o: o["station"]
    )

    # the generating mechanism is among the tied best solutions
    solutions, last = _all_solutions(client, rep["mechanism_id"])
    assert len(solutions) == rep["solution_count"]
    assert last["min_misfit_weight"] == 0.0
    true_planes = [TRUE_MECH, focal.auxiliary_plane(*TRUE_MECH)]
    assert any(
        _planes_close(_solution_plane_tuples(s), true_planes) for s in solutions
    )

    # every solution: two nodal planes + per-station observed/predicted detail
    for sol in solutions:
        assert sol["misfit_weight"] == 0.0
        assert sol["misfit_count"] == 0
        assert len(sol["planes"]) == 2
        headline = (sol["strike_deg"], sol["dip_deg"], sol["rake_deg"])
        assert headline == _solution_plane_tuples(sol)[0]
        # the two planes are each other's auxiliary plane (same mechanism)
        p1, p2 = _solution_plane_tuples(sol)
        assert _moment_keys_close(p1, p2)
        assert len(sol["stations"]) == 8
        for st in sol["stations"]:
            assert st["observed"] in ("positive", "negative")
            assert st["predicted"] in ("positive", "negative", "zero")
            assert st["misfit"] is False
            assert st["observed"] == st["predicted"]


def test_all_flipped_polarities_find_reverse_mechanism(client):
    data, body = _process(client, "FR", n_stations=8)
    event = _best(data)
    stations = _stations_of(body)
    observations = _polarity_observations(client, event["event_id"], stations)
    for obs in observations:  # flip every station
        obs["polarity"] = (
            "negative" if obs["polarity"] == "positive" else "positive"
        )
    rep = _create(client, event["event_id"], observations).json()
    assert rep["min_misfit_weight"] == 0.0
    solutions, _ = _all_solutions(client, rep["mechanism_id"])
    reverse_planes = [REVERSE_MECH, focal.auxiliary_plane(*REVERSE_MECH)]
    true_planes = [TRUE_MECH, focal.auxiliary_plane(*TRUE_MECH)]
    assert any(
        _planes_close(_solution_plane_tuples(s), reverse_planes) for s in solutions
    )
    assert not any(
        _planes_close(_solution_plane_tuples(s), true_planes) for s in solutions
    )


def test_flipped_station_weight_bounds_min_misfit(client):
    data, body = _process(client, "FW", n_stations=8)
    event = _best(data)
    stations = _stations_of(body)
    flipped_station = sorted(stations)[0]
    observations = _polarity_observations(
        client, event["event_id"], stations, flips={flipped_station}
    )
    for obs in observations:
        obs["weight"] = 2.5
    rep = _create(client, event["event_id"], observations).json()
    # the generating mechanism now misfits exactly one station of weight 2.5,
    # so the optimum cannot be worse than that (another mechanism may still
    # fit the flipped set perfectly)
    assert 0.0 <= rep["min_misfit_weight"] <= 2.5
    solutions, _ = _all_solutions(client, rep["mechanism_id"])
    for sol in solutions:
        # per-solution misfit equals the sum of its mismatched station weights
        total = sum(st["weight"] for st in sol["stations"] if st["misfit"])
        assert sol["misfit_weight"] == pytest.approx(total)
        assert sol["misfit_count"] == sum(
            1 for st in sol["stations"] if st["misfit"]
        )


# --------------------------------------------------------------------------- #
# e2e: idempotency / concurrency
# --------------------------------------------------------------------------- #
def test_mechanism_is_idempotent(client):
    data, body = _process(client, "FI", n_stations=8)
    event = _best(data)
    observations = _polarity_observations(
        client, event["event_id"], _stations_of(body)
    )
    first = _create(client, event["event_id"], observations)
    assert first.status_code == 201
    second = _create(client, event["event_id"], observations)
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["mechanism_id"] == first.json()["mechanism_id"]
    assert second.json()["params_hash"] == first.json()["params_hash"]
    assert second.json()["solution_count"] == first.json()["solution_count"]


def test_idempotency_uses_normalized_polarity_set(client):
    """Two picks of one station (weights 1+1) normalize like one pick of weight 2."""
    data, body = _process(client, "FN", n_stations=8)
    event = _best(data)
    stations = _stations_of(body)
    observations = _polarity_observations(client, event["event_id"], stations)
    pick_station = _pick_station_map(client, event["event_id"])
    target_station = sorted(stations)[0]

    # variant A: the station's single P pick carries weight 2.0
    obs_a = [dict(o) for o in observations]
    for o in obs_a:
        if pick_station[o["pick_id"]] == target_station:
            o["weight"] = 2.0
    rep_a = _create(client, event["event_id"], obs_a)
    assert rep_a.status_code == 201, rep_a.text

    # variant B: a second associated P pick of the same station, 1.0 + 1.0
    obs_b = [dict(o) for o in observations]
    extra_pick = _add_extra_p_pick(event["event_id"], target_station)
    polarity_of_station = next(
        o["polarity"] for o in observations
        if pick_station[o["pick_id"]] == target_station
    )
    obs_b.append(
        {"pick_id": extra_pick, "polarity": polarity_of_station, "weight": 1.0}
    )
    rep_b = _create(client, event["event_id"], obs_b)
    assert rep_b.status_code == 200, rep_b.text
    assert rep_b.json()["created"] is False
    assert rep_b.json()["mechanism_id"] == rep_a.json()["mechanism_id"]
    # the normalized station weight is the sum of its pick weights
    frozen_row = next(
        s for s in rep_b.json()["frozen"]["stations"]
        if s["station"] == target_station
    )
    assert frozen_row["weight"] == 2.0
    assert len(frozen_row["pick_ids"]) == 1  # frozen from the first request


def test_concurrent_identical_mechanisms_collapse(client):
    data, body = _process(client, "FC", n_stations=8)
    event = _best(data)
    observations = _polarity_observations(
        client, event["event_id"], _stations_of(body)
    )
    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = _create(client, event["event_id"], observations)
        results.append((r.status_code, r.json()["mechanism_id"]))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()

    statuses = sorted(s for s, _ in results)
    assert statuses == [200, 201], results
    assert results[0][1] == results[1][1]
    listed = client.get(
        "/v1/focal-mechanisms", params={"event_id": event["event_id"]}
    ).json()
    assert len(listed["mechanisms"]) == 1


# --------------------------------------------------------------------------- #
# e2e: rejections
# --------------------------------------------------------------------------- #
def test_unknown_event_returns_404(client):
    resp = _create(client, "0" * 32, [{"pick_id": "x", "polarity": "positive"}])
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "event_not_found"


def test_unknown_pick_returns_404(client):
    data, body = _process(client, "FUP", n_stations=8)
    event = _best(data)
    observations = _polarity_observations(
        client, event["event_id"], _stations_of(body)
    )
    observations[0]["pick_id"] = "f" * 32
    resp = _create(client, event["event_id"], observations)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "pick_not_found"


def test_unassociated_and_s_picks_are_rejected(client):
    data, body = _process(client, "FUA", n_stations=8)
    event = _best(data)
    observations = _polarity_observations(
        client, event["event_id"], _stations_of(body)
    )
    # an S pick of the same event is not a P observation
    s_pick = next(p for p in event["picks"] if p["phase"] == "S")
    obs_s = [dict(o) for o in observations]
    obs_s[0] = {"pick_id": s_pick["pick_id"], "polarity": "positive", "weight": 1.0}
    resp = _create(client, event["event_id"], obs_s)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "pick_not_associated"

    # an unassociated pick of the same ingestion is rejected too
    unassociated = data["unassociated_picks"]
    if unassociated:
        obs_u = [dict(o) for o in observations]
        obs_u[0] = {
            "pick_id": unassociated[0]["pick_id"],
            "polarity": "positive",
            "weight": 1.0,
        }
        resp = _create(client, event["event_id"], obs_u)
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "pick_not_associated"


def test_duplicate_pick_is_rejected(client):
    data, body = _process(client, "FD", n_stations=8)
    event = _best(data)
    observations = _polarity_observations(
        client, event["event_id"], _stations_of(body)
    )
    observations.append(dict(observations[0]))
    resp = _create(client, event["event_id"], observations)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "duplicate_pick"


def test_contradictory_polarity_is_rejected(client):
    data, body = _process(client, "FX", n_stations=8)
    event = _best(data)
    stations = _stations_of(body)
    observations = _polarity_observations(client, event["event_id"], stations)
    pick_station = _pick_station_map(client, event["event_id"])
    target_station = sorted(stations)[0]
    extra_pick = _add_extra_p_pick(event["event_id"], target_station)
    opposite = next(
        "negative" if o["polarity"] == "positive" else "positive"
        for o in observations
        if pick_station[o["pick_id"]] == target_station
    )
    observations.append(
        {"pick_id": extra_pick, "polarity": opposite, "weight": 1.0}
    )
    resp = _create(client, event["event_id"], observations)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "contradictory_polarity"


def test_fewer_than_six_stations_is_rejected(client):
    data, body = _process(client, "F5", n_stations=8)
    event = _best(data)
    observations = _polarity_observations(
        client, event["event_id"], _stations_of(body)
    )[:5]
    resp = _create(client, event["event_id"], observations)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "insufficient_polarity_stations"


def _arc_payload(prefix, angles, ev_lat=35.05, ev_lon=-117.95, ev_depth_km=12.0):
    """Synthetic event with stations at explicit azimuths from the event."""
    import numpy as np

    origin_offset_s = 30.0
    radius_km = 0.25 * 111.32
    tp = MODEL.travel_time(ev_depth_km, radius_km, "P")[0]
    ts = MODEL.travel_time(ev_depth_km, radius_km, "S")[0]
    waveforms = []
    for i, az in enumerate(angles):
        code = f"{prefix}{i:02d}"
        lat = ev_lat + 0.25 * np.cos(np.radians(az))
        lon = ev_lon + 0.25 * np.sin(np.radians(az)) / np.cos(np.radians(ev_lat))
        b64 = make_waveform_b64(
            code, 100.0, BASE_TIME,
            [(origin_offset_s + tp, 4.0), (origin_offset_s + ts, 6.0)],
        )
        waveforms.append(
            {
                "station": code,
                "latitude": float(lat),
                "longitude": float(lon),
                "elevation_m": 0.0,
                "sampling_rate": 100.0,
                "start_time": BASE_TIME,
                "data_b64": b64,
            }
        )
    return {
        "waveforms": waveforms,
        "manual_picks": [],
        "model": {"name": "default", "version": "1.0"},
        "params": {"max_residual_s": 1.0, "min_stations": 4},
    }


def test_azimuthal_gap_of_180_or_more_is_rejected(client):
    # six stations inside a ~100-degree arc leave a ~260-degree gap
    resp = client.post(
        "/v1/process", json=_arc_payload("FG", [0, 20, 40, 60, 80, 100])
    )
    assert resp.status_code == 200, resp.text
    event = _best(resp.json())
    event_detail, p_picks = _p_picks(client, event["event_id"])
    assert len(p_picks) == 6
    observations = [
        {"pick_id": p["pick_id"], "polarity": "positive", "weight": 1.0}
        for p in p_picks
    ]
    resp = _create(client, event["event_id"], observations)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "azimuthal_gap_too_large"


def test_request_schema_validation(client):
    data, body = _process(client, "FV", n_stations=8)
    event = _best(data)
    _, p_picks = _p_picks(client, event["event_id"])
    pick_id = p_picks[0]["pick_id"]

    # empty observation list
    resp = _create(client, event["event_id"], [])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "validation_error"
    # unknown polarity literal
    resp = _create(
        client, event["event_id"], [{"pick_id": pick_id, "polarity": "up"}]
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "validation_error"
    # non-positive / too-large weights
    for bad_weight in (0.0, -1.0, 11.0):
        resp = _create(
            client,
            event["event_id"],
            [{"pick_id": pick_id, "polarity": "positive", "weight": bad_weight}],
        )
        assert resp.status_code == 422
        assert resp.json()["detail"]["code"] == "validation_error"


# --------------------------------------------------------------------------- #
# e2e: queries / pagination
# --------------------------------------------------------------------------- #
def test_solutions_pagination_is_stable(client):
    data, body = _process(client, "FP", n_stations=8)
    event = _best(data)
    observations = _polarity_observations(
        client, event["event_id"], _stations_of(body)
    )
    rep = _create(client, event["event_id"], observations).json()
    solutions, last = _all_solutions(client, rep["mechanism_id"], limit=2)
    # gapless stable order over the whole tied set
    assert [s["sort_index"] for s in solutions] == list(range(len(solutions)))
    assert len(solutions) == rep["solution_count"]
    assert last["solution_count"] == rep["solution_count"]
    assert last["mechanism_id"] == rep["mechanism_id"]
    ids = [s["solution_id"] for s in solutions]
    assert len(ids) == len(set(ids))


def test_solution_cursor_bound_to_mechanism(client):
    data1, body1 = _process(client, "FQ1", n_stations=8)
    data2, body2 = _process(client, "FQ2", n_stations=8)
    m1 = _create(
        client, _best(data1)["event_id"],
        _polarity_observations(client, _best(data1)["event_id"], _stations_of(body1)),
    ).json()
    m2 = _create(
        client, _best(data2)["event_id"],
        _polarity_observations(client, _best(data2)["event_id"], _stations_of(body2)),
    ).json()
    page = client.get(
        f"/v1/focal-mechanisms/{m1['mechanism_id']}/solutions", params={"limit": 1}
    ).json()
    if page["next_cursor"] is None:
        pytest.skip("single-solution inversion cannot paginate")
    bad = client.get(
        f"/v1/focal-mechanisms/{m2['mechanism_id']}/solutions",
        params={"limit": 1, "cursor": page["next_cursor"]},
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "invalid_cursor"


def test_mechanism_list_filters_and_pagination(client):
    data1, body1 = _process(client, "FL1", n_stations=8)
    data2, body2 = _process(client, "FL2", n_stations=8)
    e1, e2 = _best(data1)["event_id"], _best(data2)["event_id"]
    m1 = _create(client, e1, _polarity_observations(client, e1, _stations_of(body1))).json()
    m2 = _create(client, e2, _polarity_observations(client, e2, _stations_of(body2))).json()

    by_event = client.get("/v1/focal-mechanisms", params={"event_id": e1}).json()
    assert [m["mechanism_id"] for m in by_event["mechanisms"]] == [
        m1["mechanism_id"]
    ]
    by_run = client.get(
        "/v1/focal-mechanisms", params={"run_id": data2["run_id"]}
    ).json()
    assert [m["mechanism_id"] for m in by_run["mechanisms"]] == [
        m2["mechanism_id"]
    ]

    # keyset pagination walks every inversion exactly once
    seen, cursor = set(), None
    for _ in range(10):
        params = {"limit": 1}
        if cursor:
            params["cursor"] = cursor
        page = client.get("/v1/focal-mechanisms", params=params).json()
        seen.update(m["mechanism_id"] for m in page["mechanisms"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert seen == {m1["mechanism_id"], m2["mechanism_id"]}

    # list cursor is bound to the event/run filter fingerprint
    page = client.get(
        "/v1/focal-mechanisms", params={"limit": 1, "event_id": e1}
    ).json()
    mismatched = client.get(
        "/v1/focal-mechanisms",
        params={"limit": 1, "event_id": e2, "cursor": page["next_cursor"]},
    )
    assert mismatched.status_code == 422
    assert mismatched.json()["detail"]["code"] == "invalid_cursor"

    # unknown filters are 404
    resp = client.get("/v1/focal-mechanisms", params={"event_id": "e" * 32})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "event_not_found"
    resp = client.get("/v1/focal-mechanisms", params={"run_id": "r" * 32})
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "run_not_found"


def test_mechanism_not_found(client):
    resp = client.get("/v1/focal-mechanisms/deadbeef")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "focal_mechanism_not_found"
    resp = client.get("/v1/focal-mechanisms/deadbeef/solutions")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "focal_mechanism_not_found"


def test_get_mechanism_returns_frozen_report(client):
    data, body = _process(client, "FG1", n_stations=8)
    event = _best(data)
    created = _create(
        client, event["event_id"],
        _polarity_observations(client, event["event_id"], _stations_of(body)),
    ).json()
    fetched = client.get(f"/v1/focal-mechanisms/{created['mechanism_id']}")
    assert fetched.status_code == 200
    rep = fetched.json()
    assert rep["created"] is False
    assert rep["mechanism_id"] == created["mechanism_id"]
    assert rep["frozen"] == created["frozen"]
    assert rep["solution_count"] == created["solution_count"]


# --------------------------------------------------------------------------- #
# e2e: read-only layer (original event/run and processing semantics unchanged)
# --------------------------------------------------------------------------- #
def test_mechanism_does_not_mutate_event_or_run(client):
    data, body = _process(client, "FRO", n_stations=8)
    event = _best(data)
    run_before = client.get(f"/v1/runs/{data['run_id']}").json()
    event_before = client.get(f"/v1/events/{event['event_id']}").json()

    observations = _polarity_observations(
        client, event["event_id"], _stations_of(body)
    )
    _create(client, event["event_id"], observations)
    _create(client, event["event_id"], observations)  # repeat as well

    assert client.get(f"/v1/runs/{data['run_id']}").json() == run_before
    assert client.get(f"/v1/events/{event['event_id']}").json() == event_before

    # process semantics are untouched (same window still idempotent)
    again = client.post("/v1/process", json=body)
    assert again.json()["run_id"] == data["run_id"]
    assert again.json()["created"] is False
