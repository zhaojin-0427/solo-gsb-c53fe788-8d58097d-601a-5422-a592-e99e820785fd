"""End-to-end tests for the P-wave first-motion focal mechanism inversion."""
from __future__ import annotations

import math
import threading

import pytest

from app.config import DEFAULT_MODEL_LAYERS
from app.services import focal_mechanism as fm
from app.services.association import distance_km
from app.services.traveltime import LayeredModel

from tests.conftest import (
    BASE_TIME,
    make_waveform_b64,
    station_ring,
    synthetic_event_payload,
)

# Grid mechanism whose station amplitudes are all well away from nodal for the
# default synthetic geometry (min |A| ~ 0.21, balanced 4+/4- polarities).
TRUE_MECHANISM = (340.0, 70.0, -170.0)


def _process(client, prefix, **kwargs):
    resp = client.post(
        "/v1/process", json=synthetic_event_payload(prefix=prefix, **kwargs)[0]
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _best(data):
    return max(data["events"], key=lambda e: e["station_count"])


def _p_picks(event):
    return [p for p in event["picks"] if p["phase"] == "P"]


def _polarities_for_mechanism(event, prefix, mechanism, n_stations=8):
    """Client-side polarity assignment for a known mechanism.

    Uses the same public geometry/radiation helpers the service exposes, over
    the event's reported hypocenter and the ring station coordinates.
    """
    model = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
    coords = {code: (lat, lon, elev) for code, lat, lon, elev in station_ring(prefix, n=n_stations)}
    normal, slip = fm.fault_vectors(*mechanism)
    observations = []
    for pick in _p_picks(event):
        lat, lon, elev = coords[pick["station"]]
        az, to = fm.ray_geometry(
            model,
            event["latitude"],
            event["longitude"],
            event["depth_km"],
            lat,
            lon,
            elev,
        )
        amplitude = fm.p_radiation(normal, slip, az, to)
        assert abs(amplitude) > 0.05, "test geometry too close to a nodal plane"
        observations.append(
            {
                "pick_id": pick["pick_id"],
                "polarity": "positive" if amplitude > 0 else "negative",
                "weight": 1.0,
            }
        )
    return observations


def _invert(client, event_id, observations):
    return client.post(
        "/v1/focal-mechanisms",
        json={"event_id": event_id, "observations": observations},
    )


def _invert_known(client, prefix="FM", mechanism=TRUE_MECHANISM, **kwargs):
    data = _process(client, prefix, **kwargs)
    event = _best(data)
    obs = _polarities_for_mechanism(event, prefix, mechanism)
    return data, event, obs, _invert(client, event["event_id"], obs)


def _solution_key(solution):
    p = solution["plane_1"]
    return fm.mechanism_key(p["strike_deg"], p["dip_deg"], p["rake_deg"])


def _mechanism_key_close(key_a, key_b, tol=1e-4):
    """Mechanism keys from display-rounded angles agree within rounding noise."""
    return all(abs(a - b) <= tol for a, b in zip(key_a, key_b))


def _all_solutions(client, inversion_id, limit=7):
    """Walk the solutions endpoint page by page and return every solution."""
    seen, cursor = [], None
    for _ in range(1000):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        resp = client.get(f"/v1/focal-mechanisms/{inversion_id}/solutions", params=params)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        seen.extend(data["solutions"])
        cursor = data["next_cursor"]
        if cursor is None:
            return seen
    raise AssertionError("pagination did not terminate")


# --------------------------------------------------------------------------- #
# double-couple geometry (unit)
# --------------------------------------------------------------------------- #
def test_plane_angles_round_trip_preserves_mechanism():
    for angles in [
        (0, 90, 0),
        (30, 60, -90),
        (120, 45, 10),
        (350, 80, 170),
        (0, 90, 90),  # auxiliary plane is horizontal
        (200, 30, 45),
        (90, 10, -180),
    ]:
        normal, slip = fm.fault_vectors(*angles)
        recovered = fm.plane_angles(normal, slip)
        assert fm.mechanism_key(*recovered) == fm.mechanism_key(*angles)


def test_mechanism_key_invariant_under_nodal_plane_swap():
    normal, slip = fm.fault_vectors(30, 60, -90)
    aux = fm.plane_angles(slip, normal)  # auxiliary plane: normal <-> slip
    assert aux == pytest.approx((210.0, 30.0, -90.0), abs=1e-9)
    assert fm.mechanism_key(*aux) == fm.mechanism_key(30, 60, -90)


def test_nodal_prediction_counts_as_misfit():
    # N-striking vertical strike-slip: azimuth 0 sits exactly on a nodal plane.
    normal, slip = fm.fault_vectors(0, 90, 0)
    amplitude = fm.p_radiation(normal, slip, 0.0, 120.0)
    assert abs(amplitude) <= fm.NODAL_EPS
    assert fm._predicted_sign(amplitude) == 0

    rays = {"STA": fm._ray_vector(0.0, 120.0)}
    best, tied = fm.grid_search(rays, {"STA": (1, 1.0)})
    assert best == 0.0
    # the nodal mechanism misfits (zero amplitude counts as a misfit), so it
    # cannot be among the zero-misfit ties
    nodal_key = fm.mechanism_key(0, 90, 0)
    assert all(fm.mechanism_key(*t[:3]) != nodal_key for t in tied)


def test_grid_search_weights_select_heavier_match():
    # Two stations at the same takeoff direction with opposite polarities: any
    # mechanism misfits at least one; the lighter one must be sacrificed.
    direction = fm._ray_vector(45.0, 115.0)
    rays = {"A": direction, "B": direction}
    best, tied = fm.grid_search(rays, {"A": (1, 1.0), "B": (-1, 2.5)})
    assert best == 1.0
    assert tied, "expected tied solutions"
    for strike, dip, rake, misfit, count in tied:
        normal, slip = fm.fault_vectors(strike, dip, rake)
        assert fm._predicted_sign(fm.p_radiation(normal, slip, 45.0, 115.0)) == -1
        assert misfit == 1.0
        assert count == 1


# --------------------------------------------------------------------------- #
# happy path: known mechanism recovered, structure, dedupe
# --------------------------------------------------------------------------- #
def test_inversion_recovers_known_mechanism(client):
    data, event, obs, resp = _invert_known(client)
    assert resp.status_code == 201, resp.text
    rep = resp.json()

    assert rep["created"] is True
    assert rep["event_id"] == event["event_id"]
    assert rep["run_id"] == data["run_id"]
    assert rep["grid_step_deg"] == 10.0
    assert rep["station_count"] == 8
    assert rep["min_misfit_weight"] == 0.0
    assert rep["solution_count"] >= 1
    assert len(rep["solutions"]) == min(50, rep["solution_count"])

    # the true mechanism is among the tied best solutions
    keys = {_solution_key(s) for s in _all_solutions(client, rep["inversion_id"])}
    assert fm.mechanism_key(*TRUE_MECHANISM) in keys


def test_solution_structure_and_frozen_context(client):
    data, event, obs, resp = _invert_known(client, prefix="FS")
    assert resp.status_code == 201, resp.text
    rep = resp.json()

    frozen = rep["frozen"]
    assert frozen["model"]["name"] == "default"
    assert frozen["model"]["version"] == "1.0"
    assert frozen["model"]["layers"]
    hyp = frozen["hypocenter"]
    assert hyp["latitude"] == event["latitude"]
    assert hyp["longitude"] == event["longitude"]
    assert hyp["depth_km"] == event["depth_km"]
    assert len(frozen["stations"]) == 8
    for st in frozen["stations"]:
        assert 0.0 <= st["azimuth_deg"] < 360.0
        assert 90.0 < st["takeoff_deg"] <= 180.0  # upgoing rays to the surface
    assert frozen["azimuthal_gap_deg"] < 180.0
    # normalized polarity set: one row per station, sorted by station code
    assert [p["station"] for p in frozen["polarities"]] == sorted(
        p["station"] for p in frozen["polarities"]
    )
    assert {p["polarity"] for p in frozen["polarities"]} == {"positive", "negative"}
    assert rep["request"]["event_id"] == event["event_id"]
    assert rep["request"]["model"] == {"name": "default", "version": "1.0"}

    pick_station = {p["pick_id"]: p["station"] for p in event["picks"]}
    for page_solution in _all_solutions(client, rep["inversion_id"]):
        p1, p2 = page_solution["plane_1"], page_solution["plane_2"]
        for plane in (p1, p2):
            assert 0.0 <= plane["strike_deg"] < 360.0
            assert 0.0 < plane["dip_deg"] <= 90.0
            assert -180.0 <= plane["rake_deg"] < 180.0
        # the two planes are the same mechanism (swapped representation)
        key_1 = fm.mechanism_key(p1["strike_deg"], p1["dip_deg"], p1["rake_deg"])
        key_2 = fm.mechanism_key(p2["strike_deg"], p2["dip_deg"], p2["rake_deg"])
        assert _mechanism_key_close(key_1, key_2)
        # per-station observed/predicted rows are consistent with the misfit
        observations = page_solution["observations"]
        assert len(observations) == 8
        mismatches = [o for o in observations if not o["match"]]
        assert page_solution["misfit_count"] == len(mismatches)
        assert page_solution["misfit_weight"] == round(
            sum(o["weight"] for o in mismatches), 6
        )
        for o in observations:
            assert pick_station[o["pick_id"]] == o["station"]
            assert o["observed"] in ("positive", "negative")
            assert o["predicted"] in ("positive", "negative", "nodal")
            assert o["match"] == (o["observed"] == o["predicted"])


def test_equivalent_plane_swap_solutions_are_deduplicated(client):
    _, _, _, resp = _invert_known(client, prefix="DD")
    assert resp.status_code == 201, resp.text
    rep = resp.json()
    solutions = _all_solutions(client, rep["inversion_id"])
    assert len(solutions) == rep["solution_count"]
    keys = [_solution_key(s) for s in solutions]
    assert len(keys) == len(set(keys)), "swapped-plane duplicates must be removed"
    # stable presentation order: (strike, dip, rake) of plane 1
    order = [
        (s["plane_1"]["strike_deg"], s["plane_1"]["dip_deg"], s["plane_1"]["rake_deg"])
        for s in solutions
    ]
    assert order == sorted(order)
    assert [s["solution_index"] for s in solutions] == list(range(len(solutions)))


# --------------------------------------------------------------------------- #
# idempotency / concurrency
# --------------------------------------------------------------------------- #
def test_inversion_is_idempotent(client):
    _, event, obs, first = _invert_known(client, prefix="II")
    assert first.status_code == 201
    second = _invert(client, event["event_id"], obs)
    assert second.status_code == 200
    assert second.json()["created"] is False
    assert second.json()["inversion_id"] == first.json()["inversion_id"]
    assert second.json()["solutions"] == first.json()["solutions"]


def test_request_order_and_duplicates_normalize_to_same_inversion(client):
    _, event, obs, first = _invert_known(client, prefix="IN")
    assert first.status_code == 201
    # reversed order plus duplicated entries (same polarity) normalize away
    shuffled = list(reversed(obs)) + obs[:3]
    second = _invert(client, event["event_id"], shuffled)
    assert second.status_code == 200
    assert second.json()["inversion_id"] == first.json()["inversion_id"]


def test_different_polarity_set_is_a_distinct_inversion(client):
    _, event, obs, first = _invert_known(client, prefix="ID")
    assert first.status_code == 201

    heavier = [dict(o, weight=2.0) if i == 0 else dict(o) for i, o in enumerate(obs)]
    second = _invert(client, event["event_id"], heavier)
    assert second.status_code == 201
    assert second.json()["inversion_id"] != first.json()["inversion_id"]

    flipped = [dict(o) for o in obs]
    flipped[0] = dict(flipped[0], polarity="negative"
                      if flipped[0]["polarity"] == "positive" else "positive")
    third = _invert(client, event["event_id"], flipped)
    assert third.status_code == 201
    assert third.json()["inversion_id"] != first.json()["inversion_id"]
    assert third.json()["inversion_id"] != second.json()["inversion_id"]


def test_concurrent_identical_inversions_collapse(client):
    _, event, obs, _ = _invert_known(client, prefix="IC")
    # fresh inversion of the same event: drop one station so the request hash
    # differs from the one created above
    obs = obs[:-1]
    results = []
    barrier = threading.Barrier(2)

    def worker():
        barrier.wait()
        r = _invert(client, event["event_id"], obs)
        results.append((r.status_code, r.json()["inversion_id"]))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start(); t2.start(); t1.join(); t2.join()

    statuses = sorted(s for s, _ in results)
    assert statuses == [200, 201], results
    assert results[0][1] == results[1][1]


# --------------------------------------------------------------------------- #
# explicit rejections
# --------------------------------------------------------------------------- #
def test_unknown_event_returns_404(client):
    resp = _invert(client, "0" * 32, [{"pick_id": "f" * 32, "polarity": "positive"}])
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "event_not_found"


def test_unknown_pick_returns_404(client):
    _, event, obs, _ = _invert_known(client, prefix="UP")
    obs[0] = {"pick_id": "f" * 32, "polarity": "positive", "weight": 1.0}
    resp = _invert(client, event["event_id"], obs)
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "pick_not_found"


def test_pick_of_another_event_is_rejected(client):
    _, event_a, obs_a, _ = _invert_known(client, prefix="PA")
    _, event_b, obs_b, _ = _invert_known(client, prefix="PB")
    # a valid P pick, but associated with the other event
    obs_a[0] = obs_b[0]
    resp = _invert(client, event_a["event_id"], obs_a)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "pick_not_associated"


def test_s_phase_pick_is_rejected(client):
    data = _process(client, "SP", n_stations=8)
    event = _best(data)
    obs = _polarities_for_mechanism(event, "SP", TRUE_MECHANISM)
    s_pick = next(p for p in event["picks"] if p["phase"] == "S")
    obs[0] = {"pick_id": s_pick["pick_id"], "polarity": "positive", "weight": 1.0}
    resp = _invert(client, event["event_id"], obs)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "pick_not_associated"


def test_unassociated_pick_is_rejected(client):
    body, _ = synthetic_event_payload(prefix="UA", n_stations=8)
    body["manual_picks"] = [
        {"station": "UA00", "phase": "P", "time": BASE_TIME, "weight": 1.0}
    ]
    data = client.post("/v1/process", json=body).json()
    event = _best(data)
    stray = next(p for p in data["unassociated_picks"] if p["station"] == "UA00")
    obs = _polarities_for_mechanism(event, "UA", TRUE_MECHANISM)
    obs[0] = {"pick_id": stray["pick_id"], "polarity": "positive", "weight": 1.0}
    resp = _invert(client, event["event_id"], obs)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "pick_not_associated"


def test_contradictory_polarity_at_one_station_is_rejected(client):
    _, event, obs, _ = _invert_known(client, prefix="CP")
    dup = dict(obs[0])
    dup["polarity"] = "negative" if obs[0]["polarity"] == "positive" else "positive"
    resp = _invert(client, event["event_id"], obs + [dup])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "contradictory_polarity"


def test_fewer_than_six_stations_is_rejected(client):
    _, event, obs, _ = _invert_known(client, prefix="ST")
    resp = _invert(client, event["event_id"], obs[:5])
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "insufficient_polarity_stations"
    # exactly six independent stations is acceptable
    ok = _invert(client, event["event_id"], obs[:6])
    assert ok.status_code == 201, ok.text
    assert ok.json()["station_count"] == 6


def test_large_azimuthal_gap_is_rejected(client):
    """Stations clustered in a 70-degree wedge leave a ~290-degree gap."""
    ev_lat, ev_lon, ev_depth = 35.05, -117.95, 12.0
    model = LayeredModel.from_dicts(DEFAULT_MODEL_LAYERS)
    waveforms = []
    for i in range(8):
        az = 10.0 + i * 10.0  # azimuths 10..80 degrees
        lat = ev_lat + 0.25 * math.cos(math.radians(az))
        lon = ev_lon + 0.25 * math.sin(math.radians(az)) / math.cos(math.radians(ev_lat))
        code = f"GA{i:02d}"
        dist = distance_km(ev_lat, ev_lon, lat, lon)
        tp = model.travel_time(ev_depth, dist, "P")[0]
        ts = model.travel_time(ev_depth, dist, "S")[0]
        waveforms.append(
            {
                "station": code,
                "latitude": lat,
                "longitude": lon,
                "elevation_m": 0.0,
                "sampling_rate": 100.0,
                "start_time": BASE_TIME,
                "data_b64": make_waveform_b64(
                    code, 100.0, BASE_TIME, [(30.0 + tp, 4.0), (30.0 + ts, 6.0)]
                ),
            }
        )
    body = {
        "waveforms": waveforms,
        "manual_picks": [],
        "model": {"name": "default", "version": "1.0"},
        "params": {"max_residual_s": 1.0, "min_stations": 4},
    }
    data = client.post("/v1/process", json=body)
    assert data.status_code == 200, data.text
    event = _best(data.json())
    obs = [
        {"pick_id": p["pick_id"], "polarity": "positive", "weight": 1.0}
        for p in _p_picks(event)
    ]
    assert len(obs) >= 6
    resp = _invert(client, event["event_id"], obs)
    assert resp.status_code == 422
    assert resp.json()["detail"]["code"] == "azimuthal_gap_too_large"


# --------------------------------------------------------------------------- #
# pagination
# --------------------------------------------------------------------------- #
def test_solutions_pagination_is_stable(client):
    # all-positive polarities leave many tied best solutions to page through
    data = _process(client, "PG", n_stations=8)
    event = _best(data)
    obs = [
        {"pick_id": p["pick_id"], "polarity": "positive", "weight": 1.0}
        for p in _p_picks(event)
    ]
    rep = _invert(client, event["event_id"], obs)
    assert rep.status_code == 201, rep.text
    inversion_id = rep.json()["inversion_id"]
    total = rep.json()["solution_count"]
    assert total > 20, "expected enough tied solutions to exercise pagination"

    solutions = _all_solutions(client, inversion_id, limit=7)
    assert len(solutions) == total
    assert [s["solution_index"] for s in solutions] == list(range(total))
    keys = [s["plane_1"]["strike_deg"] for s in solutions]
    assert keys == sorted(keys)


def test_solution_cursor_bound_to_inversion(client):
    _, event, obs, first = _invert_known(client, prefix="CB")
    assert first.status_code == 201
    inv1 = first.json()["inversion_id"]
    heavier = [dict(o, weight=2.0) if i == 0 else dict(o) for i, o in enumerate(obs)]
    second = _invert(client, event["event_id"], heavier)
    assert second.status_code == 201
    inv2 = second.json()["inversion_id"]

    page = client.get(f"/v1/focal-mechanisms/{inv1}/solutions", params={"limit": 2})
    assert page.status_code == 200
    cursor = page.json()["next_cursor"]
    assert cursor is not None
    bad = client.get(
        f"/v1/focal-mechanisms/{inv2}/solutions",
        params={"limit": 2, "cursor": cursor},
    )
    assert bad.status_code == 422
    assert bad.json()["detail"]["code"] == "invalid_cursor"


def test_list_inversions_filter_and_pagination(client):
    _, event1, obs1, r1 = _invert_known(client, prefix="L1")
    _, event2, _, r2 = _invert_known(client, prefix="L2")
    assert r1.status_code == 201 and r2.status_code == 201
    inv1, inv2 = r1.json()["inversion_id"], r2.json()["inversion_id"]
    # a second, distinct inversion of event1 (different weights)
    heavier = [dict(o, weight=2.0) if i == 0 else dict(o) for i, o in enumerate(obs1)]
    r1b = _invert(client, event1["event_id"], heavier)
    assert r1b.status_code == 201
    inv1b = r1b.json()["inversion_id"]

    by_event = client.get("/v1/focal-mechanisms", params={"event_id": event1["event_id"]})
    assert by_event.status_code == 200
    assert {i["inversion_id"] for i in by_event.json()["inversions"]} == {inv1, inv1b}

    # keyset pagination walks every inversion exactly once
    seen, cursor = set(), None
    for _ in range(10):
        params = {"limit": 1}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/v1/focal-mechanisms", params=params).json()
        seen.update(i["inversion_id"] for i in data["inversions"])
        cursor = data["next_cursor"]
        if cursor is None:
            break
    assert seen == {inv1, inv1b, inv2}

    # list cursor is bound to the event filter fingerprint
    page = client.get(
        "/v1/focal-mechanisms", params={"limit": 1, "event_id": event1["event_id"]}
    ).json()
    assert page["next_cursor"] is not None
    mismatched = client.get(
        "/v1/focal-mechanisms",
        params={"limit": 1, "event_id": event2["event_id"],
                "cursor": page["next_cursor"]},
    )
    assert mismatched.status_code == 422
    assert mismatched.json()["detail"]["code"] == "invalid_cursor"


def test_get_inversion_and_not_found(client):
    _, event, obs, first = _invert_known(client, prefix="GN")
    assert first.status_code == 201
    inversion_id = first.json()["inversion_id"]

    got = client.get(f"/v1/focal-mechanisms/{inversion_id}")
    assert got.status_code == 200
    assert got.json()["created"] is False
    assert got.json()["inversion_id"] == inversion_id
    assert got.json()["solutions"] == first.json()["solutions"]

    missing = client.get("/v1/focal-mechanisms/" + "d" * 32)
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "focal_mechanism_not_found"
    missing = client.get("/v1/focal-mechanisms/" + "d" * 32 + "/solutions")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "focal_mechanism_not_found"

    unknown_event = client.get("/v1/focal-mechanisms", params={"event_id": "e" * 32})
    assert unknown_event.status_code == 404
    assert unknown_event.json()["detail"]["code"] == "event_not_found"


# --------------------------------------------------------------------------- #
# read-only: original event, run and catalog content never change
# --------------------------------------------------------------------------- #
def test_inversion_does_not_mutate_event_or_run(client):
    data, event, obs, resp = _invert_known(client, prefix="RO")
    assert resp.status_code == 201
    run_before = client.get(f"/v1/runs/{data['run_id']}").json()
    event_before = client.get(f"/v1/events/{event['event_id']}").json()

    # repeat the inversion (idempotent replay) and list everything
    _invert(client, event["event_id"], obs)
    client.get("/v1/focal-mechanisms")
    client.get(f"/v1/focal-mechanisms/{resp.json()['inversion_id']}/solutions")

    assert client.get(f"/v1/runs/{data['run_id']}").json() == run_before
    assert client.get(f"/v1/events/{event['event_id']}").json() == event_before

    # process semantics are untouched: the same window maps to the same run
    again = client.post("/v1/process", json=synthetic_event_payload(prefix="RO")[0])
    assert again.json()["run_id"] == data["run_id"]
    assert again.json()["created"] is False
