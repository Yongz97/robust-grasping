from dataclasses import replace

import numpy as np
import pytest

from graspbench.cli import episode, interval
from graspbench.controller import Config, run
from graspbench.perception import backproject, detect_cube
from graspbench.simulation import Scene


@pytest.mark.parametrize("seed", [0, 3, 7, 11, 29])
def test_rgbd_localizes_randomized_cube(seed):
    with Scene(seed) as scene:
        obs = scene.observe()
        detection = detect_cube(obs)
        assert detection is not None
        actual = scene.evaluation_state()["position"]
        assert np.linalg.norm(detection.center - actual) < 0.012
        assert detect_cube(replace(obs, rgb=np.zeros_like(obs.rgb))) is None
        # Depth/backprojection must return visible cube surfaces close to table height.
        rows, cols = np.nonzero(detection.mask)
        points = backproject(obs, rows, cols)
        assert np.percentile(points[:, 2], 97) == pytest.approx(0.04, abs=0.004)


def test_physical_pick_place():
    result = episode(7, Config(max_attempts=1))
    assert result["success"] and result["visual_success"]
    assert result["attempts"] == 1
    assert any(e["state"] == "LIFT_CONFIRMED" for e in result["events"])


def test_recovery_after_first_miss():
    baseline = episode(7, Config(max_attempts=1, first_miss=True))
    recovered = episode(7, Config(max_attempts=3, first_miss=True))
    assert not baseline["success"] and not baseline["visual_success"]
    assert recovered["success"] and recovered["visual_success"]
    assert recovered["attempts"] == 2
    states = [e["state"] for e in recovered["events"]]
    assert states.index("LIFT_NOT_CONFIRMED") < states.index("RECOVER") < states.index("SUCCESS")
    assert states.count("OBSERVE") == 2


def test_controller_does_not_read_object_oracle(monkeypatch):
    with Scene(7) as scene:
        def forbidden(*args, **kwargs):
            raise AssertionError("Policy accessed simulator ground truth")
        monkeypatch.setattr(scene, "evaluation_state", forbidden)
        monkeypatch.setattr(scene.p, "getBasePositionAndOrientation", forbidden)
        assert run(scene, 7, Config()).visual_success


def test_missing_target_fails_without_grasping(monkeypatch):
    with Scene(7) as scene:
        obs = scene.observe()
        blank = replace(obs, rgb=np.zeros_like(obs.rgb))
        monkeypatch.setattr(scene, "observe", lambda: blank)
        result = run(scene, 7, Config(max_attempts=2))
        assert not result.visual_success
        assert all(e["state"] != "CLOSE_FINGERS" for e in result.events)


def test_interval_exposes_small_sample_uncertainty():
    lo, hi = interval(10, 10)
    assert lo < 0.75 and hi == pytest.approx(1)
    lo, hi = interval(0, 10)
    assert lo == pytest.approx(0) and hi > 0.25
