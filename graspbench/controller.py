"""Observe -> grasp -> visually verify -> place or re-observe and retry."""
from dataclasses import dataclass, field

import numpy as np

from .perception import detect_cube
from .simulation import Scene


@dataclass
class Config:
    max_attempts: int = 3
    localization_noise: float = 0.0  # Gaussian std in metres, added to estimated x/y
    first_miss: bool = False  # explicit 7.5-cm transient positioning fault


@dataclass
class Result:
    seed: int
    visual_success: bool = False
    attempts: int = 0
    events: list = field(default_factory=list)
    initial_estimate: list | None = None


def run(scene: Scene, seed: int, config: Config) -> Result:
    if config.max_attempts < 1 or config.localization_noise < 0:
        raise ValueError("Attempts must be positive and noise non-negative")
    result = Result(seed)
    rng = np.random.default_rng(seed + 100_000)

    def event(state, **data):
        scene.state = state
        scene.detail = f"attempt {result.attempts}/{config.max_attempts}"
        if config.first_miss and result.attempts == 1:
            scene.detail += " | injected +75mm positioning error"
        result.events.append({"state": state, "sim_time": round(scene.steps * scene.dt, 3), **data})

    def visible_lift():
        # Require two observations with the cube raised and close to the measured TCP.
        for _ in range(2):
            detection = detect_cube(scene.observe())
            if detection is None:
                return False
            if detection.center[2] < 0.12 or np.linalg.norm(detection.center - scene.tcp()) > 0.085:
                return False
            scene.step(36)
        return True

    for attempt in range(config.max_attempts):
        result.attempts = attempt + 1
        event("OBSERVE")
        detection = detect_cube(scene.observe())
        if detection is None:
            event("TARGET_NOT_VISIBLE")
            scene.move(scene.home)
            continue
        if result.initial_estimate is None:
            result.initial_estimate = detection.center.tolist()
        position = detection.center.copy()
        position[:2] += rng.normal(0, config.localization_noise, 2)
        if config.first_miss and attempt == 0:
            position[1] += 0.075
            event("INJECTED_POSITION_ERROR", offset_m=[0, 0.075, 0])
        if not (0.25 < position[0] < 0.68 and abs(position[1]) < 0.33):
            event("ESTIMATE_OUT_OF_WORKSPACE")
            continue
        yaw = [0, np.pi / 2, -np.pi / 4, np.pi / 4][attempt % 4]
        position[2] = np.clip(position[2] + 0.005, 0.021, 0.055)
        above = np.array([*position[:2], 0.25])
        event("APPROACH", estimated_center=detection.center.tolist(), target=position.tolist(), yaw=float(yaw))
        scene.open_gripper()
        scene.move(above, yaw)
        event("DESCEND")
        scene.move(position, yaw, duration=0.8)
        event("CLOSE_FINGERS")
        scene.close_gripper()
        event("LIFT")
        scene.move(above, yaw, duration=1.0)
        event("VERIFY_LIFT")
        lifted = visible_lift()
        event("LIFT_CONFIRMED" if lifted else "LIFT_NOT_CONFIRMED")
        if lifted:
            event("TRANSPORT")
            scene.move([*scene.goal[:2], 0.25], yaw, duration=1.2)
            event("PLACE")
            scene.move(scene.goal, yaw, duration=0.9)
            scene.open_gripper()
            scene.move([*scene.goal[:2], 0.25], yaw)
            scene.move(scene.home)
            scene.step(120)
            placed = detect_cube(scene.observe())
            result.visual_success = bool(placed is not None
                                         and np.max(np.abs(placed.center[:2] - scene.goal[:2])) < 0.045
                                         and abs(placed.center[2] - scene.side / 2) < 0.025)
            if result.visual_success:
                event("SUCCESS")
                scene.step(120)
                return result
            event("PLACEMENT_NOT_CONFIRMED")
        # Lower before opening so an uncertain held object is not released from height.
        event("RECOVER")
        if not lifted:
            scene.move([*position[:2], 0.07], yaw)
            scene.open_gripper()
            scene.move(above, yaw)
        scene.move(scene.home)
        scene.step(120)
    event("FAILED")
    scene.step(120)
    return result
