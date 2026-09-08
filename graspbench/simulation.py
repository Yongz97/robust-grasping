"""Panda scene, physical finger grasping, and camera observations."""
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from .perception import Observation, detect_cube


class Scene:
    dt = 1 / 240
    home = np.array([0.40, 0.00, 0.40])
    goal = np.array([0.50, -0.23, 0.025])
    side = 0.04

    def __init__(self, seed=0, gui=False, video: Path | None = None):
        self.p = BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        self.gui = gui
        self.steps = 0
        self.state = "INITIALIZING"
        self.detail = ""
        self.writer = None
        if video:
            video.parent.mkdir(parents=True, exist_ok=True)
            self.writer = imageio.get_writer(str(video), fps=20, codec="libx264", quality=8)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -9.81)
        self.p.setTimeStep(self.dt)
        self.p.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1)
        self.p.loadURDF("plane.urdf", [0, 0, -0.06])
        self._box([0.43, 0, -0.03], [0.55, 0.46, 0.03], [0.30, 0.36, 0.40, 1])
        self._box([*self.goal[:2], 0.001], [0.065, 0.065, 0.001], [0.18, 0.70, 0.57, 1], collision=False)
        self.robot = self.p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        joints = [0, -0.45, 0, -2.35, 0, 1.95, 0.78]
        for i, value in enumerate(joints):
            self.p.resetJointState(self.robot, i, value)
        for j in (9, 10):
            self.p.resetJointState(self.robot, j, 0.04)
            self.p.changeDynamics(self.robot, j, lateralFriction=1.5, spinningFriction=0.01,
                                  frictionAnchor=1)
        gear = self.p.createConstraint(self.robot, 9, self.robot, 10, p.JOINT_GEAR,
                                      [1, 0, 0], [0, 0, 0], [0, 0, 0])
        self.p.changeConstraint(gear, gearRatio=-1, erp=0.1, maxForce=50)
        rng = np.random.default_rng(seed)
        pos = [rng.uniform(0.43, 0.58), rng.uniform(-0.10, 0.14), self.side / 2 + 0.001]
        yaw = rng.uniform(-np.pi, np.pi)
        self.cube = self._box(pos, [self.side / 2] * 3, [0.90, 0.055, 0.045, 1], mass=0.06,
                              orientation=self.p.getQuaternionFromEuler([0, 0, yaw]))
        self.p.changeDynamics(self.cube, -1, lateralFriction=1.0, spinningFriction=0.005,
                              restitution=0, linearDamping=0.04, angularDamping=0.04)
        self.view = self.p.computeViewMatrix([0.95, -0.65, 0.80], [0.48, 0, 0.10], [0, 0, 1])
        self.projection = self.p.computeProjectionMatrixFOV(52, 1, 0.05, 2.5)
        self.show_view = self.p.computeViewMatrix([1.35, -1.35, 1.15], [0.30, 0, 0.30], [0, 0, 1])
        self.show_projection = self.p.computeProjectionMatrixFOV(48, 960 / 640, 0.05, 3)
        self.p.resetDebugVisualizerCamera(1.3, 135, -35, [0.35, 0, 0.12])
        self.open_gripper()
        self.move(self.home)

    def _box(self, pos, half, color, mass=0, collision=True, orientation=(0, 0, 0, 1)):
        visual = self.p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        shape = self.p.createCollisionShape(p.GEOM_BOX, halfExtents=half) if collision else -1
        return self.p.createMultiBody(mass, shape, visual, pos, orientation)

    def close(self):
        if self.writer:
            self.writer.close()
        self.p.disconnect()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def step(self, count):
        for _ in range(count):
            self.p.stepSimulation()
            self.steps += 1
            if self.writer and self.steps % 12 == 0:
                self.writer.append_data(self.frame())
            if self.gui:
                time.sleep(self.dt)

    def observe(self):
        image = self.p.getCameraImage(320, 320, self.view, self.projection,
                                      renderer=p.ER_TINY_RENDERER, flags=p.ER_NO_SEGMENTATION_MASK)
        return Observation(np.asarray(image[2], dtype=np.uint8).reshape(320, 320, 4)[:, :, :3].copy(),
                           np.asarray(image[3]).reshape(320, 320), np.array(self.view).reshape(4, 4, order="F"),
                           np.array(self.projection).reshape(4, 4, order="F"))

    def frame(self):
        image = self.p.getCameraImage(960, 640, self.show_view, self.show_projection,
                                      renderer=p.ER_TINY_RENDERER, flags=p.ER_NO_SEGMENTATION_MASK)
        rgb = np.asarray(image[2], dtype=np.uint8).reshape(640, 960, 4)[:, :, :3].copy()
        rgb[:92] = [17, 25, 36]
        cv2.putText(rgb, "GRASPBENCH / vision + recovery", (24, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (233, 241, 248), 1, cv2.LINE_AA)
        cv2.putText(rgb, f"{self.state}   {self.detail}", (24, 67), cv2.FONT_HERSHEY_SIMPLEX,
                    0.54, (98, 221, 183), 1, cv2.LINE_AA)
        observation = self.observe()
        detection = detect_cube(observation)
        if detection:
            x, y, w, h = detection.bbox
            cv2.rectangle(observation.rgb, (x, y), (x + w, y + h), (98, 255, 183), 2)
        rgb[108:316, 736:944] = cv2.resize(observation.rgb, (208, 208))
        rgb[316:343, 736:944] = [17, 25, 36]
        cv2.putText(rgb, "ROBOT CAMERA / RGB-D", (745, 334), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (233, 241, 248), 1, cv2.LINE_AA)
        rgb[606:] = [17, 25, 36]
        cv2.putText(rgb, "CPU simulation  |  physical finger contact  |  green square = placement goal",
                    (24, 628), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 205, 218), 1, cv2.LINE_AA)
        return rgb

    def tcp(self):
        return np.array(self.p.getLinkState(self.robot, 11, computeForwardKinematics=True)[4])

    def move(self, target, yaw=0, duration=0.8):
        target = np.asarray(target, dtype=float)
        if not (0.20 <= target[0] <= 0.72 and abs(target[1]) <= 0.38 and 0.016 <= target[2] <= 0.55):
            raise ValueError(f"Target outside the configured tabletop workspace: {target}")
        start = self.tcp()
        orientation = self.p.getQuaternionFromEuler([np.pi, 0, yaw])
        count = max(1, int(duration / self.dt))
        # Cartesian interpolation with motor control; no joint teleportation during a trial.
        for i in range(count):
            t = (i + 1) / count
            smooth = t * t * (3 - 2 * t)
            point = start + (target - start) * smooth
            joints = self.p.calculateInverseKinematics(self.robot, 11, point.tolist(), orientation,
                                                       maxNumIterations=80, residualThreshold=1e-5)
            self.p.setJointMotorControlArray(self.robot, list(range(7)), p.POSITION_CONTROL,
                                             targetPositions=joints[:7], forces=[120] * 7,
                                             positionGains=[0.18] * 7)
            self.step(1)
        self.step(48)

    def open_gripper(self):
        self.p.setJointMotorControlArray(self.robot, [9, 10], p.POSITION_CONTROL,
                                         targetPositions=[0.04, 0.04], forces=[30, 30])
        self.step(90)

    def close_gripper(self):
        self.p.setJointMotorControlArray(self.robot, [9, 10], p.POSITION_CONTROL,
                                         targetPositions=[0, 0], forces=[30, 30])
        self.step(150)

    def evaluation_state(self):
        """Oracle reserved for final benchmark scoring; never consumed by the policy."""
        position, orientation = self.p.getBasePositionAndOrientation(self.cube)
        linear, angular = self.p.getBaseVelocity(self.cube)
        return {"position": list(position), "orientation": list(orientation),
                "speed": float(np.linalg.norm(linear)), "angular_speed": float(np.linalg.norm(angular))}
