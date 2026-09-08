"""Run a demo or a paired, seeded evaluation without a display or GPU."""
import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from .controller import Config, run
from .perception import detect_cube
from .simulation import Scene


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def nonnegative(value):
    value = float(value)
    if value < 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def episode(seed, config, gui=False, video=None, snapshot=None):
    start = time.perf_counter()
    with Scene(seed=seed, gui=gui, video=video) as scene:
        before = scene.evaluation_state()
        if snapshot:
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snapshot), cv2.cvtColor(scene.frame(), cv2.COLOR_RGB2BGR))
            obs = scene.observe()
            detection = detect_cube(obs)
            if detection:
                x, y, w, h = detection.bbox
                cv2.rectangle(obs.rgb, (x, y), (x + w, y + h), (0, 255, 180), 2)
            cv2.imwrite(str(snapshot.with_name("camera.png")), cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR))
        result = run(scene, seed, config)
        after = scene.evaluation_state()
        xyz = np.array(after["position"])
        # Whole cube must be approximately inside the 13-cm goal square and settled.
        truth_success = bool(np.max(np.abs(xyz[:2] - scene.goal[:2])) < 0.045
                             and abs(xyz[2] - scene.side / 2) < 0.01
                             and after["speed"] < 0.02)
        error = (float(np.linalg.norm(np.array(result.initial_estimate) - before["position"]))
                 if result.initial_estimate is not None else None)
        return {**asdict(result), "config": asdict(config), "success": truth_success,
                "initial_localization_error_m": error, "initial_state": before, "final_state": after,
                "sim_seconds": round(scene.steps * scene.dt, 3),
                "wall_seconds": round(time.perf_counter() - start, 3)}


def interval(successes, n):
    """Wilson 95% confidence interval, useful even for tiny pilot samples."""
    z = 1.96
    rate = successes / n
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def benchmark(args):
    args.out.mkdir(parents=True, exist_ok=True)
    scenarios = {"clean": (0, False), "noise_6mm": (0.006, False),
                 "noise_16mm": (0.016, False), "forced_first_miss": (0, True)}
    rows = []
    # Each policy gets an identical freshly seeded scene and first-attempt noise draw.
    for scenario, (noise, miss) in scenarios.items():
        for policy, attempts in (("one_attempt", 1), ("recovery", 3)):
            for seed in range(args.seed, args.seed + args.episodes):
                result = episode(seed, Config(attempts, noise, miss))
                result.update(scenario=scenario, policy=policy)
                rows.append(result)
                print(f"{scenario:20s} {policy:12s} seed={seed:3d} "
                      f"success={result['success']} attempts={result['attempts']}", flush=True)
                # Save completed trials incrementally, including failure trajectories.
                with (args.out / "episodes.jsonl").open("a" if len(rows) > 1 else "w") as stream:
                    stream.write(json.dumps(result) + "\n")
    groups = []
    for scenario in scenarios:
        for policy in ("one_attempt", "recovery"):
            selected = [r for r in rows if r["scenario"] == scenario and r["policy"] == policy]
            successes = sum(r["success"] for r in selected)
            errors = [r["initial_localization_error_m"] for r in selected
                      if r["initial_localization_error_m"] is not None]
            groups.append({"scenario": scenario, "policy": policy, "episodes": len(selected),
                           "successes": successes, "success_rate": successes / len(selected),
                           "success_ci95": interval(successes, len(selected)),
                           "mean_attempts": float(np.mean([r["attempts"] for r in selected])),
                           "mean_sim_seconds": float(np.mean([r["sim_seconds"] for r in selected])),
                           "mean_localization_error_mm": float(np.mean(errors) * 1000) if errors else None,
                           "visual_false_positives": sum(r["visual_success"] and not r["success"] for r in selected),
                           "visual_false_negatives": sum(r["success"] and not r["visual_success"] for r in selected)})
    metadata = {"python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: version(name) for name in ("pybullet", "numpy", "opencv-python-headless")},
                "seed_start": args.seed, "episodes_per_condition": args.episodes}
    (args.out / "summary.json").write_text(json.dumps({"metadata": metadata, "groups": groups}, indent=2) + "\n")
    fields = ["scenario", "policy", "seed", "success", "visual_success", "attempts",
              "initial_localization_error_m", "sim_seconds", "wall_seconds"]
    with (args.out / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = ["# GraspBench pilot results", "",
              f"{args.episodes} scenes per condition; seeds {args.seed}–{args.seed + args.episodes - 1}. "
              "Policies are evaluated on matching initial scenes. Success is scored using simulator truth.", "",
              "| Condition | Policy | Success | 95% interval | Mean attempts | Mean simulated time |",
              "|---|---|---:|---:|---:|---:|"]
    for group in groups:
        lo, hi = group["success_ci95"]
        report.append(f"| {group['scenario']} | {group['policy']} | {group['successes']}/{group['episodes']} "
                      f"| {lo:.0%}–{hi:.0%} | {group['mean_attempts']:.2f} | {group['mean_sim_seconds']:.1f} s |")
    report += ["", f"Visual task-decision errors: {sum(g['visual_false_positives'] for g in groups)} "
               f"false positives and {sum(g['visual_false_negatives'] for g in groups)} false negatives "
               "across all episodes. See summary.json for the breakdown by condition.", "",
               "Noise is Gaussian error added to the estimated grasp x/y, not raw sensor noise. "
               "The forced-miss condition injects a 7.5-cm y error on attempt one only. "
               "It is a recovery unit scenario, not evidence of general robustness.", "",
               "The retry policy has a larger action/time budget. The comparison measures the value "
               "of retries with fresh observations and changed yaw, not the isolated value of adaptive "
               "recovery versus an equal-budget retry baseline.", "",
               "This is a single known cube with ideal camera calibration, fixed lighting and friction. "
               "These results do not establish performance on hardware or unseen objects.", ""]
    (args.out / "report.md").write_text("\n".join(report))
    print("\n" + "\n".join(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a physical grasp and save a trace")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--attempts", type=positive, default=3)
    demo.add_argument("--noise", type=nonnegative, default=0)
    demo.add_argument("--first-miss", action="store_true", help="Inject an explicit first-attempt positioning fault")
    demo.add_argument("--gui", action="store_true", help="Open PyBullet's real-time window")
    demo.add_argument("--video", type=Path, help="Save an annotated MP4; rendering is slower than headless runs")
    demo.add_argument("--out", type=Path, default=Path("outputs/demo"))
    bench = sub.add_parser("benchmark", help="Compare one attempt and three attempts across four conditions")
    bench.add_argument("--episodes", type=positive, default=10, help="Scenes PER policy PER condition (total: 8x)")
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--out", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.command == "benchmark":
        benchmark(args)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        result = episode(args.seed, Config(args.attempts, args.noise, args.first_miss),
                         args.gui, args.video, args.out / "scene.png")
        (args.out / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
