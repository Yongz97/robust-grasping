# GraspBench pilot results

10 scenes per condition; seeds 0–9. Policies are evaluated on matching initial scenes. Success is scored using simulator truth.

| Condition | Policy | Success | 95% interval | Mean attempts | Mean simulated time |
|---|---|---:|---:|---:|---:|
| clean | one_attempt | 10/10 | 72%–100% | 1.00 | 11.8 s |
| clean | recovery | 10/10 | 72%–100% | 1.00 | 11.8 s |
| noise_6mm | one_attempt | 10/10 | 72%–100% | 1.00 | 11.8 s |
| noise_6mm | recovery | 10/10 | 72%–100% | 1.00 | 11.8 s |
| noise_16mm | one_attempt | 9/10 | 60%–98% | 1.00 | 11.7 s |
| noise_16mm | recovery | 10/10 | 72%–100% | 1.30 | 14.5 s |
| forced_first_miss | one_attempt | 0/10 | 0%–28% | 1.00 | 9.9 s |
| forced_first_miss | recovery | 10/10 | 72%–100% | 2.00 | 19.8 s |

Visual task-decision errors: 0 false positives and 1 false negatives across all episodes. See summary.json for the breakdown by condition.

Noise is Gaussian error added to the estimated grasp x/y, not raw sensor noise. The forced-miss condition injects a 7.5-cm y error on attempt one only. It is a recovery unit scenario, not evidence of general robustness.

The retry policy has a larger action/time budget. The comparison measures the value of retries with fresh observations and changed yaw, not the isolated value of adaptive recovery versus an equal-budget retry baseline.

This is a single known cube with ideal camera calibration, fixed lighting and friction. These results do not establish performance on hardware or unseen objects.
