"""Turn the expert's (acceleration, curvature) output into ego waypoints.

Alpamayo's action space is a unicycle parameterisation: 64 steps at dt = 0.1 s,
each carrying a longitudinal acceleration and a path curvature, both stored
normalised. Constants below are verbatim from the released config.json.

Validate against `pred_xyz` in the golden dump before trusting the geometry --
the integration order (whether velocity updates before or after the pose) shifts
the trajectory by roughly half a step, which is small enough to look right and
large enough to ruin a minADE.
"""
import numpy as np

DT = 0.1
N_WAYPOINTS = 64
ACCEL_MEAN, ACCEL_STD = 0.02902694707164455, 0.6810426736454882
CURV_MEAN, CURV_STD = 0.0002692167976330542, 0.026148280660833106
ACCEL_BOUNDS = (-9.8, 9.8)
CURV_BOUNDS = (-0.33, 0.33)


def denormalize(action):
    """action: [..., 64, 2] normalised -> physical (m/s^2, 1/m)."""
    a = np.asarray(action, dtype=np.float64)
    accel = np.clip(a[..., 0] * ACCEL_STD + ACCEL_MEAN, *ACCEL_BOUNDS)
    curv = np.clip(a[..., 1] * CURV_STD + CURV_MEAN, *CURV_BOUNDS)
    return accel, curv


def integrate(accel, curv, v0=0.0, dt=DT):
    """Forward-Euler unicycle rollout in the ego frame at t = 0.

    Returns xy [..., 64, 2] and heading [..., 64] in radians.
    """
    accel = np.asarray(accel, dtype=np.float64)
    lead = accel.shape[:-1]
    n = accel.shape[-1]
    x = np.zeros(lead); y = np.zeros(lead)
    theta = np.zeros(lead); v = np.full(lead, float(v0))
    xs, ys, ths = [], [], []
    for i in range(n):
        x = x + v * np.cos(theta) * dt
        y = y + v * np.sin(theta) * dt
        theta = theta + v * curv[..., i] * dt
        v = np.maximum(v + accel[..., i] * dt, 0.0)
        xs.append(x); ys.append(y); ths.append(theta)
    return np.stack([np.stack(xs, -1), np.stack(ys, -1)], -1), np.stack(ths, -1)


def action_to_waypoints(action, v0=0.0):
    accel, curv = denormalize(action)
    return integrate(accel, curv, v0=v0)


def min_ade(pred_xy, gt_xy):
    """pred_xy [K, 64, 2], gt_xy [64, 2] -> best per-sample average displacement."""
    d = np.linalg.norm(np.asarray(pred_xy) - np.asarray(gt_xy)[None], axis=-1)
    return float(d.mean(axis=-1).min())
