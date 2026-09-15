"""Turn the expert's (acceleration, curvature) output into ego waypoints.

Alpamayo's action space is a unicycle: 64 steps of 0.1 s, each carrying a
longitudinal acceleration and a path curvature, both stored normalised. The
integration below is transcribed from the reference
(alpamayo_r1.action_space.unicycle_accel_curvature.action_to_traj): trapezoidal
in position, with the kappa*accel*dt^2/2 term in the heading. Plain forward Euler
drifts from it.

It integrates from the car's CURRENT SPEED. That matters more than anything else
here: with v0 = 0 a correct model produces a few metres of travel instead of ~57,
because the car is assumed to start from standstill.
"""
import numpy as np

DT = 0.1
N_WAYPOINTS = 64
ACCEL_MEAN, ACCEL_STD = 0.02902694707164455, 0.6810426736454882
CURV_MEAN, CURV_STD = 0.0002692167976330542, 0.026148280660833106
ACCEL_BOUNDS = (-9.8, 9.8)
CURV_BOUNDS = (-0.33, 0.33)


def denormalize(action):
    """action: [..., 64, 2] normalised -> physical (m/s^2, 1/m). The reference does
    not clip here; the bounds are for reporting an out-of-range prediction."""
    a = np.asarray(action, dtype=np.float64)
    return a[..., 0] * ACCEL_STD + ACCEL_MEAN, a[..., 1] * CURV_STD + CURV_MEAN


def integrate(accel, curv, v0=0.0, dt=DT):
    """Unicycle rollout in the ego frame at t = 0, as the reference does it.

    v     = [v0, v0 + cumsum(a dt)]                                  (N+1)
    theta = [0,  cumsum(k v[:-1] dt) + cumsum(k a dt^2/2)]           (N+1)
    x     = cumsum(v[:-1] cos th[:-1] dt/2) + cumsum(v[1:] cos th[1:] dt/2)

    Returns xy [..., 64, 2] and heading [..., 64] in radians.
    """
    accel = np.asarray(accel, dtype=np.float64)
    curv = np.asarray(curv, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)
    zeros = np.zeros(accel.shape[:-1] + (1,))
    v = np.concatenate([zeros + v0[..., None] if v0.ndim else zeros + v0,
                        (zeros + v0[..., None] if v0.ndim else zeros + v0)
                        + np.cumsum(accel * dt, -1)], -1)
    theta = np.concatenate([zeros, np.cumsum(curv * v[..., :-1] * dt, -1)
                            + np.cumsum(curv * accel * (0.5 * dt * dt), -1)], -1)
    half = 0.5 * dt
    x = (np.cumsum(v[..., :-1] * np.cos(theta[..., :-1]) * half, -1)
         + np.cumsum(v[..., 1:] * np.cos(theta[..., 1:]) * half, -1))
    y = (np.cumsum(v[..., :-1] * np.sin(theta[..., :-1]) * half, -1)
         + np.cumsum(v[..., 1:] * np.sin(theta[..., 1:]) * half, -1))
    return np.stack([x, y], -1), theta[..., 1:]


def action_to_waypoints(action, v0=0.0):
    accel, curv = denormalize(action)
    return integrate(accel, curv, v0=v0)


def estimate_v0(history_xyz, dt=DT):
    """Speed at t = 0 from the ego history, [..., N, 3] in the ego frame.

    A stand-in for the reference's regularised least-squares fit over the whole
    history (action_space.estimate_t0_states). This averages the last three steps,
    which on the golden clip agrees with it to about 1%. Prefer the exact value:
    a3b_fixtures.py writes it into fixtures/meta.json as "v0".
    """
    p = np.asarray(history_xyz, dtype=np.float64).reshape(-1, np.shape(history_xyz)[-1])[:, :2]
    if len(p) < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(p[-4:], axis=0), axis=-1)
    return float(steps.mean() / dt)


def min_ade(pred_xy, gt_xy):
    """pred_xy [K, 64, 2], gt_xy [64, 2] -> best per-sample average displacement."""
    d = np.linalg.norm(np.asarray(pred_xy) - np.asarray(gt_xy)[None], axis=-1)
    return float(d.mean(axis=-1).min())
