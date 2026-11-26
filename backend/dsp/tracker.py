"""2D constant-velocity Kalman tracker with Mahalanobis innovation gating for moving sources."""

from __future__ import annotations

import math

import numpy as np

from shared.protocol import RoomConfig


class MovingTracker2D:
    """Track a moving 2D acoustic source while rejecting sudden multipath outliers."""

    def __init__(self, x_m: float, y_m: float) -> None:
        self._x = np.array([float(x_m), float(y_m), 0.0, 0.0], dtype=np.float64)
        self._P = np.diag([0.25, 0.25, 0.5, 0.5])
        self._rejected_streak = 0
        self._H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)

    @staticmethod
    def _transition(dt_s: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(1e-3, float(dt_s))
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.82, 0.0],
                [0.0, 0.0, 0.0, 0.82],
            ],
            dtype=np.float64,
        )
        q_a = 1.2 ** 2
        Q = q_a * np.array(
            [
                [0.25 * dt ** 4, 0.0, 0.5 * dt ** 3, 0.0],
                [0.0, 0.25 * dt ** 4, 0.0, 0.5 * dt ** 3],
                [0.5 * dt ** 3, 0.0, dt ** 2, 0.0],
                [0.0, 0.5 * dt ** 3, 0.0, dt ** 2],
            ],
            dtype=np.float64,
        )
        Q[0, 0] += 0.01
        Q[1, 1] += 0.01
        return F, Q

    @staticmethod
    def _clamp_xy(x_m: float, y_m: float, room: RoomConfig) -> tuple[float, float]:
        return (
            float(min(max(x_m, 0.0), room.width_m)),
            float(min(max(y_m, 0.0), room.height_m)),
        )

    def position(self) -> tuple[float, float]:
        return float(self._x[0]), float(self._x[1])

    def peek_prediction(self, dt_s: float, room: RoomConfig) -> tuple[float, float, float]:
        """Return non-mutating predicted (x_m, y_m, prior_sigma_m) for the next frame."""
        F, Q = self._transition(dt_s)
        x_pred = F @ self._x
        P_pred = F @ self._P @ F.T + Q
        px, py = self._clamp_xy(float(x_pred[0]), float(x_pred[1]), room)
        sigma = float(np.clip(math.hypot(math.sqrt(max(0.0, P_pred[0, 0])), math.sqrt(max(0.0, P_pred[1, 1]))) * 1.8, 1.0, 2.2))
        return px, py, sigma

    def step(
        self,
        dt_s: float,
        meas_x: float,
        meas_y: float,
        *,
        residual_ms: float,
        confidence: float,
        n_inliers: int,
        n_pairs: int,
        room: RoomConfig,
    ) -> tuple[float, float, bool]:
        """Advance state one step and return (x_m, y_m, accepted)."""
        F, Q = self._transition(dt_s)
        H = self._H
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

        inlier_frac = float(n_inliers) / float(max(1, n_pairs))
        bad_quality = (n_pairs >= 6 and n_inliers < 3) or residual_ms > 1.1 or confidence < 0.45
        if bad_quality:
            self._x[0], self._x[1] = self._clamp_xy(float(self._x[0]), float(self._x[1]), room)
            return float(self._x[0]), float(self._x[1]), False

        sigma_m = max(0.25, 0.20 + 0.6 * float(residual_ms)) / max(0.35, inlier_frac)
        R = np.diag([sigma_m * sigma_m, sigma_m * sigma_m])
        z = np.array([float(meas_x), float(meas_y)], dtype=np.float64)
        nu = z - H @ self._x
        S = H @ self._P @ H.T + R
        S_inv = np.linalg.inv(S)
        d2 = float(nu.T @ S_inv @ nu)

        if d2 > 9.21 and self._rejected_streak < 3:
            self._rejected_streak += 1
            self._x[0], self._x[1] = self._clamp_xy(float(self._x[0]), float(self._x[1]), room)
            return float(self._x[0]), float(self._x[1]), False

        self._rejected_streak = 0
        K = self._P @ H.T @ S_inv
        self._x = self._x + K @ nu
        self._P = (np.eye(4, dtype=np.float64) - K @ H) @ self._P
        self._x[0], self._x[1] = self._clamp_xy(float(self._x[0]), float(self._x[1]), room)
        return float(self._x[0]), float(self._x[1]), True
