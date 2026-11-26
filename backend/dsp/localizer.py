"""Hybrid 2D SRP-PHAT global heatmap + cycle-consistent Huber multilateration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter1d
from scipy.optimize import least_squares

from shared.protocol import (
    Heatmap2D,
    LocalizationResult,
    PairTdoa,
    PhoneConfig,
    RoomConfig,
    speed_of_sound,
)
from backend.dsp.gcc_phat import compute_gcc_phat
from backend.dsp.trigger import EventWindow


@dataclass(slots=True)
class _GridCache:
    key: tuple
    pids: tuple[str, ...]
    phone_xy: np.ndarray  # (M, 2)
    pairs: list[tuple[int, int]]
    pair_max_tau: np.ndarray  # (P,)
    grid_x: np.ndarray  # (nx,)
    grid_y: np.ndarray  # (ny,)
    tau_grid: np.ndarray  # (P, ny, nx)


def _srp_pair_contribution(
    acc_corr: np.ndarray,
    lags_ref: np.ndarray,
    tau_grid_p: np.ndarray,
) -> np.ndarray:
    """Evaluate dilated GCC-PHAT curve over a pair's 2D room delay grid."""
    if len(lags_ref) < 2:
        return np.zeros_like(tau_grid_p)
    dt = float(lags_ref[1] - lags_ref[0])
    dilate_bins = max(1, int(round(0.00035 / dt)))
    sc = maximum_filter1d(acc_corr, size=2 * dilate_bins + 1)
    f_idx = (tau_grid_p - lags_ref[0]) / dt
    i0 = np.clip(np.floor(f_idx).astype(np.intp), 0, len(sc) - 2)
    frac = np.clip(f_idx - i0, 0.0, 1.0)
    return sc[i0] * (1.0 - frac) + sc[i0 + 1] * frac


class AcousticLocalizer2D:
    """Two-stage 2D acoustic localizer: vectorized SRP-PHAT -> robust Huber least-squares."""

    def __init__(
        self,
        grid_spacing_m: float = 0.15,
        *,
        f_min: float = 300.0,
        f_max: float = 6000.0,
        phat_beta: float = 0.8,
        cycle_threshold_s: float = 0.0015,
        loo_residual_threshold_s: float = 0.0005,
    ) -> None:
        self.grid_spacing_m = grid_spacing_m
        self.f_min = f_min
        self.f_max = f_max
        self.phat_beta = phat_beta
        self.cycle_threshold_s = cycle_threshold_s
        self.loo_residual_threshold_s = loo_residual_threshold_s
        self._cache: _GridCache | None = None

    def _get_grid(
        self,
        pids: list[str],
        phones: dict[str, PhoneConfig],
        room: RoomConfig,
        c: float,
    ) -> _GridCache:
        pids_t = tuple(pids)
        coords_t = tuple((pid, round(phones[pid].x_m, 4), round(phones[pid].y_m, 4)) for pid in pids_t)
        key = (round(room.width_m, 3), round(room.height_m, 3), round(c, 2), coords_t)
        if self._cache is not None and self._cache.key == key:
            return self._cache

        nx = max(12, int(round(room.width_m / self.grid_spacing_m)) + 1)
        ny = max(12, int(round(room.height_m / self.grid_spacing_m)) + 1)
        grid_x = np.linspace(0.0, room.width_m, nx, dtype=np.float64)
        grid_y = np.linspace(0.0, room.height_m, ny, dtype=np.float64)
        xx, yy = np.meshgrid(grid_x, grid_y)  # shape (ny, nx)

        phone_xy = np.array([[phones[pid].x_m, phones[pid].y_m] for pid in pids_t], dtype=np.float64)
        dists = np.sqrt(
            (xx[None, :, :] - phone_xy[:, 0, None, None]) ** 2
            + (yy[None, :, :] - phone_xy[:, 1, None, None]) ** 2
        )

        pairs: list[tuple[int, int]] = []
        tau_list: list[np.ndarray] = []
        max_tau_list: list[float] = []
        m = len(pids_t)
        for i in range(m):
            for j in range(i + 1, m):
                pairs.append((i, j))
                tau_list.append((dists[i] - dists[j]) / c)
                d_ij = float(np.linalg.norm(phone_xy[i] - phone_xy[j]))
                max_tau_list.append(d_ij / c + 0.0005)

        cache = _GridCache(
            key=key,
            pids=pids_t,
            phone_xy=phone_xy,
            pairs=pairs,
            pair_max_tau=np.array(max_tau_list, dtype=np.float64),
            grid_x=grid_x,
            grid_y=grid_y,
            tau_grid=np.stack(tau_list, axis=0),
        )
        self._cache = cache
        return cache

    def localize_event(
        self,
        event: EventWindow,
        phones: dict[str, PhoneConfig],
        room: RoomConfig,
        fs: int,
        event_id: str,
        timestamp_s: float,
        prior_xy: tuple[float, float] | None = None,
        prior_sigma_m: float = 1.4,
    ) -> LocalizationResult | None:
        pids = [pid for pid in event.frames[0].keys() if pid in phones]
        if len(pids) < 2:
            return None

        c = speed_of_sound(room.temperature_c)
        grid = self._get_grid(pids, phones, room, c)
        ny, nx = len(grid.grid_y), len(grid.grid_x)
        srp_map = np.zeros((ny, nx), dtype=np.float64)

        pair_lags: list[np.ndarray] = []
        pair_corrs: list[np.ndarray] = []
        pair_taus: list[float] = []
        pair_weights: list[float] = []

        # Downweight loose/drifting phones (> 0.5 ms calibration residual) when >= 3 tight phones exist.
        tight_count = sum(
            1 for pid in grid.pids
            if phones[pid].residual_ms is None or phones[pid].residual_ms <= 0.5
        )
        phone_trust: list[float] = []
        for pid in grid.pids:
            res_ms = phones[pid].residual_ms
            if tight_count >= 3 and res_ms is not None and res_ms > 0.5:
                phone_trust.append(1.0 / (1.0 + ((res_ms - 0.5) / 0.5) ** 2))
            else:
                phone_trust.append(1.0)

        # Stage 1: Frame-averaged GCC-PHAT-beta -> vectorized 2D SRP-PHAT map
        interp_f = 2 if len(event.frames[0][pids[0]]) >= 8000 else 4
        for p_idx, (i, j) in enumerate(grid.pairs):
            pi, pj = grid.pids[i], grid.pids[j]
            max_tau = float(grid.pair_max_tau[p_idx])
            pair_trust = min(phone_trust[i], phone_trust[j])

            acc_corr: np.ndarray | None = None
            lags_ref: np.ndarray | None = None
            weights: list[float] = []

            for frame in event.frames:
                lags, corr, _, w = compute_gcc_phat(
                    frame[pi],
                    frame[pj],
                    fs=fs,
                    max_tau_s=max_tau,
                    f_min=self.f_min,
                    f_max=self.f_max,
                    phat_beta=self.phat_beta,
                    interp_factor=interp_f,
                )
                if len(lags) < 2:
                    continue
                peak = float(np.max(corr))
                norm_corr = corr / peak if peak > 1e-12 else corr
                if acc_corr is None:
                    acc_corr = norm_corr.copy()
                    lags_ref = lags
                elif len(norm_corr) == len(acc_corr):
                    acc_corr += norm_corr
                weights.append(w)

            if acc_corr is None or lags_ref is None:
                lags_ref = np.array([-max_tau, max_tau], dtype=np.float64)
                acc_corr = np.zeros(2, dtype=np.float64)
                w_avg = 0.05 * pair_trust
                tau_global = 0.0
            else:
                acc_corr /= float(len(event.frames))
                w_avg = (float(np.mean(weights)) if weights else 0.05) * pair_trust
                k_best = int(np.argmax(acc_corr))
                tau_global = float(lags_ref[k_best])

            pair_lags.append(lags_ref)
            pair_corrs.append(acc_corr)
            pair_taus.append(tau_global)
            pair_weights.append(w_avg)

            # Dilate correlation curve by half a grid-cell time step (~0.35 ms) ONLY for coarse
            # Stage-1 grid lookup so narrow PHAT impulses never fall between 0.15 m grid nodes.
            srp_map += w_avg * _srp_pair_contribution(acc_corr, lags_ref, grid.tau_grid[p_idx])

        srp_search = srp_map
        if prior_xy is not None and prior_sigma_m > 0.0:
            dist2 = (grid.grid_x[None, :] - prior_xy[0]) ** 2 + (grid.grid_y[:, None] - prior_xy[1]) ** 2
            srp_search = srp_map * (1.0 + 0.35 * np.exp(-dist2 / (2.0 * prior_sigma_m * prior_sigma_m)))

        iy0, ix0 = np.unravel_index(int(np.argmax(srp_search)), srp_search.shape)
        x0 = float(grid.grid_x[ix0])
        y0 = float(grid.grid_y[iy0])

        # Build 32x24 normalized heatmap for dashboard visualization
        hm_ny, hm_nx = 24, 32
        gy = np.linspace(0.0, ny - 1.0, hm_ny)
        gx = np.linspace(0.0, nx - 1.0, hm_nx)
        my, mx = np.meshgrid(gy, gx, indexing="ij")
        hm_resampled = map_coordinates(srp_map, [my, mx], order=1, mode="nearest")
        hm_min = float(np.min(hm_resampled))
        hm_max = float(np.max(hm_resampled))
        if hm_max - hm_min > 1e-12:
            hm_norm = (hm_resampled - hm_min) / (hm_max - hm_min)
        else:
            hm_norm = np.zeros_like(hm_resampled)
        heatmap = Heatmap2D(
            nx=hm_nx,
            ny=hm_ny,
            values=[round(float(v), 4) for v in hm_norm.ravel()],
        )

        # Stage 2: Local SRP-guided peak validation + triangle cycle consistency + Huber LSQ
        refined_taus = np.array(pair_taus, dtype=np.float64)
        eff_weights = np.array(pair_weights, dtype=np.float64)
        inliers = [True] * len(grid.pairs)

        for p_idx in range(len(grid.pairs)):
            tau_srp = float(grid.tau_grid[p_idx, iy0, ix0])
            if abs(refined_taus[p_idx] - tau_srp) > 0.0025:
                lags = pair_lags[p_idx]
                corr = pair_corrs[p_idx]
                win_mask = np.abs(lags - tau_srp) <= 0.0025
                g_peak = float(np.max(corr))
                if np.any(win_mask):
                    sub_lags = lags[win_mask]
                    sub_corr = corr[win_mask]
                    k_loc = int(np.argmax(sub_corr))
                    if g_peak > 1e-12 and float(sub_corr[k_loc]) >= 0.35 * g_peak:
                        refined_taus[p_idx] = float(sub_lags[k_loc])
                        eff_weights[p_idx] *= 0.5
                    else:
                        inliers[p_idx] = False
                        eff_weights[p_idx] *= 0.05
                else:
                    inliers[p_idx] = False
                    eff_weights[p_idx] *= 0.05

        # Cycle consistency check when M >= 4
        m = len(grid.pids)
        if m >= 4:
            pair_lookup: dict[tuple[int, int], float] = {}
            for p_idx, (i, j) in enumerate(grid.pairs):
                pair_lookup[(i, j)] = float(refined_taus[p_idx])
                pair_lookup[(j, i)] = -float(refined_taus[p_idx])

            for p_idx, (i, j) in enumerate(grid.pairs):
                loop_errs: list[float] = []
                for k in range(m):
                    if k == i or k == j:
                        continue
                    loop_errs.append(abs(pair_lookup[(i, j)] + pair_lookup[(j, k)] - pair_lookup[(i, k)]))
                if loop_errs and float(np.median(loop_errs)) > self.cycle_threshold_s:
                    inliers[p_idx] = False
                    eff_weights[p_idx] *= 0.1

        pair_i = np.array([i for i, _ in grid.pairs], dtype=np.intp)
        pair_j = np.array([j for _, j in grid.pairs], dtype=np.intp)
        sqrt_w = np.sqrt(np.maximum(eff_weights, 1e-4))

        def residuals_fn(xy: np.ndarray) -> np.ndarray:
            d = np.sqrt((xy[0] - grid.phone_xy[:, 0]) ** 2 + (xy[1] - grid.phone_xy[:, 1]) ** 2)
            pred_tau = (d[pair_i] - d[pair_j]) / c
            return sqrt_w * (pred_tau - refined_taus)

        if m >= 3:
            lsq = least_squares(
                residuals_fn,
                x0=np.array([x0, y0], dtype=np.float64),
                bounds=([0.0, 0.0], [room.width_m, room.height_m]),
                loss="soft_l1",
                f_scale=0.0008,
                max_nfev=40,
            )
            x_est, y_est = float(lsq.x[0]), float(lsq.x[1])
            d_est = np.sqrt((x_est - grid.phone_xy[:, 0]) ** 2 + (y_est - grid.phone_xy[:, 1]) ** 2)
            raw_res = (d_est[pair_i] - d_est[pair_j]) / c - refined_taus
            residual_s = float(np.sqrt(np.sum(eff_weights * (raw_res ** 2)) / (np.sum(eff_weights) + 1e-12)))

            # If M >= 4 and either residual is high (> 0.8 ms) OR Stage 2 had to reject >=2 pairs,
            # test leave-one-phone-out consensus to isolate a single occluded/echo-corrupted phone.
            if m >= 4 and (residual_s > 0.0008 or sum(inliers) <= len(grid.pairs) // 2 + 1):
                best_drop: tuple[float, float, float, int, object, np.ndarray] | None = None
                full_global_res = float(
                    np.sqrt(np.mean((((d_est[pair_i] - d_est[pair_j]) / c) - np.array(pair_taus)) ** 2))
                )
                for bad_p in range(m):
                    keep_pairs = (pair_i != bad_p) & (pair_j != bad_p)
                    if int(np.sum(keep_pairs)) < 3:
                        continue
                    sub_srp = np.zeros((ny, nx), dtype=np.float64)
                    for p_idx in np.flatnonzero(keep_pairs):
                        sub_srp += pair_weights[p_idx] * _srp_pair_contribution(
                            pair_corrs[p_idx], pair_lags[p_idx], grid.tau_grid[p_idx]
                        )
                    siy, six = np.unravel_index(int(np.argmax(sub_srp)), sub_srp.shape)
                    sx0, sy0 = float(grid.grid_x[six]), float(grid.grid_y[siy])
                    keep_w = np.sqrt(np.maximum(np.array(pair_weights)[keep_pairs], 1e-4))
                    keep_t = np.array(pair_taus)[keep_pairs]
                    ki, kj = pair_i[keep_pairs], pair_j[keep_pairs]

                    def sub_res_fn(xy: np.ndarray) -> np.ndarray:
                        d = np.sqrt((xy[0] - grid.phone_xy[:, 0]) ** 2 + (xy[1] - grid.phone_xy[:, 1]) ** 2)
                        return keep_w * ((d[ki] - d[kj]) / c - keep_t)

                    sub_lsq = least_squares(
                        sub_res_fn,
                        x0=np.array([sx0, sy0], dtype=np.float64),
                        bounds=([0.0, 0.0], [room.width_m, room.height_m]),
                        loss="soft_l1",
                        f_scale=0.0008,
                        max_nfev=30,
                    )
                    sd = np.sqrt((sub_lsq.x[0] - grid.phone_xy[:, 0]) ** 2 + (sub_lsq.x[1] - grid.phone_xy[:, 1]) ** 2)
                    s_raw = (sd[ki] - sd[kj]) / c - keep_t
                    s_res = float(np.sqrt(np.mean(s_raw ** 2)))
                    if best_drop is None or s_res < best_drop[0]:
                        best_drop = (s_res, float(sub_lsq.x[0]), float(sub_lsq.x[1]), bad_p, sub_lsq, sub_srp)

                if best_drop is not None and best_drop[0] < self.loo_residual_threshold_s and best_drop[0] < 0.35 * full_global_res:
                    residual_s, x_est, y_est, bad_p, lsq, srp_map = best_drop
                    inliers = [not (i == bad_p or j == bad_p) for i, j in grid.pairs]
                    for p_idx, is_in in enumerate(inliers):
                        eff_weights[p_idx] = pair_weights[p_idx] if is_in else pair_weights[p_idx] * 0.05
                    refined_taus = np.array(pair_taus, dtype=np.float64)
                    # Refresh heatmap from clean subset
                    hm_resampled = map_coordinates(srp_map, [my, mx], order=1, mode="nearest")
                    hm_min, hm_max = float(np.min(hm_resampled)), float(np.max(hm_resampled))
                    hm_norm = (hm_resampled - hm_min) / (hm_max - hm_min) if hm_max - hm_min > 1e-12 else np.zeros_like(hm_resampled)
                    heatmap = Heatmap2D(nx=hm_nx, ny=hm_ny, values=[round(float(v), 4) for v in hm_norm.ravel()])

            # J has units s/m, so sqrt(trace((J^T J)^-1)) * residual_s yields positional uncertainty in meters
            jtj = lsq.jac.T @ lsq.jac + 1e-4 * np.eye(2)
            gdop = float(np.sqrt(np.trace(np.linalg.inv(jtj))))
            uncertainty_m = float(np.clip(gdop * max(residual_s, 0.0002), 0.05, 5.0))
        else:
            # 2-phone diagnostic mode: 1 pair cannot disambiguate hyperbola, report SRP grid peak
            x_est, y_est = x0, y0
            residual_s = 0.0
            uncertainty_m = float(max(room.width_m, room.height_m) * 0.25)

        inlier_frac = float(sum(inliers)) / float(len(inliers))
        srp_sharpness = float(np.clip((float(np.max(srp_map)) - float(np.mean(srp_map))) / (float(np.std(srp_map)) + 1e-9) / 6.0, 0.0, 1.0))
        res_score = float(math.exp(-((residual_s * 1000.0) / 1.5)))
        confidence = float(np.clip(0.4 * srp_sharpness + 0.3 * inlier_frac + 0.3 * res_score, 0.05, 0.99))

        pair_tdoas = [
            PairTdoa(
                i=grid.pids[i],
                j=grid.pids[j],
                tdoa_ms=round(float(refined_taus[p_idx] * 1000.0), 4),
                weight=round(float(eff_weights[p_idx]), 3),
                inlier=bool(inliers[p_idx]),
            )
            for p_idx, (i, j) in enumerate(grid.pairs)
        ]

        return LocalizationResult(
            event_id=event_id,
            timestamp_s=timestamp_s,
            x_m=round(x_est, 4),
            y_m=round(y_est, 4),
            uncertainty_m=round(uncertainty_m, 3),
            residual_ms=round(residual_s * 1000.0, 4),
            confidence=round(confidence, 3),
            phones_used=list(grid.pids),
            pair_tdoas_ms=pair_tdoas,
            heatmap=heatmap,
        )
