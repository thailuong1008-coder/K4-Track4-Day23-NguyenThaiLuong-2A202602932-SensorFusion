"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from scipy.stats import chi2
from fusion_lab.workspace_support import get_tracking_params
from fusion_lab.workspace_loader import load_workspace_module

kalman = load_workspace_module("kalman")


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    H = meas.sensor.get_H(track.x)
    gamma = kalman.innovation(track.x, meas)
    S = kalman.innovation_covariance(track.P, meas, H)
    dist_sq = gamma.T @ np.linalg.inv(S) @ gamma
    return float(dist_sq[0, 0])


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    params = get_tracking_params()
    threshold = chi2.ppf(params.gating_threshold, sensor.dim_meas)
    return mhd_sq < threshold


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    cost_matrix = np.full((len(track_list), len(meas_list)), np.inf)
    for i, track in enumerate(track_list):
        for j, meas in enumerate(meas_list):
            if meas.sensor.in_fov(track.x):
                try:
                    mhd_sq = mahalanobis_distance(track, meas)
                    if chi2_gate(mhd_sq, meas.sensor):
                        cost_matrix[i, j] = mhd_sq
                except ValueError:
                    pass
    return np.matrix(cost_matrix)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    mat = np.asarray(association_matrix)
    if mat.size == 0 or np.all(np.isinf(mat)):
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)
    
    min_idx = np.unravel_index(np.argmin(mat), mat.shape)
    if not np.isfinite(mat[min_idx]):
        return np.nan, np.nan, association_matrix, list(unassigned_tracks), list(unassigned_meas)
        
    i, j = min_idx
    track = unassigned_tracks[i]
    meas = unassigned_meas[j]
    
    new_tracks = list(unassigned_tracks)
    new_tracks.pop(i)
    
    new_meas = list(unassigned_meas)
    new_meas.pop(j)
    
    new_mat = np.delete(mat, i, axis=0)
    new_mat = np.delete(new_mat, j, axis=1)
    
    return track, meas, np.matrix(new_mat), new_tracks, new_meas


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)
    
    if unassigned_tracks and unassigned_meas:
        cost_matrix = association_cost_matrix(unassigned_tracks, unassigned_meas)
        while True:
            track, meas, cost_matrix, unassigned_tracks, unassigned_meas = pick_next_pair(
                cost_matrix, unassigned_tracks, unassigned_meas
            )
            if track is np.nan:
                break
            
            filter_obj.update(track, meas)
            manager.handle_updated_track(track, sensor)
            
    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
