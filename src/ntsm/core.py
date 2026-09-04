"""Static nTSM graph diffusion and whole-run spectral concentration."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass

import numba
import numpy as np
from numba import njit, prange
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .config import MappingConfig

_OFFSETS = np.asarray(
    [
        (dx, dy, dz)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ],
    dtype=np.int8,
)

_AUTO_WORKER_CAP = 32


@dataclass(frozen=True)
class MappingSummary:
    centers: int
    finite_centers: int
    workers: int
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


@njit(inline="always")
def _linear_index(x: int, y: int, z: int, nx: int, ny: int) -> int:
    # NIfTI x is the fastest-changing index in the flattened volume.
    return x + nx * (y + ny * z)


@njit(inline="always")
def _center_is_gm(tpm: np.ndarray, center: int) -> bool:
    gm = tpm[center, 0]
    wm = tpm[center, 1]
    csf = tpm[center, 2]
    return gm >= wm and gm >= csf


@njit
def _compute_one_voxel(
    center: int,
    nx: int,
    ny: int,
    nz: int,
    signal: np.ndarray,
    signal_row: np.ndarray,
    tpm: np.ndarray,
    mask: np.ndarray,
    offsets: np.ndarray,
    n_step: int,
    k_keep: int,
    radius: int,
    distance_power: float,
    stop_on_convergence: bool,
    tolerance: float,
    restrict_gm: bool,
    midline_one_based: float,
    minimum_local: int,
    failed_value: float,
) -> np.float32:
    plane = nx * ny
    cz = center // plane
    remainder = center - cz * plane
    cy = remainder // nx
    cx = remainder - cy * nx

    x0 = max(0, cx - radius)
    x1 = min(nx - 1, cx + radius)
    y0 = max(0, cy - radius)
    y1 = min(ny - 1, cy + radius)
    z0 = max(0, cz - radius)
    z1 = min(nz - 1, cz + radius)
    wx = x1 - x0 + 1
    wy = y1 - y0 + 1
    wz = z1 - z0 + 1
    capacity = wx * wy * wz

    local_lookup = np.full(capacity, -1, dtype=np.int32)
    local_global = np.empty(capacity, dtype=np.int64)
    local_x = np.empty(capacity, dtype=np.int16)
    local_y = np.empty(capacity, dtype=np.int16)
    local_z = np.empty(capacity, dtype=np.int16)
    center_local = -1
    local_count = 0
    gm_center = restrict_gm and _center_is_gm(tpm, center)
    center_left = float(cx + 1) <= midline_one_based

    for z in range(z0, z1 + 1):
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                global_index = _linear_index(x, y, z, nx, ny)
                if not mask[global_index]:
                    continue
                if gm_center:
                    voxel_left = float(x + 1) <= midline_one_based
                    if voxel_left != center_left:
                        continue
                slot = (x - x0) + wx * ((y - y0) + wy * (z - z0))
                local_lookup[slot] = local_count
                local_global[local_count] = global_index
                local_x[local_count] = x
                local_y[local_count] = y
                local_z[local_count] = z
                if global_index == center:
                    center_local = local_count
                local_count += 1

    if center_local < 0 or local_count < minimum_local:
        return np.float32(failed_value)

    tc0 = np.float64(tpm[center, 0])
    tc1 = np.float64(tpm[center, 1])
    tc2 = np.float64(tpm[center, 2])
    center_similarity = np.empty(local_count, dtype=np.float32)
    for i in range(local_count):
        global_index = local_global[i]
        ti0 = np.float64(tpm[global_index, 0])
        ti1 = np.float64(tpm[global_index, 1])
        ti2 = np.float64(tpm[global_index, 2])
        similarity = ti0 * tc0 + ti1 * tc1 + ti2 * tc2
        if not math.isfinite(similarity) or similarity < 0:
            similarity = 0.0
        center_similarity[i] = np.float32(similarity)

    neighbors = np.full((local_count, 26), -1, dtype=np.int32)
    transition = np.zeros((local_count, 26), dtype=np.float64)
    degree = np.zeros(local_count, dtype=np.float64)
    for i in range(local_count):
        x = int(local_x[i])
        y = int(local_y[i])
        z = int(local_z[i])
        gi = local_global[i]
        for edge in range(26):
            dx = int(offsets[edge, 0])
            dy = int(offsets[edge, 1])
            dz = int(offsets[edge, 2])
            xn = x + dx
            yn = y + dy
            zn = z + dz
            if xn < x0 or xn > x1 or yn < y0 or yn > y1 or zn < z0 or zn > z1:
                continue
            slot = (xn - x0) + wx * ((yn - y0) + wy * (zn - z0))
            j = local_lookup[slot]
            if j < 0:
                continue
            gj = local_global[j]
            sim = np.float32(0.0)
            sim = np.float32(sim + np.float32(tpm[gi, 0] * tpm[gj, 0]))
            sim = np.float32(sim + np.float32(tpm[gi, 1] * tpm[gj, 1]))
            sim = np.float32(sim + np.float32(tpm[gi, 2] * tpm[gj, 2]))
            if sim < 0:
                sim = np.float32(0.0)
            center_factor = np.float32(
                math.sqrt(
                    max(
                        float(np.float32(center_similarity[i] * center_similarity[j])),
                        0.0,
                    )
                )
            )
            tissue_term = np.float32(sim * center_factor)
            distance = math.sqrt(float(dx * dx + dy * dy + dz * dz))
            weight = float(tissue_term) / (distance**distance_power)
            neighbors[i, edge] = j
            transition[i, edge] = weight
            degree[i] += weight

    for i in range(local_count):
        if degree[i] > 0:
            for edge in range(26):
                if neighbors[i, edge] >= 0:
                    transition[i, edge] /= degree[i]

    q = np.zeros(local_count, dtype=np.float64)
    q[center_local] = 1.0
    for _ in range(n_step):
        q_next = np.zeros(local_count, dtype=np.float64)
        for i in range(local_count):
            if degree[i] <= 0:
                q_next[i] += q[i]
                continue
            qi = q[i]
            if qi == 0:
                continue
            for edge in range(26):
                j = neighbors[i, edge]
                if j >= 0:
                    q_next[j] += qi * transition[i, edge]
        difference = 0.0
        for i in range(local_count):
            difference += abs(q_next[i] - q[i])
        q = q_next
        if stop_on_convergence and difference < tolerance:
            break

    q_single = np.zeros(local_count, dtype=np.float32)
    q_total = np.float32(0.0)
    positive_count = 0
    for i in range(local_count):
        value = np.float32(q[i])
        if math.isfinite(float(value)) and value > 0:
            q_single[i] = value
            q_total = np.float32(q_total + value)
            positive_count += 1
    if not q_total > 0:
        return np.float32(0.0)
    for i in range(local_count):
        q_single[i] = np.float32(q_single[i] / q_total)

    selected_count = min(positive_count, k_keep)
    selected = np.empty(selected_count, dtype=np.int32)
    if positive_count <= k_keep:
        position = 0
        for i in range(local_count):
            if q_single[i] > 0:
                selected[position] = i
                position += 1
    else:
        # Stable sorting makes equal-weight support selection reproducible.
        order = np.argsort(-q_single, kind="mergesort")
        for i in range(selected_count):
            selected[i] = order[i]
        has_center = False
        for i in range(selected_count):
            if selected[i] == center_local:
                has_center = True
                break
        if not has_center:
            selected[selected_count - 1] = center_local

    kept_total = np.float32(0.0)
    for i in range(selected_count):
        kept_total = np.float32(kept_total + q_single[selected[i]])
    if not kept_total > 0:
        return np.float32(0.0)

    timepoints = signal.shape[1]
    x_weighted = np.empty((selected_count, timepoints), dtype=np.float32)
    for row in range(selected_count):
        local_index = selected[row]
        global_index = local_global[local_index]
        signal_index = signal_row[global_index]
        if signal_index < 0:
            return np.float32(failed_value)
        weight = np.float32(q_single[local_index] / kept_total)
        square_root = np.float32(math.sqrt(float(weight)))
        for timepoint in range(timepoints):
            value = np.float32(signal[signal_index, timepoint] * square_root)
            x_weighted[row, timepoint] = value if math.isfinite(float(value)) else np.float32(0.0)

    x_double = x_weighted.astype(np.float64)
    covariance = x_double @ x_double.T
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    total_energy = 0.0
    for i in range(eigenvalues.size):
        if eigenvalues[i] < 0:
            eigenvalues[i] = 0.0
        total_energy += eigenvalues[i]
    if not total_energy > 0:
        return np.float32(0.0)

    entropy = 0.0
    rank = 0
    for i in range(eigenvalues.size):
        proportion = eigenvalues[i] / total_energy
        if proportion > 1e-12 and math.isfinite(proportion):
            entropy -= proportion * math.log(proportion)
            rank += 1
    if rank <= 1:
        return np.float32(1.0)
    concentration = 1.0 - entropy / math.log(float(rank))
    return np.float32(min(1.0, max(0.0, concentration)))


@njit(parallel=True)
def _compute_center_chunk(
    centers: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    signal: np.ndarray,
    signal_row: np.ndarray,
    tpm: np.ndarray,
    mask: np.ndarray,
    offsets: np.ndarray,
    n_step: int,
    k_keep: int,
    radius: int,
    distance_power: float,
    stop_on_convergence: bool,
    tolerance: float,
    restrict_gm: bool,
    midline_one_based: float,
    minimum_local: int,
    failed_value: float,
) -> np.ndarray:
    output = np.empty(centers.size, dtype=np.float32)
    for index in prange(centers.size):
        output[index] = _compute_one_voxel(
            int(centers[index]),
            nx,
            ny,
            nz,
            signal,
            signal_row,
            tpm,
            mask,
            offsets,
            n_step,
            k_keep,
            radius,
            distance_power,
            stop_on_convergence,
            tolerance,
            restrict_gm,
            midline_one_based,
            minimum_local,
            failed_value,
        )
    return output


def compute_ntsm_map(
    signal: np.ndarray,
    signal_row: np.ndarray,
    tissue_probabilities: np.ndarray,
    analysis_mask: np.ndarray,
    config: MappingConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, MappingSummary]:
    """Compute a static nTSM map using every time point in ``signal``."""

    mask_3d = np.asarray(analysis_mask, dtype=np.bool_)
    if mask_3d.ndim != 3:
        raise ValueError("analysis_mask must be 3-D")
    shape = tuple(int(value) for value in mask_3d.shape)
    voxel_count = int(np.prod(shape))
    mask = np.ascontiguousarray(mask_3d.reshape(-1, order="F"))
    tpm = np.ascontiguousarray(tissue_probabilities, dtype=np.float32)
    x = np.ascontiguousarray(signal, dtype=np.float32)
    row_index = np.ascontiguousarray(signal_row, dtype=np.int64)
    if tpm.shape != (voxel_count, 3):
        raise ValueError(f"tissue_probabilities must have shape ({voxel_count}, 3)")
    if row_index.shape != (voxel_count,):
        raise ValueError(f"signal_row must have shape ({voxel_count},)")
    if x.ndim != 2 or x.shape[0] <= 0 or x.shape[1] <= 1:
        raise ValueError("signal must be a nonempty voxel-by-time matrix")
    if np.any(row_index[mask] < 0):
        raise ValueError("every analysis-mask voxel must have a signal row")

    centers = np.flatnonzero(mask).astype(np.int64)
    map_flat = np.full(voxel_count, np.float32(config.failed_voxel_value), dtype=np.float32)

    available_workers = min(os.cpu_count() or 1, numba.config.NUMBA_NUM_THREADS)
    requested_workers = config.workers if config.workers > 0 else _AUTO_WORKER_CAP
    worker_count = max(1, min(requested_workers, available_workers))
    numba.set_num_threads(worker_count)
    if progress:
        progress(f"Using {worker_count} parallel mapping worker(s)")
    midline = config.midline_voxel_one_based
    if midline is None:
        midline = (shape[0] + 1) / 2.0

    progress_bar = tqdm(
        total=int(centers.size),
        desc="nTSM",
        unit="voxel",
        dynamic_ncols=True,
        disable=progress is None,
    )
    for first in range(0, centers.size, config.chunk_size):
        last = min(centers.size, first + config.chunk_size)
        with threadpool_limits(limits=1):
            values = _compute_center_chunk(
                centers[first:last],
                shape[0],
                shape[1],
                shape[2],
                x,
                row_index,
                tpm,
                mask,
                _OFFSETS,
                config.n_step,
                config.k,
                config.spatial_radius,
                config.distance_decay_power,
                config.random_walk_stop_mode == "converge_or_nstep",
                config.random_walk_tolerance,
                config.restrict_gm_to_hemisphere,
                float(midline),
                config.minimum_local_voxels,
                config.failed_voxel_value,
            )
        map_flat[centers[first:last]] = values
        progress_bar.update(last - first)
    progress_bar.close()
    finite = map_flat[mask & np.isfinite(map_flat)].astype(np.float64)
    if finite.size == 0:
        raise RuntimeError("nTSM produced no finite analysis-mask values")
    summary = MappingSummary(
        centers=int(centers.size),
        finite_centers=int(finite.size),
        workers=worker_count,
        mean=float(finite.mean()),
        standard_deviation=float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        minimum=float(finite.min()),
        maximum=float(finite.max()),
    )
    return map_flat.reshape(shape, order="F"), summary


def compress_signal_to_mask(bold: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Store only analysis-mask time series while retaining volume indices."""

    data = np.asarray(bold, dtype=np.float32)
    mask_3d = np.asarray(mask, dtype=np.bool_)
    if data.ndim != 4 or data.shape[:3] != mask_3d.shape:
        raise ValueError("BOLD must be 4-D and match the analysis mask")
    flat_mask = mask_3d.reshape(-1, order="F")
    volume_rows = np.flatnonzero(flat_mask)
    full_signal = data.reshape((-1, data.shape[3]), order="F")
    signal = np.ascontiguousarray(full_signal[volume_rows, :], dtype=np.float32)
    signal[~np.isfinite(signal)] = 0
    row_index = np.full(flat_mask.size, -1, dtype=np.int64)
    row_index[volume_rows] = np.arange(volume_rows.size, dtype=np.int64)
    return signal, row_index


def stack_tissue_probabilities(gm: np.ndarray, wm: np.ndarray, csf: np.ndarray) -> np.ndarray:
    """Return voxel-by-tissue values using NIfTI x-fastest indexing."""

    if gm.shape != wm.shape or gm.shape != csf.shape or gm.ndim != 3:
        raise ValueError("GM, WM, and CSF maps must be 3-D and share a shape")
    return np.ascontiguousarray(
        np.column_stack(
            (
                np.asarray(gm, dtype=np.float32).reshape(-1, order="F"),
                np.asarray(wm, dtype=np.float32).reshape(-1, order="F"),
                np.asarray(csf, dtype=np.float32).reshape(-1, order="F"),
            )
        ),
        dtype=np.float32,
    )
