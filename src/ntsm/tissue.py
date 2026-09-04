"""Preparation of fMRIPrep GM/WM/CSF probability maps."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from numba import njit

from .io import load_nifti_float32, save_nifti_float32, voxel_sizes_mm


def gaussian_kernel_1d(sigma_voxels: float) -> np.ndarray:
    """Return the finite Gaussian kernel used for TPM smoothing."""

    if sigma_voxels <= 0:
        return np.ones(1, dtype=np.float32)
    radius = max(1, int(np.ceil(3.0 * sigma_voxels)))
    coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coordinate**2) / (2.0 * sigma_voxels**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


@njit(fastmath={"contract"})
def _convolve_axis_float32(source: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    nx, ny, nz = source.shape
    radius = kernel.size // 2
    output = np.zeros_like(source)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                total = np.float32(0.0)
                # Keep this order: float32 rounding can change the top-k boundary.
                for offset in range(radius, -radius - 1, -1):
                    xx, yy, zz = x, y, z
                    if axis == 0:
                        xx += offset
                    elif axis == 1:
                        yy += offset
                    else:
                        zz += offset
                    if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                        total = np.float32(
                            total
                            + np.float32(source[xx, yy, zz] * kernel[offset + radius])
                        )
                output[x, y, z] = total
    return output


def smooth_tissue_probability(
    probability: np.ndarray,
    fwhm_mm: float,
    voxel_size_mm: tuple[float, float, float],
) -> np.ndarray:
    """Smooth one TPM with a separable Gaussian and zero padding."""

    output = np.nan_to_num(np.asarray(probability, dtype=np.float32), copy=True)
    if fwhm_mm <= 0:
        return np.clip(output, 0.0, 1.0)
    sigma_mm = float(fwhm_mm) / 2.3548
    for axis, size_mm in enumerate(voxel_size_mm):
        kernel = gaussian_kernel_1d(sigma_mm / float(size_mm))
        output = _convolve_axis_float32(output, kernel, axis)
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)


def prepare_tissue_file(
    source: str | Path,
    destination: str | Path,
    *,
    fwhm_mm: float,
    overwrite: bool = False,
) -> tuple[nib.spatialimages.SpatialImage, np.ndarray, Path]:
    """Load, smooth, cache, and return one fMRIPrep tissue probability map."""

    target = Path(destination)
    if target.is_file() and not overwrite:
        image, data = load_nifti_float32(target)
        return image, data, target
    image, data = load_nifti_float32(source)
    if data.ndim != 3:
        raise ValueError(f"tissue probability image is not 3-D: {source}")
    smoothed = smooth_tissue_probability(data, fwhm_mm, voxel_sizes_mm(image))
    save_nifti_float32(smoothed, image, target)
    cached_image, cached_data = load_nifti_float32(target)
    return cached_image, cached_data, target


def normalize_tissue_probabilities(
    gm: np.ndarray, wm: np.ndarray, csf: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optionally normalize tissue probabilities so they sum to one per voxel."""

    total = gm.astype(np.float64) + wm.astype(np.float64) + csf.astype(np.float64)
    valid = total > np.finfo(np.float32).eps
    outputs = []
    for source in (gm, wm, csf):
        output = np.zeros_like(source, dtype=np.float32)
        output[valid] = (source[valid] / total[valid]).astype(np.float32)
        outputs.append(output)
    return outputs[0], outputs[1], outputs[2]
