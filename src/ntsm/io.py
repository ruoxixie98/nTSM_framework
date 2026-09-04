"""NIfTI and checksum helpers used throughout nTSM."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np


def load_nifti_float32(path: str | Path) -> tuple[nib.spatialimages.SpatialImage, np.ndarray]:
    """Load a NIfTI image and apply header scaling into a float32 array."""

    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
    return image, data


def assert_same_grid(
    named_images: Iterable[tuple[str, nib.spatialimages.SpatialImage]],
    *,
    affine_atol: float = 1e-5,
) -> None:
    """Require identical 3-D shapes and numerically matching affines."""

    items = list(named_images)
    if not items:
        raise ValueError("at least one image is required for grid validation")
    reference_name, reference = items[0]
    reference_shape = tuple(reference.shape[:3])
    reference_axes = nib.aff2axcodes(reference.affine)
    for name, image in items[1:]:
        if tuple(image.shape[:3]) != reference_shape:
            raise ValueError(
                f"grid shape mismatch: {reference_name}={reference_shape}, "
                f"{name}={tuple(image.shape[:3])}"
            )
        axes = nib.aff2axcodes(image.affine)
        if axes != reference_axes:
            raise ValueError(
                f"orientation mismatch: {reference_name}={reference_axes}, {name}={axes}"
            )
        if not np.allclose(reference.affine, image.affine, rtol=0.0, atol=affine_atol):
            raise ValueError(
                f"affine mismatch between {reference_name} and {name}; "
                "inputs are not reoriented or resampled automatically"
            )


def assert_anatomical_xyz_axes(image: nib.spatialimages.SpatialImage) -> None:
    """Reject permuted spatial axes before applying the x-hemisphere rule."""

    axes = nib.aff2axcodes(image.affine)
    valid = axes[0] in {"L", "R"} and axes[1] in {"A", "P"} and axes[2] in {"I", "S"}
    if not valid:
        raise ValueError(
            f"permuted spatial axes are not supported: {axes}; expected anatomical x/y/z axes"
        )


def voxel_sizes_mm(image: nib.spatialimages.SpatialImage) -> tuple[float, float, float]:
    """Return positive spatial zooms in millimetres."""

    zooms = tuple(float(abs(value)) for value in image.header.get_zooms()[:3])
    if len(zooms) != 3 or any(not np.isfinite(value) or value <= 0 for value in zooms):
        raise ValueError(f"invalid NIfTI voxel sizes: {zooms}")
    return zooms


def save_nifti_float32(
    data: np.ndarray,
    reference: nib.spatialimages.SpatialImage,
    path: str | Path,
) -> Path:
    """Atomically save float32 data on the reference image grid."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    output = nib.Nifti1Image(np.asarray(data, dtype=np.float32), reference.affine, header)
    qform, qform_code = reference.get_qform(coded=True)
    sform, sform_code = reference.get_sform(coded=True)
    if qform is not None:
        output.set_qform(qform, int(qform_code))
    if sform is not None:
        output.set_sform(sform, int(sform_code))
    suffix = ".nii.gz" if destination.name.endswith(".nii.gz") else destination.suffix
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=suffix, dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        nib.save(output, str(temporary))
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
