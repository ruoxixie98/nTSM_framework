"""Downstream preprocessing for whole-run fMRIPrep BOLD data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
from scipy.signal import butter, filtfilt

from .config import PreprocessingConfig

_MOTION_BASE = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
_MOTION_SUFFIX = ("", "_derivative1", "_power2", "_derivative1_power2")
FRISTON_24_COLUMNS = tuple(
    f"{base}{suffix}" for base in _MOTION_BASE for suffix in _MOTION_SUFFIX
)


ConfigInput = PreprocessingConfig | None
_CONSTANT_TOLERANCE = 1e-12


def preprocess_fmriprep_bold(
    bold: np.ndarray,
    brain_mask: np.ndarray,
    confounds: str | Path,
    *,
    tr: float,
    voxel_size_mm: float | Sequence[float] = 2.0,
    config: ConfigInput = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clean one complete fMRIPrep BOLD run."""

    cfg = _resolve_config(config)
    _validate_preprocessing_config(cfg)
    tr_value = _validate_tr(tr)

    bold_array = _validate_bold(bold)
    mask = _validate_mask(brain_mask, bold_array.shape[:3])
    raw_timepoints = int(bold_array.shape[3])
    drop = cfg.drop_initial_volumes
    if drop >= raw_timepoints:
        raise ValueError(
            f"drop_initial_volumes={drop} must be smaller than BOLD "
            f"volume count={raw_timepoints}."
        )
    final_timepoints = raw_timepoints - drop
    if final_timepoints < 2:
        raise ValueError("At least two volumes must remain for linear detrending.")

    voxel_sizes = _validate_voxel_sizes(voxel_size_mm)
    sigma_vox = _fwhm_to_sigma_vox(cfg.spatial_smoothing_fwhm_mm, voxel_sizes)
    filter_info = _validate_filter_settings(cfg, tr_value, final_timepoints)

    nuisance, confound_qc = build_fmriprep_confound_matrix(
        confounds,
        raw_timepoints=raw_timepoints,
        drop_initial_volumes=drop,
        config=cfg,
    )

    work = np.asarray(bold_array[..., drop:], dtype=np.float32).copy()
    work[~mask, :] = 0.0
    _require_finite(work, "in-mask BOLD")
    work = smooth_masked_bold(
        work,
        mask,
        sigma_vox=sigma_vox,
        kernel_radius_sigma=cfg.smoothing_kernel_truncate_sigma,
    )
    mask_flat = mask.reshape(-1)
    timeseries = work.reshape(-1, final_timepoints)[mask_flat, :].T.astype(
        np.float64, copy=False
    )
    timeseries = _linear_detrend(timeseries)

    design = np.column_stack((np.ones(final_timepoints, dtype=np.float64), nuisance))
    beta, _, design_rank, singular_values = np.linalg.lstsq(
        design, timeseries, rcond=None
    )
    timeseries = timeseries - design @ beta
    _require_finite(timeseries, "regression residuals")

    timeseries = apply_temporal_filter(timeseries, tr=tr_value, config=cfg)
    _require_finite(timeseries, "temporally filtered data")

    timeseries, standardization_info = standardize_timeseries(
        timeseries, mode=cfg.standardization
    )

    output_dtype = np.dtype("float32")
    cleaned = np.zeros(
        (*bold_array.shape[:3], final_timepoints), dtype=output_dtype
    )
    cleaned.reshape(-1, final_timepoints)[mask_flat, :] = timeseries.T.astype(
        output_dtype, copy=False
    )

    if singular_values.size and singular_values[-1] > 0:
        condition_number: float | None = float(singular_values[0] / singular_values[-1])
    else:
        condition_number = None
    confound_qc["design_columns"] = ["intercept", *confound_qc["used_columns"]]
    confound_qc["design_rank"] = int(design_rank)
    confound_qc["design_condition_number"] = condition_number

    qc: dict[str, Any] = {
        "input_shape": [int(value) for value in bold_array.shape],
        "output_shape": [int(value) for value in cleaned.shape],
        "raw_volumes": raw_timepoints,
        "dropped_initial_volumes": drop,
        "final_volumes": final_timepoints,
        "brain_voxels": int(mask.sum()),
        "tr_seconds": tr_value,
        "nyquist_hz": 0.5 / tr_value,
        "smoothing": {
            "fwhm_mm": _json_number_or_list(cfg.spatial_smoothing_fwhm_mm),
            "voxel_size_mm": [float(value) for value in voxel_sizes],
        },
        "detrending": {"model": "intercept_plus_linear_time"},
        "confounds": confound_qc,
        "filtering": filter_info,
        "standardization": standardization_info,
        "output_dtype": output_dtype.name,
    }
    return cleaned, qc


def build_fmriprep_confound_matrix(
    confounds: str | Path,
    *,
    raw_timepoints: int,
    drop_initial_volumes: int,
    config: ConfigInput = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the nuisance-regression matrix."""

    cfg = _resolve_config(config)
    _validate_preprocessing_config(cfg)
    if raw_timepoints <= 0:
        raise ValueError("raw_timepoints must be positive.")
    if not 0 <= drop_initial_volumes < raw_timepoints:
        raise ValueError("drop_initial_volumes must be smaller than raw_timepoints")

    path = Path(confounds)
    if not path.is_file():
        raise FileNotFoundError(f"Confounds TSV not found: {path}")
    frame = pd.read_csv(path, sep="\t")
    if len(frame) != raw_timepoints:
        raise ValueError(
            f"Confound length mismatch: expected {raw_timepoints} raw rows, "
            f"found {len(frame)}."
        )
    if frame.columns.duplicated().any():
        duplicated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Confound column names must be unique; duplicates: {duplicated}")

    all_columns = [str(value) for value in frame.columns]
    frame.columns = all_columns
    trimmed = frame.iloc[drop_initial_volumes:].reset_index(drop=True)

    motion_outlier_columns = sorted(
        name for name in all_columns if name.startswith("motion_outlier")
    )
    requested = list(FRISTON_24_COLUMNS)
    missing_motion = [name for name in FRISTON_24_COLUMNS if name not in all_columns]
    if missing_motion:
        raise ValueError(
            "Missing required Friston-24 confounds: " + ", ".join(missing_motion)
        )

    a_available = sorted(name for name in all_columns if name.startswith("a_comp_cor_"))
    t_available = sorted(name for name in all_columns if name.startswith("t_comp_cor_"))
    a_selected = a_available[: cfg.confounds.acompcor_components]
    t_selected = t_available[: cfg.confounds.tcompcor_components]
    requested.extend(a_selected)
    requested.extend(t_selected)
    compcor_counts = {
        "a_comp_cor_requested": cfg.confounds.acompcor_components,
        "a_comp_cor_available": len(a_available),
        "a_comp_cor_selected": len(a_selected),
        "t_comp_cor_requested": cfg.confounds.tcompcor_components,
        "t_comp_cor_available": len(t_available),
        "t_comp_cor_selected": len(t_selected),
    }

    optional = []
    if cfg.confounds.csf:
        optional.append("csf")
    if cfg.confounds.white_matter:
        optional.append("white_matter")
    if cfg.confounds.global_signal:
        optional.append("global_signal")
    missing_optional = [name for name in optional if name not in all_columns]
    if missing_optional:
        raise ValueError(
            "Requested optional confounds are missing: " + ", ".join(missing_optional)
        )
    requested.extend(optional)
    if cfg.confounds.motion_outliers:
        requested.extend(motion_outlier_columns)
    requested = _unique_preserving_order(requested)
    selected = [name for name in requested if name in all_columns]

    if not selected:
        raise ValueError("No nuisance columns were selected.")
    numeric = _numeric_frame(trimmed.loc[:, selected], context="selected confounds")
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    if np.isinf(values).any():
        count = int(np.isinf(values).sum())
        raise ValueError(f"Selected confounds contain {count} infinite values.")

    imputed_nan_by_column: dict[str, int] = {}
    all_nan_columns: list[str] = []
    for index, name in enumerate(selected):
        vector = values[:, index]
        missing = np.isnan(vector)
        if not missing.any():
            continue
        finite_values = vector[~missing]
        if finite_values.size:
            fill_value = float(np.median(finite_values))
        else:
            fill_value = 0.0
            all_nan_columns.append(name)
        vector[missing] = fill_value
        imputed_nan_by_column[name] = int(missing.sum())

    if values.shape[0] < 2:
        column_sd = np.zeros(values.shape[1], dtype=np.float64)
    else:
        column_sd = np.std(values, axis=0, ddof=1)
    keep = np.isfinite(column_sd) & (column_sd >= _CONSTANT_TOLERANCE)
    used = [name for name, use in zip(selected, keep, strict=True) if use]
    dropped_constant = [name for name, use in zip(selected, keep, strict=True) if not use]
    matrix = values[:, keep]
    _require_finite(matrix, "prepared confound matrix")

    used_set = set(used)
    qc: dict[str, Any] = {
        "used_columns": used,
        "dropped_constant_columns": dropped_constant,
        "imputed_nan_by_column": imputed_nan_by_column,
        "imputed_nan_total": int(sum(imputed_nan_by_column.values())),
        "all_nan_columns_filled_with_zero": all_nan_columns,
        "motion_outlier_columns_total": len(motion_outlier_columns),
        "motion_outlier_columns_in_design": sum(
            name in used_set for name in motion_outlier_columns
        ),
        **compcor_counts,
    }
    return matrix, qc


def smooth_masked_bold(
    bold: np.ndarray,
    brain_mask: np.ndarray,
    *,
    sigma_vox: float | Sequence[float],
    kernel_radius_sigma: float = 2.0,
) -> np.ndarray:
    """Smooth each BOLD volume inside the brain mask."""

    array = np.asarray(bold)
    if array.ndim != 4:
        raise ValueError(f"bold must be 4-D; got shape {array.shape}.")
    mask = _validate_mask(brain_mask, array.shape[:3])
    sigma = _as_spatial_triplet(sigma_vox, name="sigma_vox", allow_zero=True)
    radius_sigma = _positive_finite_float(
        kernel_radius_sigma, "kernel_radius_sigma"
    )
    mask_float = mask.astype(np.float32, copy=False)
    output = np.empty(array.shape, dtype=np.float32)
    for time_index in range(array.shape[3]):
        volume = np.asarray(array[..., time_index], dtype=np.float32) * mask_float
        for axis, axis_sigma in enumerate(sigma):
            if axis_sigma <= 0:
                continue
            radius = int(np.ceil(radius_sigma * axis_sigma))
            coordinate = np.arange(-radius, radius + 1, dtype=np.float64)
            kernel = np.exp(-0.5 * (coordinate / axis_sigma) ** 2)
            kernel /= kernel.sum()
            volume = convolve1d(volume, kernel, axis=axis, mode="nearest")
        output[..., time_index] = volume * mask_float
    return output


def dct_highpass(
    data: np.ndarray, *, tr: float, cutoff_hz: float | None
) -> np.ndarray:
    """Apply a DCT high-pass projection."""

    values = _validate_2d_timeseries(data)
    tr_value = _validate_tr(tr)
    if cutoff_hz is None:
        return values.copy()
    cutoff = _positive_finite_float(cutoff_hz, "cutoff_hz")
    nyquist = 0.5 / tr_value
    if cutoff >= nyquist:
        raise ValueError(
            f"cutoff_hz={cutoff:g} must be below Nyquist={nyquist:g} Hz."
        )
    n_timepoints = values.shape[0]
    cutoff_period_seconds = 1.0 / cutoff
    n_dct = int(
        np.floor(2.0 * (n_timepoints * tr_value) / cutoff_period_seconds + 1.0)
    )
    n_dct = min(max(n_dct, 1), n_timepoints)
    if n_dct <= 1:
        return values.copy()
    sample = np.arange(n_timepoints, dtype=np.float64)[:, None]
    harmonic = np.arange(1, n_dct, dtype=np.float64)[None, :]
    regressors = np.sqrt(2.0 / n_timepoints) * np.cos(
        np.pi * (2.0 * sample + 1.0) * harmonic / (2.0 * n_timepoints)
    )
    filtered = values - regressors @ (regressors.T @ values)
    return filtered


def apply_temporal_filter(
    data: np.ndarray, *, tr: float, config: ConfigInput = None
) -> np.ndarray:
    """Apply the configured high-pass backend and Butterworth low-pass."""

    cfg = _resolve_config(config)
    _validate_preprocessing_config(cfg)
    values = _validate_2d_timeseries(data)
    tr_value = _validate_tr(tr)
    _validate_filter_settings(cfg, tr_value, values.shape[0])
    backend = _normalise_filter_backend(cfg.filter_backend)
    filtered = values.copy()

    if cfg.bandpass_low_hz is not None:
        if backend == "dct":
            filtered = dct_highpass(
                filtered, tr=tr_value, cutoff_hz=cfg.bandpass_low_hz
            )
        else:
            filtered = _butterworth_filtfilt(
                filtered,
                cutoff_hz=float(cfg.bandpass_low_hz),
                tr=tr_value,
                order=cfg.filter_order_highpass,
                kind="highpass",
            )
    if cfg.bandpass_high_hz is not None:
        filtered = _butterworth_filtfilt(
            filtered,
            cutoff_hz=float(cfg.bandpass_high_hz),
            tr=tr_value,
            order=cfg.filter_order_lowpass,
            kind="lowpass",
        )
    return filtered


def standardize_timeseries(
    data: np.ndarray, *, mode: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """Standardize voxel time series."""

    values = _validate_2d_timeseries(data)
    normalized_mode = _normalise_standardization_mode(mode)
    output = values.copy()
    bad: np.ndarray
    denominator_label: str | None
    if normalized_mode == "diff_noise_std":
        if output.shape[0] < 3:
            raise ValueError(
                "Difference-noise standardization requires at least three timepoints."
            )
        output -= np.mean(output, axis=0)
        denominator = np.std(np.diff(output, axis=0), axis=0, ddof=1) / np.sqrt(2.0)
        bad = (denominator == 0) | ~np.isfinite(denominator)
        safe_denominator = denominator.copy()
        safe_denominator[bad] = 1.0
        output /= safe_denominator
        output[:, bad] = 0.0
        denominator_label = "sample_std(first_difference)/sqrt(2)"
    elif normalized_mode == "zscore":
        output -= np.mean(output, axis=0)
        denominator = np.std(output, axis=0, ddof=1)
        bad = (denominator == 0) | ~np.isfinite(denominator)
        safe_denominator = denominator.copy()
        safe_denominator[bad] = 1.0
        output /= safe_denominator
        output[:, bad] = 0.0
        denominator_label = "sample_std"
    else:
        bad = np.zeros(output.shape[1], dtype=bool)
        denominator_label = None
    output[~np.isfinite(output)] = 0.0
    info: dict[str, Any] = {
        "mode": normalized_mode,
        "ddof": 1 if normalized_mode != "none" else None,
        "denominator": denominator_label,
        "zero_or_nonfinite_scale_voxels": int(bad.sum()),
    }
    return output, info


def _resolve_config(config: ConfigInput) -> PreprocessingConfig:
    if config is None:
        return PreprocessingConfig()
    if not isinstance(config, PreprocessingConfig):
        raise TypeError("config must be PreprocessingConfig")
    return config


def _validate_preprocessing_config(cfg: PreprocessingConfig) -> None:
    _validate_nonnegative_integer(cfg.drop_initial_volumes, "drop_initial_volumes")
    _validate_nonnegative_integer(cfg.confounds.acompcor_components, "acompcor_components")
    _validate_nonnegative_integer(cfg.confounds.tcompcor_components, "tcompcor_components")
    _validate_positive_integer(cfg.filter_order_highpass, "filter_order_highpass")
    _validate_positive_integer(cfg.filter_order_lowpass, "filter_order_lowpass")
    _normalise_filter_backend(cfg.filter_backend)
    _normalise_standardization_mode(cfg.standardization)
    _as_spatial_triplet(
        cfg.spatial_smoothing_fwhm_mm,
        name="spatial_smoothing_fwhm_mm",
        allow_zero=True,
    )
    _positive_finite_float(
        cfg.smoothing_kernel_truncate_sigma,
        "smoothing_kernel_truncate_sigma",
    )
    for name in (
        "motion_outliers",
        "csf",
        "white_matter",
        "global_signal",
    ):
        if not isinstance(getattr(cfg.confounds, name), (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean.")


def _validate_filter_settings(
    cfg: PreprocessingConfig, tr: float, n_timepoints: int
) -> dict[str, Any]:
    nyquist = 0.5 / tr
    low = cfg.bandpass_low_hz
    high = cfg.bandpass_high_hz
    if low is not None:
        low = _positive_finite_float(low, "low_frequency_hz")
        if low >= nyquist:
            raise ValueError(
                f"low_frequency_hz={low:g} must be below Nyquist={nyquist:g} Hz."
            )
    if high is not None:
        high = _positive_finite_float(high, "high_frequency_hz")
        if high >= nyquist:
            raise ValueError(
                f"high_frequency_hz={high:g} must be below Nyquist={nyquist:g} Hz."
            )
    if low is not None and high is not None and low >= high:
        raise ValueError("low_frequency_hz must be lower than high_frequency_hz.")

    backend = _normalise_filter_backend(cfg.filter_backend)
    highpass_padlen = None
    if low is not None and backend == "butterworth":
        highpass_padlen = 3 * cfg.filter_order_highpass
        if n_timepoints <= highpass_padlen:
            raise ValueError(
                f"BOLD has {n_timepoints} retained volumes, but the Butterworth "
                f"high-pass requires more than {highpass_padlen}."
            )
    lowpass_padlen = None
    if high is not None:
        lowpass_padlen = 3 * cfg.filter_order_lowpass
        if n_timepoints <= lowpass_padlen:
            raise ValueError(
                f"BOLD has {n_timepoints} retained volumes, but the Butterworth "
                f"low-pass requires more than {lowpass_padlen}."
            )

    n_dct = 0
    cutoff_period_seconds = None
    if low is not None and backend == "dct":
        cutoff_period_seconds = 1.0 / low
        n_total = int(
            np.floor(2.0 * (n_timepoints * tr) / cutoff_period_seconds + 1.0)
        )
        n_dct = max(0, min(n_total, n_timepoints) - 1)
    return {
        "backend": backend,
        "low_frequency_hz": low,
        "high_frequency_hz": high,
        "highpass_period_seconds": cutoff_period_seconds,
        "dct_regressor_count_excluding_constant": n_dct,
        "highpass_order": cfg.filter_order_highpass if backend == "butterworth" else None,
        "lowpass_order": cfg.filter_order_lowpass if high is not None else None,
        "filtfilt_padtype": "odd" if highpass_padlen or lowpass_padlen else None,
        "highpass_padlen": highpass_padlen,
        "lowpass_padlen": lowpass_padlen,
    }


def _butterworth_filtfilt(
    data: np.ndarray,
    *,
    cutoff_hz: float,
    tr: float,
    order: int,
    kind: str,
) -> np.ndarray:
    nyquist = 0.5 / tr
    coefficients_b, coefficients_a = butter(order, cutoff_hz / nyquist, btype=kind)
    # Use three filter orders of odd-symmetric edge extension.
    padlen = 3 * (max(len(coefficients_a), len(coefficients_b)) - 1)
    return filtfilt(
        coefficients_b,
        coefficients_a,
        data,
        axis=0,
        padtype="odd",
        padlen=padlen,
        method="pad",
    )


def _numeric_frame(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    try:
        return frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} contain non-numeric values: {error}") from error


def _validate_bold(bold: np.ndarray) -> np.ndarray:
    array = np.asarray(bold)
    if array.ndim != 4:
        raise ValueError(f"BOLD must be 4-D (X, Y, Z, T); got shape {array.shape}.")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("BOLD must be a real numeric array.")
    if any(size <= 0 for size in array.shape):
        raise ValueError("BOLD dimensions must all be non-empty.")
    return array


def _validate_mask(mask: np.ndarray, spatial_shape: Sequence[int]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"brain_mask must be 3-D; got shape {array.shape}.")
    if tuple(array.shape) != tuple(spatial_shape):
        raise ValueError(
            f"brain_mask shape {array.shape} does not match BOLD spatial "
            f"shape {tuple(spatial_shape)}."
        )
    if not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_):
        raise TypeError("brain_mask must be numeric or boolean.")
    if np.iscomplexobj(array):
        raise TypeError("brain_mask must be real-valued.")
    _require_finite(array, "brain_mask")
    result = array > 0
    if not result.any():
        raise ValueError("brain_mask contains no positive voxels.")
    return result


def _validate_2d_timeseries(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError(f"timeseries must have shape (T, P); got {values.shape}.")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError("timeseries must be a real numeric array.")
    if values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("timeseries must have at least two rows and one column.")
    return np.asarray(values, dtype=np.float64)


def _linear_detrend(data: np.ndarray) -> np.ndarray:
    n_timepoints = data.shape[0]
    design = np.column_stack(
        (np.ones(n_timepoints, dtype=np.float64), np.arange(n_timepoints, dtype=np.float64))
    )
    beta, _, _, _ = np.linalg.lstsq(design, data, rcond=None)
    result = data - design @ beta
    _require_finite(result, "linearly detrended data")
    return result


def _fwhm_to_sigma_vox(
    fwhm_mm: float | Sequence[float], voxel_size_mm: tuple[float, float, float]
) -> tuple[float, float, float]:
    fwhm = _as_spatial_triplet(fwhm_mm, name="smoothing_fwhm_mm", allow_zero=True)
    if np.isscalar(fwhm_mm):
        scalar_sigma = fwhm[0] / float(np.mean(voxel_size_mm)) / 2.355
        return (scalar_sigma, scalar_sigma, scalar_sigma)
    return tuple(
        axis_fwhm / axis_size / 2.355
        for axis_fwhm, axis_size in zip(fwhm, voxel_size_mm, strict=True)
    )


def _validate_voxel_sizes(value: float | Sequence[float]) -> tuple[float, float, float]:
    return _as_spatial_triplet(value, name="voxel_size_mm", allow_zero=False)


def _as_spatial_triplet(
    value: float | Sequence[float], *, name: str, allow_zero: bool
) -> tuple[float, float, float]:
    if np.isscalar(value):
        raw = (value, value, value)
    else:
        try:
            raw = tuple(value)
        except TypeError as error:
            raise TypeError(f"{name} must be a scalar or length-three sequence.") from error
        if len(raw) != 3:
            raise ValueError(f"{name} must be a scalar or length-three sequence.")
    result = tuple(float(item) for item in raw)
    if not all(np.isfinite(item) for item in result):
        raise ValueError(f"{name} values must be finite.")
    if allow_zero:
        valid = all(item >= 0 for item in result)
        wording = "non-negative"
    else:
        valid = all(item > 0 for item in result)
        wording = "positive"
    if not valid:
        raise ValueError(f"{name} values must be {wording}.")
    return result  # type: ignore[return-value]


def _validate_tr(tr: float) -> float:
    return _positive_finite_float(tr, "tr")


def _positive_finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number, not boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number.") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return result


def _validate_nonnegative_integer(value: Any, name: str) -> None:
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_integer(value: Any, name: str) -> None:
    _validate_nonnegative_integer(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _normalise_filter_backend(value: str) -> str:
    normalized = str(value).lower()
    if normalized not in {"dct", "butterworth"}:
        raise ValueError("filter_backend must be dct or butterworth")
    return normalized


def _normalise_standardization_mode(value: str) -> str:
    normalized = str(value).lower()
    modes = {"diff_noise": "diff_noise_std", "zscore": "zscore", "none": "none"}
    if normalized not in modes:
        raise ValueError("standardization must be diff_noise, zscore, or none")
    return modes[normalized]


def _require_finite(array: np.ndarray, name: str) -> None:
    finite = np.isfinite(array)
    if not finite.all():
        nan_count = int(np.isnan(array).sum())
        inf_count = int(np.isinf(array).sum())
        raise ValueError(
            f"{name} contains non-finite values (NaN={nan_count}, Inf={inf_count})."
        )


def _unique_preserving_order(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _json_number_or_list(value: float | Sequence[float]) -> float | list[float]:
    if np.isscalar(value):
        return float(value)
    return [float(item) for item in value]


__all__ = [
    "FRISTON_24_COLUMNS",
    "apply_temporal_filter",
    "build_fmriprep_confound_matrix",
    "preprocess_fmriprep_bold",
    "smooth_masked_bold",
    "dct_highpass",
    "standardize_timeseries",
]
