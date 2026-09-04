"""End-to-end BIDS/fMRIPrep workflow for static nTSM maps."""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bids import FMRIPrepRun, discover_fmriprep_runs, ntsm_output_name
from .config import NTSMConfig
from .core import compress_signal_to_mask, compute_ntsm_map, stack_tissue_probabilities
from .io import (
    assert_anatomical_xyz_axes,
    assert_same_grid,
    load_nifti_float32,
    save_nifti_float32,
    voxel_sizes_mm,
)
from .preprocessing import preprocess_fmriprep_bold
from .tissue import (
    normalize_tissue_probabilities,
    prepare_tissue_file,
    smooth_tissue_probability,
)


def _run_output_directory(root: Path, run: FMRIPrepRun) -> Path:
    directory = root / run.subject
    if run.session:
        directory /= run.session
    return directory / "func"


def _prepare_tissues(
    run: FMRIPrepRun,
    output_root: Path,
    config: NTSMConfig,
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    cache = output_root / "tissue" / run.subject
    if run.session:
        cache /= run.session
    if config.output.save_smoothed_tissue_maps:
        cache.mkdir(parents=True, exist_ok=True)
    sources = {"GM": run.gm_probseg, "WM": run.wm_probseg, "CSF": run.csf_probseg}
    images = {}
    arrays = {}
    for label, source in sources.items():
        if config.output.save_smoothed_tissue_maps:
            target = cache / f"{run.subject}_{label}_smoothed_probseg.nii.gz"
            image, data, _ = prepare_tissue_file(
                source,
                target,
                fwhm_mm=config.tissue.smoothing_fwhm_mm,
                overwrite=config.output.overwrite,
            )
        else:
            image, raw = load_nifti_float32(source)
            if raw.ndim != 3:
                raise ValueError(f"tissue probability image is not 3-D: {source}")
            data = smooth_tissue_probability(
                raw,
                config.tissue.smoothing_fwhm_mm,
                voxel_sizes_mm(image),
            )
        images[label] = image
        arrays[label] = data
    assert_same_grid([(label, images[label]) for label in ("GM", "WM", "CSF")])
    if config.tissue.normalize_after_smoothing:
        arrays["GM"], arrays["WM"], arrays["CSF"] = normalize_tissue_probabilities(
            arrays["GM"], arrays["WM"], arrays["CSF"]
        )
    return images["GM"], arrays["GM"], arrays["WM"], arrays["CSF"]


def _analysis_mask(
    config: NTSMConfig,
    bold_mask: np.ndarray,
) -> tuple[np.ndarray, object | None]:
    mode = config.input.analysis_mask
    if mode == "bold":
        return np.asarray(bold_mask > 0, dtype=bool), None
    mask_image, mask_data = load_nifti_float32(mode)
    return np.asarray(mask_data > 0, dtype=bool), mask_image


def process_run(
    run: FMRIPrepRun,
    output_root: str | Path,
    config: NTSMConfig,
    *,
    progress: Callable[[str], None] | None = print,
) -> dict[str, object]:
    """Clean one complete fMRI run and compute its static nTSM map."""

    derivative_root = Path(output_root)
    run_dir = _run_output_directory(derivative_root, run)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_name = ntsm_output_name(run, config.mapping.n_step, config.mapping.k)
    output_map = run_dir / output_name
    if output_map.is_file() and not config.output.overwrite:
        if progress:
            progress(f"Reusing existing output: {output_map}")
        return {
            "subject": run.subject,
            "run": run.run_key,
            "output": str(output_map),
            "status": "reused",
        }

    if progress:
        progress(f"Loading {run.subject} {run.run_key}")
    bold_image, bold = load_nifti_float32(run.bold)
    mask_image, bold_mask = load_nifti_float32(run.brain_mask)
    gm_image, gm, wm, csf = _prepare_tissues(run, derivative_root, config)
    assert_same_grid(
        [
            ("BOLD", bold_image),
            ("brain mask", mask_image),
            ("GM probability", gm_image),
        ]
    )
    assert_anatomical_xyz_axes(bold_image)
    if bold.ndim != 4 or bold_mask.ndim != 3:
        raise ValueError("the preprocessed BOLD must be 4-D and its brain mask 3-D")
    bold_voxel_sizes = voxel_sizes_mm(bold_image)
    if not np.allclose(bold_voxel_sizes, (2.0, 2.0, 2.0), rtol=0.0, atol=1e-3):
        raise ValueError(
            f"nTSM requires 2 mm isotropic data; got voxel sizes {bold_voxel_sizes} mm"
        )

    if progress:
        progress("Running downstream fMRI preprocessing")
    cleaned, _ = preprocess_fmriprep_bold(
        bold,
        bold_mask > 0,
        run.confounds_tsv,
        tr=run.repetition_time,
        voxel_size_mm=bold_voxel_sizes,
        config=config.preprocessing,
    )
    del bold
    gc.collect()

    cleaned_path = run_dir / output_name.replace(
        f"_nTSM_nstep-{config.mapping.n_step}_k-{config.mapping.k}.nii.gz",
        "_desc-cleaned_bold.nii.gz",
    )
    if config.output.save_preprocessed_bold:
        save_nifti_float32(cleaned, bold_image, cleaned_path)

    mask, optional_mask_image = _analysis_mask(config, bold_mask)
    if optional_mask_image is not None:
        assert_same_grid([("BOLD", bold_image), ("analysis mask", optional_mask_image)])
    signal, signal_row = compress_signal_to_mask(cleaned, mask)
    del cleaned
    gc.collect()
    tissue_matrix = stack_tissue_probabilities(gm, wm, csf)
    result, mapping_summary = compute_ntsm_map(
        signal,
        signal_row,
        tissue_matrix,
        mask,
        config.mapping,
        progress=progress,
    )
    save_nifti_float32(result, gm_image, output_map)
    return {
        "subject": run.subject,
        "session": run.session or "",
        "task": run.task,
        "run": run.run or "",
        "output": str(output_map),
        "status": "completed",
        **asdict(mapping_summary),
    }


def run_pipeline(
    bids_root: str | Path,
    output_root: str | Path,
    config: NTSMConfig,
    *,
    fmriprep_root: str | Path | None = None,
    participant_labels: Iterable[str] | None = None,
    progress: Callable[[str], None] | None = print,
) -> list[dict[str, object]]:
    """Discover selected runs and process them sequentially."""

    config.validate()
    derivative_root = Path(output_root)
    derivative_root.mkdir(parents=True, exist_ok=True)
    runs = discover_fmriprep_runs(
        bids_root,
        config.input,
        fmriprep_root=fmriprep_root,
        participant_labels=participant_labels,
    )
    if progress:
        progress(f"Found {len(runs)} matching fMRIPrep run(s)")
    rows = [process_run(run, derivative_root, config, progress=progress) for run in runs]
    return rows
