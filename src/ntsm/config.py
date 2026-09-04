"""Configuration models for nTSM."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConfoundsConfig:
    """fMRIPrep nuisance regressors used during downstream cleaning."""

    acompcor_components: int = 4
    tcompcor_components: int = 4
    motion_outliers: bool = True
    csf: bool = False
    white_matter: bool = False
    global_signal: bool = False


@dataclass
class PreprocessingConfig:
    """Downstream preprocessing applied to one complete fMRI run."""

    drop_initial_volumes: int = 6
    spatial_smoothing_fwhm_mm: float = 4.0
    bandpass_low_hz: float = 0.01
    bandpass_high_hz: float = 0.18
    filter_order_highpass: int = 2
    filter_order_lowpass: int = 4
    filter_backend: str = "dct"
    standardization: str = "zscore"
    smoothing_kernel_truncate_sigma: float = 2.0
    confounds: ConfoundsConfig = field(default_factory=ConfoundsConfig)


@dataclass
class TissueConfig:
    """Preparation of fMRIPrep tissue-probability maps."""

    smoothing_fwhm_mm: float = 4.0
    normalize_after_smoothing: bool = False


@dataclass
class MappingConfig:
    """Static nTSM graph-diffusion and spectral readout parameters."""

    # Main user-adjustable parameters (--nstep and --k).
    n_step: int = 20
    k: int = 125

    # The remaining mapping settings normally stay at their defaults.
    spatial_radius: int = 10
    neighborhood: int = 26
    distance_decay_power: float = 1.0
    random_walk_stop_mode: str = "converge_or_nstep"
    random_walk_tolerance: float = 1e-8
    restrict_gm_to_hemisphere: bool = True
    midline_voxel_one_based: float | None = None
    minimum_local_voxels: int = 20
    failed_voxel_value: float = float("nan")
    chunk_size: int = 10_000

    # 0 selects automatically (up to 32); a positive value requests that many threads.
    workers: int = 0


@dataclass
class InputConfig:
    """Run selection and fMRIPrep derivative naming."""

    space: str = "MNI152NLin6Asym"
    resolution: str = "02"
    task: str | None = None
    run: str | None = None
    session: str | None = None
    analysis_mask: str = "bold"


@dataclass
class OutputConfig:
    """Output and cache behavior."""

    save_preprocessed_bold: bool = False
    save_smoothed_tissue_maps: bool = False
    overwrite: bool = False


@dataclass
class NTSMConfig:
    """Runtime settings for an nTSM run."""

    input: InputConfig = field(default_factory=InputConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    tissue: TissueConfig = field(default_factory=TissueConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Check configuration values."""

        p = self.preprocessing
        m = self.mapping
        t = self.tissue
        if p.drop_initial_volumes < 0:
            raise ValueError("preprocessing.drop_initial_volumes must be nonnegative")
        if p.spatial_smoothing_fwhm_mm < 0 or t.smoothing_fwhm_mm < 0:
            raise ValueError("smoothing FWHM values must be nonnegative")
        if not (0 <= p.bandpass_low_hz < p.bandpass_high_hz):
            raise ValueError("band-pass frequencies must satisfy 0 <= low < high")
        if p.standardization not in {"zscore", "diff_noise", "none"}:
            raise ValueError("standardization must be zscore, diff_noise, or none")
        if p.filter_backend not in {"dct", "butterworth"}:
            raise ValueError("filter_backend must be dct or butterworth")
        if m.n_step < 1 or m.k < 1 or m.spatial_radius < 0:
            raise ValueError("n_step and k must be positive; spatial_radius must be nonnegative")
        if m.neighborhood != 26:
            raise ValueError("mapping.neighborhood must be 26")
        if m.random_walk_stop_mode not in {"converge_or_nstep", "fixed_nstep"}:
            raise ValueError("unsupported random_walk_stop_mode")
        if m.minimum_local_voxels < 1 or m.chunk_size < 1 or m.workers < 0:
            raise ValueError(
                "minimum_local_voxels/chunk_size must be positive; workers nonnegative"
            )
        if not self.input.analysis_mask:
            raise ValueError("input.analysis_mask must be bold or a NIfTI path")
        if self.input.resolution not in {"2", "02"}:
            raise ValueError("nTSM requires 2 mm fMRIPrep data")
