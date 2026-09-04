"""Discovery of fMRIPrep BOLD runs and tissue probability maps."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .config import InputConfig

_ENTITY_PATTERN = re.compile(r"(?:^|_)(?P<key>[A-Za-z0-9]+)-(?P<value>[^_]+)")


@dataclass(frozen=True)
class FMRIPrepRun:
    """All files needed to process one complete fMRIPrep BOLD run."""

    subject: str
    session: str | None
    task: str
    run: str | None
    entities: dict[str, str]
    bold: Path
    bold_json: Path
    brain_mask: Path
    confounds_tsv: Path
    gm_probseg: Path
    wm_probseg: Path
    csf_probseg: Path
    repetition_time: float

    @property
    def run_key(self) -> str:
        """Return a concise session/task/run identifier for logs and folders."""

        parts = []
        if self.session:
            parts.append(self.session)
        parts.append(f"task-{self.task}")
        if self.run:
            parts.append(f"run-{self.run}")
        return "_".join(parts)


def parse_entities(name: str) -> dict[str, str]:
    """Parse underscore-delimited BIDS-style entities from a filename."""

    stem = name.removesuffix(".nii.gz").removesuffix(".nii")
    return {match.group("key"): match.group("value") for match in _ENTITY_PATTERN.finditer(stem)}


def _normalise_label(value: str | None, prefix: str) -> str | None:
    if value is None:
        return None
    return value.removeprefix(f"{prefix}-")


def _same_resolution(actual: str | None, requested: str) -> bool:
    if actual is None:
        return False
    if actual.isdigit() and requested.isdigit():
        return int(actual) == int(requested)
    return actual == requested


def _single_existing(paths: Iterable[Path], description: str) -> Path:
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    if len(unique) != 1:
        rendered = "\n  ".join(str(path) for path in unique) or "<none>"
        raise FileNotFoundError(
            f"expected exactly one {description}; found {len(unique)}:\n  {rendered}"
        )
    return unique[0]


def _find_tissue_map(
    subject_dir: Path,
    session: str | None,
    label: str,
    settings: InputConfig,
) -> Path:
    anat_dirs = []
    if session:
        anat_dirs.append(subject_dir / session / "anat")
    anat_dirs.append(subject_dir / "anat")
    candidates: list[Path] = []
    for anat_dir in anat_dirs:
        if not anat_dir.is_dir():
            continue
        for path in anat_dir.glob(f"*_label-{label}_probseg.nii*"):
            entities = parse_entities(path.name)
            if entities.get("space") != settings.space:
                continue
            if not _same_resolution(entities.get("res"), settings.resolution):
                continue
            candidates.append(path)
        if candidates:
            break
    return _single_existing(candidates, f"{label} probability map")


def _matching_auxiliary_files(bold: Path) -> tuple[Path, Path, Path]:
    name = bold.name
    suffix = "_desc-preproc_bold.nii.gz"
    if not name.endswith(suffix):
        raise ValueError(f"unsupported fMRIPrep BOLD name: {bold}")
    brain_mask = bold.with_name(name.replace(suffix, "_desc-brain_mask.nii.gz"))
    bold_json = bold.with_name(name.removesuffix(".nii.gz") + ".json")
    prefix = name.split("_space-", maxsplit=1)[0]
    confounds = bold.with_name(f"{prefix}_desc-confounds_timeseries.tsv")
    missing = [path for path in (brain_mask, bold_json, confounds) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing matching fMRIPrep file(s): " + ", ".join(map(str, missing))
        )
    return bold_json, brain_mask, confounds


def _read_repetition_time(sidecar: Path) -> float:
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    try:
        repetition_time = float(payload["RepetitionTime"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid RepetitionTime in {sidecar}") from error
    if not (repetition_time > 0):
        raise ValueError(f"RepetitionTime must be positive in {sidecar}")
    return repetition_time


def discover_fmriprep_runs(
    bids_root: str | Path,
    settings: InputConfig,
    *,
    fmriprep_root: str | Path | None = None,
    participant_labels: Iterable[str] | None = None,
) -> list[FMRIPrepRun]:
    """Discover complete fMRIPrep runs using explicit space/resolution filters."""

    root = Path(fmriprep_root) if fmriprep_root else Path(bids_root) / "derivatives" / "fmriprep"
    if not root.is_dir():
        raise FileNotFoundError(f"fMRIPrep derivative root does not exist: {root}")
    requested_subjects = {
        f"sub-{_normalise_label(value, 'sub')}" for value in (participant_labels or [])
    }
    requested_session = _normalise_label(settings.session, "ses")
    requested_run = _normalise_label(settings.run, "run")
    runs: list[FMRIPrepRun] = []
    pattern = "sub-*/**/func/*_space-*_res-*_desc-preproc_bold.nii.gz"
    for bold in sorted(root.glob(pattern)):
        entities = parse_entities(bold.name)
        subject = f"sub-{entities['sub']}" if "sub" in entities else bold.parts[-3]
        if requested_subjects and subject not in requested_subjects:
            continue
        if entities.get("space") != settings.space:
            continue
        if not _same_resolution(entities.get("res"), settings.resolution):
            continue
        if settings.task and entities.get("task") != _normalise_label(settings.task, "task"):
            continue
        if requested_run and entities.get("run") != requested_run:
            continue
        session_value = entities.get("ses")
        if requested_session and session_value != requested_session:
            continue
        task = entities.get("task")
        if not task:
            raise ValueError(f"BOLD filename has no task entity: {bold}")
        subject_dir = root / subject
        session = f"ses-{session_value}" if session_value else None
        bold_json, brain_mask, confounds = _matching_auxiliary_files(bold)
        runs.append(
            FMRIPrepRun(
                subject=subject,
                session=session,
                task=task,
                run=entities.get("run"),
                entities=entities,
                bold=bold,
                bold_json=bold_json,
                brain_mask=brain_mask,
                confounds_tsv=confounds,
                gm_probseg=_find_tissue_map(subject_dir, session, "GM", settings),
                wm_probseg=_find_tissue_map(subject_dir, session, "WM", settings),
                csf_probseg=_find_tissue_map(subject_dir, session, "CSF", settings),
                repetition_time=_read_repetition_time(bold_json),
            )
        )
    if not runs:
        raise FileNotFoundError(
            f"no matching fMRIPrep BOLD runs under {root} for "
            f"space={settings.space}, resolution={settings.resolution}"
        )
    return runs


def ntsm_output_name(run: FMRIPrepRun, n_step: int, k: int, extension: str = ".nii.gz") -> str:
    """Build an nTSM output filename."""

    parts = [run.subject]
    if run.session:
        parts.append(run.session)
    parts.append(f"task-{run.task}")
    if run.run:
        parts.append(f"run-{run.run}")
    parts.extend(("nTSM", f"nstep-{n_step}", f"k-{k}"))
    return "_".join(parts) + extension
