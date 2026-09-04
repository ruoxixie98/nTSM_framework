"""Command-line interface for nTSM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .bids import discover_fmriprep_runs, ntsm_output_name
from .config import NTSMConfig
from .pipeline import run_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ntsm", description="Static whole-run nTSM for fMRIPrep derivatives"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="preprocess selected fMRI runs and compute nTSM maps")
    run.add_argument(
        "bids_root",
        type=Path,
        help="BIDS dataset root; fMRIPrep is expected under derivatives/fmriprep by default",
    )
    run.add_argument(
        "--output-root",
        type=Path,
        help="output directory (default: BIDS_ROOT/derivatives/nTSM)",
    )
    run.add_argument(
        "--fmriprep-root",
        type=Path,
        help="fMRIPrep derivative root when it is outside the BIDS dataset",
    )
    run.add_argument(
        "--participant-label",
        nargs="+",
        metavar="LABEL",
        help="one or more participant labels, with or without the sub- prefix",
    )
    run.add_argument(
        "--session-label",
        metavar="LABEL",
        help="select one BIDS session, with or without the ses- prefix",
    )
    run.add_argument(
        "--task",
        metavar="LABEL",
        help="select one BIDS task, for example rest for task-rest; omit to process all tasks",
    )
    run.add_argument(
        "--run-label",
        metavar="LABEL",
        help="select one BIDS run, with or without the run- prefix",
    )
    run.add_argument(
        "--space",
        help="fMRIPrep output space (default: MNI152NLin6Asym)",
    )
    run.add_argument("--nstep", type=int, help="random-walk step limit (default: 20)")
    run.add_argument("--k", type=int, help="number of selected local voxels (default: 125)")
    run.add_argument(
        "--workers",
        type=int,
        help="parallel worker threads; 0 selects automatically, up to 32 (default: 0)",
    )
    run.add_argument(
        "--analysis-mask",
        metavar="MASK",
        help="custom NIfTI mask path; omit or use bold for the fMRIPrep BOLD mask",
    )
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing outputs instead of reusing them",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="show matching runs without processing them",
    )
    return parser


def _apply_overrides(config, arguments) -> None:
    if arguments.session_label is not None:
        config.input.session = arguments.session_label
    if arguments.task is not None:
        config.input.task = arguments.task
    if arguments.run_label is not None:
        config.input.run = arguments.run_label
    if arguments.space is not None:
        config.input.space = arguments.space
    if arguments.nstep is not None:
        config.mapping.n_step = arguments.nstep
    if arguments.k is not None:
        config.mapping.k = arguments.k
    if arguments.workers is not None:
        config.mapping.workers = arguments.workers
    if arguments.analysis_mask is not None:
        config.input.analysis_mask = arguments.analysis_mask
    if arguments.overwrite:
        config.output.overwrite = True
    config.validate()


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = NTSMConfig()
        _apply_overrides(config, arguments)
        output_root = arguments.output_root or arguments.bids_root / "derivatives" / "nTSM"
        if arguments.dry_run:
            runs = discover_fmriprep_runs(
                arguments.bids_root,
                config.input,
                fmriprep_root=arguments.fmriprep_root,
                participant_labels=arguments.participant_label,
            )
            for item in runs:
                output_name = ntsm_output_name(
                    item, config.mapping.n_step, config.mapping.k
                )
                print(f"{item.bold} -> {output_name}")
            return 0
        run_pipeline(
            arguments.bids_root,
            output_root,
            config,
            fmriprep_root=arguments.fmriprep_root,
            participant_labels=arguments.participant_label,
        )
        return 0
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as error:
        print(f"ntsm: error: {error}", file=sys.stderr)
        return 2
