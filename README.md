# nTSM

`nTSM` (neuroanatomical topology-informed synergy mapping) is a pipeline for
voxel-wise analysis of fMRI runs. It reads a BIDS dataset and
its fMRIPrep derivatives, performs downstream preprocessing, and computes one
nTSM map for each selected run.

## Requirements

- Python 3.10 or newer
- a BIDS dataset
- fMRIPrep derivatives containing 2 mm isotropic MNI-space preprocessed BOLD,
  a matching brain mask and confounds TSV, and MNI-space GM/WM/CSF probability maps

The BOLD image, brain mask, and probability maps must use the same voxel grid
and anatomical x/y/z axis order.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install .
```

For development from a clone, use `python -m pip install -e .`.

## Quick start

```bash
ntsm run /data/my-study --participant-label 001 --task rest --output-root /data/my-study/derivatives/nTSM
```

Here, `/data/my-study` is the BIDS dataset root. By default, fMRIPrep
derivatives are read from `/data/my-study/derivatives/fmriprep`.
`--task rest` selects BIDS files containing the `task-rest` entity. It does not
change preprocessing or mapping. Selection options are optional; if `--task`
is omitted, all matching tasks are processed.

## Command-line options

Run `ntsm run --help` to see the same option reference in the terminal.

| Argument | Required | Meaning |
| --- | --- | --- |
| `BIDS_ROOT` | Yes | Root of the BIDS dataset. |
| `--participant-label LABEL [LABEL ...]` | No | Select one or more participants, such as `001 002`. The `sub-` prefix is optional. |
| `--session-label LABEL` | No | Select one session. The `ses-` prefix is optional. |
| `--task LABEL` | No | Select one BIDS task, such as `rest` for `task-rest`. Omit it to process all tasks. |
| `--run-label LABEL` | No | Select one run. The `run-` prefix is optional. |
| `--fmriprep-root PATH` | No | Location of the fMRIPrep derivatives when they are not under `BIDS_ROOT/derivatives/fmriprep`. |
| `--output-root PATH` | No | Output location. The default is `BIDS_ROOT/derivatives/nTSM`. |
| `--space LABEL` | No | Select the fMRIPrep output space. The default is `MNI152NLin6Asym`. |
| `--nstep N` | No | Override the random-walk step limit. The default is `20`. |
| `--k N` | No | Override the number of selected local voxels. The default is `125`. |
| `--workers N` | No | Set the parallel worker count. `0` chooses automatically, up to 32 workers. |
| `--analysis-mask MASK` | No | Custom NIfTI mask path. If omitted or set to `bold`, the fMRIPrep BOLD mask is used. |
| `--overwrite` | No | Replace existing outputs. Without it, existing outputs are reused. |
| `--dry-run` | No | Show matching runs without running the calculation. |

For example, select two participants and one specific session, task, and run:

```bash
ntsm run /data/my-study --participant-label 001 002 --session-label 01 --task rest --run-label 01 --dry-run
```

Mapping is parallelized across centre voxels. The default `workers: 0`
selects up to 32 threads; use `--workers N` to override it for a particular
computer.

## Citation

Citation information will be added when the associated manuscript becomes publicly available.

## Research-use notice

For research use only; not for clinical use.
