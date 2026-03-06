This repository contains code to recreate the microenvironment classification results from the paper [<em>Flagifying the Dowker Complex</em>](https://arxiv.org/abs/2508.08025).

---

## Requirements

- Python `>=3.10`
- Dependencies in `pyproject.toml`
- Data in `data/` (included in this repository)

The environment specified in `uv.lock` can be recreated by running

```bash
uv sync
```

---

## Reproducing classification results

To reproduce the results of the tumor microenvironment classification pipeline, run

```bash
uv run experiment.py <complex> [options]
```

where

- `<complex>` is one of `dowker` or `dowker_rips`;
- `--n-repeats <int>` is the number of repeated SVM evaluations (default:
  `10`);
- `--n-jobs <int>` is the number of jobs used for hyperparameter tuning
  (default: `1`);
- `--overwrite` overwrites existing outputs; and
- `--verbose <int>` sets verbosity (default: `0`; values `>=1` print progress).

As an example, to reproduce the results from the paper setting using the Dowker-Rips complex run

```bash
uv run experiment.py dowker_rips --verbose 1
```

This creates/updates `outfiles/` with

- processed point clouds (`outfiles/point_clouds_processed/`);
- persistence files (`outfiles/<complex>_persistences/`);
- persistence images (`outfiles/<complex>_persistence_images/`); and
- SVM accuracy arrays/statistics written by the experiment pipeline.

---

## Recreating benchmarking results

To reproduce the benchmarking results, run

```bash
uv run benchmarking.py --vary <size|dim> [options]
```

with options:

- `--vary <size|dim>`: vary point-cloud size or ambient dimension (required);
- `--max-exponent <int>`: number of powers of two tested (default: `11`);
- `--n-datasets <int>`: independently generated datasets per setting (default:
  `10`);
- `--n-repeats <int>`: repeated timings per dataset/configuration (default:
  `5`);
- `--seed <int>`: random seed (default: `42`);
- `--verbose`: print progress; and
- `--overwrite`: overwrite existing CSVs/plots.

As an example, to reproduce the benchmarking results from the paper run

```bash
uv run benchmarking.py --vary size --verbose
```

and

```bash
uv run benchmarking.py --vary dim --verbose
```

This creates/updates `benchmarking_results/` with

- CSV results (`benchmarking_results/benchmarking_results_*.csv`); and
- plots (`benchmarking_results/benchmarking_results_*.pdf` and `benchmarking_results/benchmarking_results_*.svg`).