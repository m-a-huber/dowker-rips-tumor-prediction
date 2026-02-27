This repository contains code to recreate the microenvironment classification results from the paper [<em>Flagifying the Dowker Complex</em>](https://arxiv.org/abs/2508.08025).

---

__Requirements__

Required dependencies are specified in `pyproject.toml`.
The dependencies `dowker-complex` and `dowker-rips-complex` can be installed either from source from [here](https://github.com/m-a-huber/dowker-complex) and [here](https://github.com/m-a-huber/dowker-rips-complex), respectively, or via `pip` by running e.g. `pip install -U dowker-complex` and `pip install -U dowker-rips-complex`, respectively.

---

__Reproducing results__

To reproduce the results, run the command `python main.py <complex> [options]`, where
- `<complex>` must be one of `dowker_rips` and `dowker`, and indicates whether to use the Dowker or the Dowker-Rips complex;
- `--n-repeats <int>` indicates the number of times training of the SVM is repeated (default: 10);
- `--n-jobs` indicates the number of jobs to run in parallel in hyperparameter tuning (default: 1);
- `--overwrite` indicates whether results existing on disk should be overwritten or not; and
- `--verbose` increases the level of verbosity during execution of the script.

For example, to reproduce the results of the paper using the Dowker-Rips complex with verbose output, run `python main.py dowker_rips --n-repeats 10 --verbose 1`

Executing `main.py` as above will create a directory named `outfiles` that contains the processed point cloud files, the persistence data, the persistence images as well as an array containing the accuracies of each of the `--n-repeats` many SVMs trained and evaluated in the process.

---

__For users of `uv`__

If `uv` is installed, the dependencies `dowker-complex` and `dowker-rips-complex` can be installed by running e.g. `uv add dowker-complex` and `uv add dowker-rips-complex`, respectively.
The required dependencies and the environment specified in `uv.lock` can be recreated by running `uv sync`.

To reproduce the results from the paper, run `uv run main.py <complex> [options]`, with options specified as above.
