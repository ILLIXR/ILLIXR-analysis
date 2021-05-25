## Installation

```
# Clone the code
$ git clone --recursive git@github.com:ILLIXR/ILLIXR-analysis.git
$ cd ILLIXR-analysis

# Check Python version
$ python3 --version
Python 3.8.3
```

If this does not report 3.8 or later, you can install 3.8 locally, without
root, and without modifying your system's Python through
[pyenv][pyenv]

```
# Install pyenv
$ curl https://pyenv.run | bash

# Install Python 3.8, without root, without modifying system's Python
$ pyenv install 3.8.3

# Activate Python 3.8.3, but only when in this project's directory.
$ pyenv local 3.8.3
```

Once you installed Python 3.8.x, you verify that it activated and
install dependencies.

```
# Verify we are using the right Python version
$ python3 --version
Python 3.8.3

# Install Poetry
$ python3 -m pip install poetry

# Install system dependencies
$ sudo apt-get install -y graphviz graphviz-dev libcgraph6

# Install project dependencies
$ poetry install
```

## Usage

For everyday usage, you can use `poetry run ...` for one-off commands,
or `poetry shell` for an interactive shell with the right `$PATH` and
`$PYTHONPATH`.

To run the code,
```
$ poetry shell
$ python -m illixr.analysis.main main
```

Before comitting, run the linter and autoformatter. Make sure there is no red text in the following command:

```
$ poetry shell
$ python -m illixr.analysis.main check [--no-modify] [--verbose]
# This will modify your code!
```

## Structure of the code

All of the code resides in the path `illixr/analysis` (AKA the package `illixr.analysis`).

- `main.py` is the entrypoint.
- `analyze_trials.py` defines the "per-trial" analyses.
  - These only look at one trial at a time. Thus they are perfectly parallelizable across trials.
    - Unfortunately, exploiting all of this parallelism at once causes us to run out of memory.
    - Instead, I will do batches at a time, and reduce the data (delete and GC) before moving to the next batch. All of the human-readable results have bene written to disk, so I need only keep that data which is necessary for the next phase (inter-trial analyses).
  - Like LLVM, analyses can operate on the results of previous analyses. For example, DFG construction requires the results of the call tree construction.
  - Every individual analysis is a [dask.delayed](https://docs.dask.org/en/latest/delayed.html) function; calling a delayed function merely returns the key representing that computation in a DAG. I call `dask.compute(...)` on a whole batch of per-trial analyses, which evaluates the delayed functions.
  - Dask parallelism can be disabled with `use_parallel = True`.
  - Most individual analyses are cached with `charmonium.cache`. There is an unfortunate bug when the cache is used by multiple functions in parallel, so it does save as much compute as I would like. I will fix this soon. Once the program gets to the serial (inter-trial analyses), progress is saved.
  - The config is passed through, so it is accessible to the analysis.
  - This generates output inside each metrics directory.

- `analyze_trials2.py` defines the "inter-trial" analyses. In future revisions, I will change `analyze_trials.py` to `per_trial_analyses.py` and `analyze_trials2.py` to `inter_trial_analyses.py` or some scheme that actually makes sense.
  - This is responsible for requesting batches of reduced per-trial data from `analyze_trials.py`. Then it runs functions that operate looking at all of the reduced data from every run together.
  - Since the config is passed through to this point, the first step is to combine the datas that come from trials with identical `conditions`.
  - This generates output in the `output_dir`.
  - When comparing datasets from identical runs, I use the [Kolmogorov-Smirnov statistical test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) (AKA [`scipy.stats.ks_2samp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)) to determine, "were these two datasets drawn from the same distribution?" This tests **internal consistency**.
    - The KS test is like [Student's T-test](https://en.wikipedia.org/wiki/Student%27s_t-test), except it doesn't assume that the distributions have the same variance. It's like [Welch's T-test](https://en.wikipedia.org/wiki/Welch%27s_t-test) except it doesn't assume a normal distribution.
    - In our data, the answer was sometimes yes (p < 0.95), and sometimes no (with p = 0.95). This is worrisome, and I raised this during meetings, but I think the consensus was that we should focus on the other graphs and other indicators of our hypothesis.
    - I also am not sure how to use the ks test when I have N datasets. So far, I have been doing N(N-2)/2 pairwise two-dataset tests. Given a huge number of datasets, we expect that some might coincidentally fail the KS test. But we don't have that many samples of identical conditions (the value of `iters` in the metarunner), so I think it's still bad to fail the KS test.

- `analyze_trials3.py` has sketchy code that I did at the last minute. It should be incorporated into `analyze_trials2.py`. It combines the timeseries from multiple runs. This is done to be consistent with the ROS experiments that Aditi has in the paper. I think this is not valid though, since the time-series of different trials with the same conditions have a different "shape." Furthermore, it hides variance since it is averaging out multiple runs, and one of our claims is about reducing variance.

## Guiding principles

- Avoid human-intervention. E.g. Avoid manually renaming or deleting directories.

- Use functions instead of scripts for each phase of analysis.
  - Functions should pass the data in-memory rather than on-disk (e.g. operate on Numpy arrays not CSV files) to avoid the burden of parsing data and shell arguments.
  - Functions should be stateless but can use `charmonium.cache` to speed them up when they haven't changed.
  - Functions can use DAG-parallelism or parallel-map to speed them up, or both. An appropriate coarseness for parallelism is one thread of one trial or one trial of ILLIXR.
  - Unfortunately, as I saw preparing for SOSP, this can create out-of-memory situations. I think the long-term fix is to change the order that nodes in the DAG get evaluated (the scheduler). The current Dask scheduler tries to expose as much parallelism as possible, but in our DAG, finding parallelism is no problem. If the N trials are on the left and N trial analysis outputs are on the right, and a DAG of intermediate results connects them, the default scheduler wants to run one column (N things in memory) at a time, while it would be better for us to run a few, say M << N, rows at a time (M things in memory). If M is set to the number of CPUs, we still exploit maximal parallelism.

- Use static types. Data processing is long enough that bugs might not cause faults until ten minutes in. Static typing helps us fail earlier.
  - I had ot abandon this for SOSP. That was a mistake. In a several cases, the script crunched data for 30 minutes only to fail due to a name error, type error, or wrong arguments error. Type-checking could have caught that ahead of time.

- Don't commit data to this repo. I tried that once; Since git stores a whole branching history of eventually-obsolete data, it made cloning the repo take as long as draining an ocean.

- Issue warnings when the data is sketchy, and an assertion when the it is inconsistent.

- The exact inputs are placed with the outputs and passed into the analysis.

- The whole analysis is one command, and it analyzed the whole grid, and generates all outputs, **verbatim** as they appear in the paper. This is made feasible by memoization and parallelism.
  - I had to slightly abandon this for SOSP due to `analyze_trials3.py`. Most of computation is still a single command, but the results need a second.
