# ILLIXR Analysis

This repo contains scripts that analyze the data produced by ILLIXR.

## Installation

```
# Clone the code
$ git clone git@github.com:ILLIXR/ILLIXR-analysis.git
$ cd ILLIXR-analysis

# Check Python version
$ python3 --version
Python 3.8.3
```

If this does not reports 3.8.x, you can install 3.8.x locally, without
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

# Install project dependencies
$ poetry install
```

## Usage

For everyday usage, you can use `poetry run ...` for one-off commands,
or `poetry shell` for an interactive shell with the right `$PATH` and
`$PYTHONPATH`.

Before comitting, run the linter and autoformatter. Make sure there is no red text in the following command:

```
$ typer illixr.analysis.main run check [--no-modify] [--verbose]
# This will modify your code in place, but only non-semantic chagnes
# --no-in-place turns that off.
```

To run the code,
```
$ typer illixr.analysis.main run
```

## Guiding principles

- Avoid human-intervention. E.g. Avoid manually renaming or deleting directories.

- Write a library first, script second. The script should just invoke the library functions. The library functions shouldn't care about files or directories (e.g. operate on Numpy arrays not CSV files). The library functions should be stateless but can use `charmonium.cache` to sepeed them up.
  - This helps us make them reusable across different trials and scenarios.

- Use static types. Data processing is long enough that bugs might not cause faults until ten minutes in. Static typing helps us fail earlier.

- Don't commit data to this repo. I tried that once; Since git stores a whole branching history of eventually-obsolete data, it made cloning the repo take as long as draining an ocean.

[pyenv]: https://github.com/pyenv/pyenv/
