"""Handles invocation from the CLI.

This module should only call out to the highest-level of functions
defined in other modules.  """

import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Union

import typer

from illixr.analysis.util import command_exists

app = typer.Typer()

# See https://clig.dev/ for guidelines
@app.command()
def analyze_trials(
    dir_of_metrics_dirs: Path,
    chunk_size: int = typer.Option(10, "--chunk-size"),
    verify: bool = typer.Option(
        False, "--verify", help="Preform extra checks on the data"
    ),
    extra_metrics: List[Path] = typer.Option([], "--also"),
) -> None:
    """Runs every analysis on every trial."""

    import dask.multiprocessing

    dask.config.set(
        scheduler="processes"
    )  # overwrite default with multiprocessing scheduler

    # This is for the distributed scheduler
    # dask.config.set({"distributed.worker.daemon": False})
    # client = dask.distributed.Client(
    #     address=dask.distributed.LocalCluster(
    #         n_workers=min(multiprocessing.cpu_count(), 20),
    #     ),
    # )
    # print(client.dashboard_link)
    # webbrowser.open(client.dashboard_link)

    candidates = [
        path
        for path in list(dir_of_metrics_dirs.iterdir()) + extra_metrics
        if path.is_dir() and (path / "log").exists() and not (path / "source").exists()
        # path / log means data is there
    ]

    # from .analyze_trials import just_read
    # just_read(candidates)

    from illixr.analysis.analyze_trials2 import analyze_trials
    trials = analyze_trials(candidates, dir_of_metrics_dirs, chunk_size)

    # For the distributed scheduler:
    # client.shutdown()

@app.command()
def analyze_trials3(
    dir_of_metrics_dirs: Path,
    chunk_size: int = typer.Option(10, "--chunk-size"),
    verify: bool = typer.Option(
        False, "--verify", help="Preform extra checks on the data"
    ),
    extra_metrics: List[Path] = typer.Option([], "--also"),
) -> None:
    """Runs analyze_trials3 on every trial."""
    candidates = [
        path
        for path in list(dir_of_metrics_dirs.iterdir()) + extra_metrics
        if path.is_dir() and (path / "source").exists()
        # path / source means it has been processed already
    ]

    from .analyze_trials3 import analyze_trials3
    analyze_trials3(candidates, dir_of_metrics_dirs, chunk_size)

@app.command()
def just_read(
    dir_of_metrics_dirs: Path,
) -> None:
    """Runs every analysis on every trial."""

    import dask.multiprocessing

    dask.config.set(
        scheduler="processes"
    )  # overwrite default with multiprocessing scheduler

    import logging
    memoize_logger = logging.getLogger("charmonium.cache.memoize")
    memoize_logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("memoize.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    memoize_logger.addHandler(fh)

    determ_hash_logger = logging.getLogger("charmonium.cache.determ_hash")
    determ_hash_logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("determ_hash.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    determ_hash_logger.addHandler(fh)

    candidates = [
        path
        for path in list(dir_of_metrics_dirs.iterdir())
        if path.is_dir() and (path / "log").exists()
    ]

    from .analyze_trials import just_read
    just_read(candidates)

@app.command()
def check(
    modify: bool = typer.Option(default=True, help="modify the code in place?"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Runs linters, checkers, and formatters on the source code."""

    source = Path(__file__).relative_to(Path().resolve()).parent.parent

    package = "illixr"

    # https://github.com/jongracecox/pylint-exit#return-codes
    pylint_codes = {
        "fatal": 1,
        "error": 2,
        "warning": 4,
        "refactor": 8,
        "convention": 16,
        "usage_error": 32,
    }
    pylint_codes["acceptable"] = (
        pylint_codes["warning"] | pylint_codes["refactor"] | pylint_codes["convention"]
    )

    commands: List[List[Union[Path, str]]] = [
        [
            "autoflake",
            "--recursive",
            "--in-place" if modify else "--check",
            "--recursive",
            "--expand-star-imports",
            "--remove-all-unused-imports",
            "--ignore-init-module-imports",
            "--verbose" if verbose else "",
            source,
        ],
        [
            "isort",
            "" if modify else "--check",
            "--color",
            "--verbose" if verbose else "",
            # Make isort compatible with Black
            # See https://copdips.com/2020/04/making-isort-compatible-with-black.html
            "--multi-line",
            "3",
            "--trailing-comma",
            "--line-width",
            "88",
            source,
        ],
        [
            "black",
            "--color",
            "--target-version",
            "py38",
            "" if modify else "--check",
            "--verbose" if verbose else "--quiet",
            source,
            # Black is not configurable by design.
            # It's opinionated.
        ],
        [
            "dmypy",
            "run",
            "--",
            "--verbose" if verbose else "",
            "--namespace-packages",
            "--color-output",
            "--package",
            package,
            # mypy can be configured on mypy.ini
        ],
        [
            "pylint",
            "--verbose" if verbose else "",
            "--output-format=colorized",
            "--jobs=0",
            package,
            # pylint can be configured in .pylintrc
        ],
        (
            ["scc", "--by-file", "--wide", "--no-cocomo", source]
            if command_exists("scc")
            else []
            # https://github.com/boyter/scc
            # Install through go's package manager or snap.
        ),
    ]

    all_success = True
    for command in filter(bool, commands):
        command = list(filter(bool, command))
        typer.secho(f"$ {shlex.join(map(str, command))}", bold=True)
        proc = subprocess.run(
            command,
            capture_output=True,
            check=False,
        )
        success = (
            proc.returncode
            if command[0] != "pylint"
            else proc.returncode & ~pylint_codes["acceptable"]
        ) == 0
        all_success = all_success and success
        typer.echo("\U00002705" if success else "\U0000274C")

        typer.echo(proc.stderr, color=True, nl=False, file=sys.stderr)
        typer.echo(proc.stdout, color=True, nl=False, file=sys.stdout)
        # if not proc.stdout.endswith(b"\n\n"):
        typer.echo()

    if all_success:
        typer.secho(
            "Your code is impeccable and ready to commit!", fg=typer.colors.GREEN
        )
    else:
        typer.secho(
            "You're probably busy building an awesome feature.\n"
            "Be sure to fix these problems before comitting.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
