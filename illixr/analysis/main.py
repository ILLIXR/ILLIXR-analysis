"""Handles invocation from the CLI.

This module should only call out to the highest-level of functions
defined in other modules.  """

import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Union

import typer

from .analyze_trials import analyze_trials
from .read_trials import read_trials
from .util import command_exists

app = typer.Typer()

# See https://clig.dev/ for guidelines
@app.command()
def main(data_dir: Path) -> None:
    """Runs every analysis on every trial."""
    trials = read_trials([data_dir])
    analyze_trials(trials)


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
        ],
        [
            "pylint",
            "--verbose" if verbose else "",
            "--output-format=colorized",
            "--jobs=0",
            package,
        ],
        (
            ["scc", "--by-file", "--wide", "--no-cocomo", source]
            if command_exists("scc")
            else []
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

        # typer.echo(repr(proc.stdout))
        typer.echo(proc.stderr, color=True, nl=False, file=sys.stderr)
        typer.echo(proc.stdout, color=True, nl=False, file=sys.stdout)
        # if not proc.stdout.endswith(b"\n\n"):
        typer.echo("\n")

    if all_success:
        typer.echo("Your code is excellent and ready to commit! \U0001F3C6")
    else:
        typer.echo("You're probably busy building an awesome feature.")
        typer.echo("Be sure to fix these problems before comitting. \U0001F44D")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
