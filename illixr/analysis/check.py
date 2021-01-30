import datetime
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Union

import typer

app = typer.Typer()

source = Path(__file__).relative_to(Path().resolve()).parent.parent
package = "illixr"


@app.command()
def check(in_place: bool = True, verbose: bool = False) -> None:

    commands: List[List[Union[Path, str]]] = [
        ["autoflake", "--recursive", "--in-place" if in_place else "--check", source],
        ["isort", *([] if in_place else ["--check"]), "--color", source],
        [
            "black",
            "--color",
            "--target-version",
            "py38",
            *([] if in_place else ["--check"]),
            "--verbose" if verbose else "--quiet",
            source,
        ],
        [
            "dmypy",
            "run",
            "--",
            *(["--verbose"] if verbose else []),
            "--namespace-packages",
            "--color-output",
            "--package",
            package,
        ],
        # ["pylint", *("--verbose" if verbose else []), "--output-format=colorized", source],
    ]

    stdout_bytes = os.fdopen(sys.stdout.fileno(), "wb")
    for command in commands:
        start = datetime.datetime.now()
        proc = subprocess.run(
            command,
            capture_output=True,
        )
        stop = datetime.datetime.now()
        success = 0 == (
            proc.returncode
            if command[0] == "pylint"
            else proc.returncode & ~(2 + 4 + 8 + 16)
        )
        command_str = "$ " + shlex.join(map(str, command))
        if proc.returncode == 0:
            typer.echo(
                typer.style(
                    command_str,
                    fg=typer.colors.GREEN,
                )
            )
        else:
            typer.echo(
                typer.style(
                    command_str,
                    fg=typer.colors.RED,
                )
            )
            # https://stackoverflow.com/a/16835064/1078199
        stdout_bytes.write(proc.stdout)
        stdout_bytes.write(proc.stderr)
        stdout_bytes.flush()


if __name__ == "__main__":
    app()
