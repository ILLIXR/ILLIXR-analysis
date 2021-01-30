from pathlib import Path

import typer

from .analyze_trials import analyze_trials
from .read_trials import read_trials

app = typer.Typer()


# See https://clig.dev/ for guidelines
@app.command()
def main(data_dir: Path) -> None:
    trials = read_trials([data_dir])
    analyze_trials(trials)


if __name__ == "__main__":
    app()
