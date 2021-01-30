"""Munges the data into human-readable outputs."""

from .types import Trial, Trials


def analyze_trials(trials: Trials) -> None:
    """Main entrypoint for inter-trial analysis.

    All inter-trial analyses should be started from here.

    """
    for trial in trials.each:
        analyze_trial(trial)


def analyze_trial(trial: Trial) -> None:
    """Main entrypoint for per-trial analyses.

    All per-trial analyses should be started from here.

    """
    ...
