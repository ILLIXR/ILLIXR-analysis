"""Munges the data into human-readable outputs.

This should not take into account ILLIXR-specific information.
"""

from typing import Callable, List

from .types import Trial, Trials


def callgraph(trial: Trial) -> None:
    """Generate a visualization of the callgraph."""
    # digraph = trial.call_forest.get_static_callgraph()
    # total_time = 1
    # for edge in digraph.edges:
    #     parent_time = 1
    #     # sum(dynamic_frame.cpu_duration for dynamic_frame in edge["dynamic_frames"])
    #     child_time = sum(
    #         cast(int, dynamic_frame.cpu_duration)
    #         for dynamic_frame in edge["dynamic_frames"]
    #     )
    #     edge["penwidth"] = clip(child_time / parent_time * 10, 0.2, 10)
    #     edge["fixedsize"] = True
    #     edge["size"] = clip(child_time / total_time * 10, 0.2, 10)
    #     del edge["dynamic_frames"]


analyze_trials_fns: List[Callable[[Trials], None]] = []
analyze_trial_fns: List[Callable[[Trial], None]] = [callgraph]


def analyze_trials(trials: Trials) -> None:
    """Main entrypoint for inter-trial analysis.

    All inter-trial analyses should be started from here.

    """
    for analyze_trials_fn in analyze_trials_fns:
        analyze_trials_fn(trials)

    for analyze_trial_fn in analyze_trial_fns:
        for trial in trials.each:
            analyze_trial_fn(trial)
