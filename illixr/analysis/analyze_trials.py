"""Munges the data into human-readable outputs.

This should not take into account ILLIXR-specific information.
"""

import collections
import subprocess
from pathlib import Path
from typing import Callable, Dict, List

import anytree  # type: ignore
import numpy  # type: ignore
import pygraphviz  # type: ignore

from .call_tree import StaticFrame
from .types import Trial, Trials
from .util import clip, command_exists


def callgraph(trial: Trial) -> None:
    """Generate a visualization of the callgraph."""
    total_time = sum([tree.root.cpu_time for tree in trial.call_trees.values()])
    graphviz = pygraphviz.AGraph(strict=True, directed=True)
    for tree in trial.call_trees.values():
        static_frame_time: Dict[StaticFrame, float] = collections.defaultdict(lambda: 0)
        for dynamic_frame in anytree.PreOrderIter(tree.root):
            static_frame_time[dynamic_frame.static_frame] += dynamic_frame.cpu_time

        # for static_frame in anytree.PreOrderIter(tree.root.static_frame):
        #     if not static_frame.is_leaf:
        #         static_frame_other = StaticFrame(
        #             dict(
        #                 function_name="<other>",
        #                 plugin_id=0,
        #                 topic_name="",
        #             ),
        #             parent=static_frame,
        #         )
        #         static_frame_time[static_frame_other] = static_frame_time[
        #             static_frame
        #         ] - sum(static_frame_time[child] for child in static_frame.children)

        for static_frame in anytree.PreOrderIter(tree.root.static_frame):
            node_weight = static_frame_time[static_frame] / total_time
            if static_frame._function_name not in {"get", "put"}:
                graphviz.add_node(
                    id(static_frame),
                    label=str(static_frame),
                    # width=clip(node_weight * 7, 0.1, 7),
                    # fixedsize=True,
                )
                if static_frame.parent is not None:
                    edge_weight = node_weight
                    graphviz.add_edge(
                        id(static_frame.parent),
                        id(static_frame),
                        penwidth=clip(numpy.sqrt(edge_weight) * 50, 0.1, 10),
                    )

    dot_path = Path(trial.output_dir / "callgraph.dot")
    img_path = trial.output_dir / "callgraph.png"
    graphviz.write(dot_path)
    graphviz.draw(img_path, prog="dot")
    if command_exists("feh"):
        subprocess.run(["feh", str(img_path)], check=True)


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
