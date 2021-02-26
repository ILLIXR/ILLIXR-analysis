"""Munges the data into human-readable outputs.

This should not take into account ILLIXR-specific information.
"""

import collections
import subprocess
from pathlib import Path
from typing import Callable, Dict, List

import anytree  # type: ignore
import networkx as nx  # type: ignore
import numpy
import pygraphviz  # type: ignore

from .call_tree import StaticFrame
from .types import Trial, Trials
from .util import clip, command_exists


def callgraph(trial: Trial) -> None:
    """Generate a visualization of the callgraph."""
    total_time = sum([tree.root.cpu_time for tree in trial.call_trees.values()])
    cpu_timer_calls = sum([tree.calls for tree in trial.call_trees.values()])
    cpu_timer_overhead = cpu_timer_calls * 400
    print(cpu_timer_overhead, total_time, cpu_timer_overhead / total_time)
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
            if static_frame.function_name not in {"get", "put"}:
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
                        penwidth=clip(numpy.sqrt(edge_weight) * 20, 0.1, 15),
                    )

    dot_path = Path(trial.output_dir / "callgraph.dot")
    img_path = trial.output_dir / "callgraph.png"
    graphviz.write(dot_path)
    graphviz.draw(img_path, prog="dot")
    if command_exists("feh"):
        subprocess.run(["feh", str(img_path)], check=True)


def data_flow_graph(trial: Trial) -> None:
    """Generates a visualization of dataflow over Switchboard between plugins."""
    dynamic_dfg = nx.DiGraph()
    sender_to_receiver = {}
    for tree in trial.call_trees.values():
        for dynamic_frame in anytree.PreOrderIter(tree.root):
            if dynamic_frame.static_frame.function_name == "put":
                dynamic_dfg.add_node(dynamic_frame)
                sender_to_receiver[
                    (dynamic_frame.topic_name, dynamic_frame.serial_no)
                ] = dynamic_frame
    for tree in trial.call_trees.values():
        for dynamic_frame in anytree.PreOrderIter(tree.root):
            if dynamic_frame.static_frame.function_name == "get":
                dynamic_dfg.add_node(dynamic_frame)
                if (
                    dynamic_frame.topic_name,
                    dynamic_frame.serial_no,
                ) in sender_to_receiver:
                    sender = sender_to_receiver[
                        (dynamic_frame.topic_name, dynamic_frame.serial_no)
                    ]
                    dynamic_dfg.add_edge(sender, dynamic_frame)
    static_dfg = nx.DiGraph()
    for source, dest in dynamic_dfg.edges:
        static_dfg.add_edge(
            get_plugin_id(source.static_frame), get_plugin_id(dest.static_frame)
        )
    dot_path = Path(trial.output_dir / "dataflow.dot")
    img_path = trial.output_dir / "dataflow.png"
    static_dfg_graphviz = nx.nx_agraph.to_agraph(static_dfg)
    static_dfg_graphviz.write(dot_path)
    static_dfg_graphviz.draw(img_path, prog="dot")
    if command_exists("feh"):
        subprocess.run(["feh", str(img_path)], check=True)


def get_plugin_id(frame: StaticFrame) -> int:
    """Returns the ID of the plugin responsible for calling this static frame (if known). Else 0."""
    if frame.plugin_id == 0 and frame.parent is not None:
        return get_plugin_id(frame.parent)
    else:
        return frame.plugin_id


analyze_trials_fns: List[Callable[[Trials], None]] = []
analyze_trial_fns: List[Callable[[Trial], None]] = [callgraph, data_flow_graph]


def analyze_trials(trials: Trials) -> None:
    """Main entrypoint for inter-trial analysis.

    All inter-trial analyses should be started from here.

    """
    for analyze_trials_fn in analyze_trials_fns:
        analyze_trials_fn(trials)

    for analyze_trial_fn in analyze_trial_fns:
        for trial in trials.each:
            analyze_trial_fn(trial)
