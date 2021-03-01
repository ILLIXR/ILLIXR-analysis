"""Munges the data into human-readable outputs.

This should not take into account ILLIXR-specific information.
"""

import collections
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Mapping, Optional, Set, Tuple

import anytree  # type: ignore
import networkx as nx  # type: ignore
import numpy
import pygraphviz  # type: ignore

from .call_tree import DynamicFrame, StaticFrame
from .types import Trial, Trials
from .util import clip


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

    for tree in trial.call_trees.values():
        for static_frame in anytree.PreOrderIter(tree.root.static_frame):
            if static_frame.function_name in {"_p_one_iteration", "callback"}:
                dynamic_frames = tree.static_to_dynamic[static_frame]
                times = (
                    numpy.array(
                        [dynamic_frame.cpu_time for dynamic_frame in dynamic_frames]
                    )
                    / 1e3
                )
                plugin = get_plugin(static_frame)
                print(
                    f"{plugin}: {numpy.mean(times):.0f} +/- {numpy.std(times):.0f} (us), 75% < {numpy.percentile(times, 75):.0f}"
                )
    # if command_exists("feh"):
    #     subprocess.run(["feh", str(img_path)], check=True)


def data_flow_graph(trial: Trial) -> None:
    """Generates a visualization of dataflow over Switchboard between plugins."""

    class EdgeType(Enum):
        program = 1
        async_ = 2
        sync = 3

    dynamic_dfg = nx.DiGraph()

    # Add all program edges
    for tree in trial.call_trees.values():
        last_comm: Optional[DynamicFrame] = None
        for frame in anytree.PreOrderIter(tree.root):
            if frame.static_frame.function_name in {"put", "get", "callback"}:
                if last_comm is not None:
                    dynamic_dfg.add_edge(
                        last_comm, frame, topic_name=None, type=EdgeType.program
                    )
                last_comm = frame

    # Compute data_to_put
    data_to_put: Dict[Tuple[str, int], DynamicFrame] = {}
    # Maps a dataitem (topic_name, serial_no) to its put.
    for tree in trial.call_trees.values():
        for frame in anytree.PreOrderIter(tree.root):
            if frame.static_frame.function_name == "put":
                data_id = (frame.static_frame.topic_name, frame.serial_no)
                assert data_id not in data_to_put
                data_to_put[data_id] = frame

    # Add all async adges (put to get)
    for tree in trial.call_trees.values():
        for frame in anytree.PreOrderIter(tree.root):
            if frame.static_frame.function_name == "get":
                data_id = (frame.static_frame.topic_name, frame.serial_no)
                put = data_to_put.get(data_id, None)
                if put is not None:
                    dynamic_dfg.add_edge(
                        put,
                        frame,
                        topic_name=frame.static_frame.topic_name,
                        type=EdgeType.async_,
                    )
                else:
                    pass
                    # print(f"get: {data_id} not found")

    # Add all sync adges (put to callback)
    for tree in trial.call_trees.values():
        # Maps a topic to its callback counter
        # defaults to zero
        topic_to_callback_count: Dict[str, int] = collections.defaultdict(lambda: 0)
        for frame in anytree.PreOrderIter(tree.root):
            if frame.static_frame.function_name == "callback":
                assert len(topic_to_callback_count) < 2
                topic_name = get_topic(frame.static_frame)
                assert topic_name
                callback_count = topic_to_callback_count[topic_name]
                topic_to_callback_count[topic_name] += 1
                data_id = (topic_name, callback_count)
                put = data_to_put.get(data_id, None)
                if put is not None:
                    dynamic_dfg.add_edge(
                        put, frame, topic_name=topic_name, type=EdgeType.sync
                    )
                else:
                    print(f"cb: {data_id} not found")

    static_dfg = nx.DiGraph()
    for src, dst, edge_attrs in dynamic_dfg.edges(data=True):
        static_dfg.add_edge(
            src.static_frame,
            dst.static_frame,
            **edge_attrs,
        )

    plugin_to_nodes: Dict[str, List[StaticFrame]] = collections.defaultdict(list)
    static_dfg_graphviz = pygraphviz.AGraph(strict=True, directed=True)

    def frame_to_id(frame: StaticFrame) -> str:
        return str(id(frame))

    def frame_to_label(frame: StaticFrame, function_name: Optional[str] = None) -> str:
        if frame.function_name in {"get", "put"}:
            assert frame.parent
            return frame_to_label(frame.parent, frame.function_name)
        else:
            file_name = "/".join(frame.file_name.split("/")[-3:])
            function_name = (
                function_name if function_name is not None else frame.function_name
            )
            topic_name = get_topic(frame)
            return f"{get_plugin(frame)} {function_name} {topic_name}\\n{file_name}:{frame.line}"

    include_program = True
    # def frame_to_id(frame: StaticFrame) -> str:
    #     return str(get_plugin(frame))

    # def frame_to_label(frame: StaticFrame) -> str:
    #     return str(get_plugin(frame))

    # include_program = False
    for src, dst, edge_attrs in static_dfg.edges(data=True):
        if include_program or edge_attrs["type"] != EdgeType.program:
            static_dfg_graphviz.add_edge(
                frame_to_id(src),
                frame_to_id(dst),
                label=(edge_attrs["topic_name"] if edge_attrs["topic_name"] else ""),
                style={
                    EdgeType.program: "bold",
                    EdgeType.async_: "dashed",
                    EdgeType.sync: "solid",
                }[edge_attrs["type"]],
            )
            for node in [src, dst]:
                plugin = get_plugin(node)
                assert plugin
                plugin_to_nodes[plugin].append(node)
                static_dfg_graphviz.get_node(frame_to_id(node)).attr[
                    "label"
                ] = frame_to_label(node)

    # for plugin, nodes in plugin_to_nodes.items():
    #     static_dfg_graphviz.add_subgraph(map(frame_to_id, nodes), plugin, rank="same", rankdir="TB")

    src_plugin = "offline_imu_cam"
    dst_plugin = "timewarp_gl"
    path_static_to_dynamic: Mapping[
        Tuple[StaticFrame, ...], List[Tuple[DynamicFrame, ...]]
    ] = collections.defaultdict(list)

    def explore(
        dynamic_frame: DynamicFrame,
        static_path: Tuple[StaticFrame, ...],
        dynamic_path: Tuple[DynamicFrame, ...],
        tabu: Set[DynamicFrame],
    ) -> Iterator[Tuple[Tuple[StaticFrame, ...], Tuple[DynamicFrame, ...]]]:
        static_path += (dynamic_frame.static_frame,)
        dynamic_path += (dynamic_frame,)
        if (
            get_plugin(dynamic_frame.static_frame) == "timewarp_gl"
            and get_topic(dynamic_frame.static_frame) == "hologram_in"
            and dynamic_frame.static_frame.function_name == "put"
        ):
            yield (static_path, dynamic_path)
        for next_dynamic_frame in dynamic_dfg[dynamic_frame]:
            if next_dynamic_frame.static_frame not in tabu:
                yield from explore(
                    next_dynamic_frame,
                    static_path,
                    dynamic_path,
                    tabu | {next_dynamic_frame.static_frame},
                )

    for static_frame in static_dfg:
        if (
            static_frame.function_name == "put"
            and get_plugin(static_frame) == src_plugin
        ):
            for call_tree in trial.call_trees.values():
                for dynamic_frame in call_tree.static_to_dynamic[static_frame]:
                    for static_path, dynamic_path in explore(
                        dynamic_frame, (), (), {dynamic_frame.static_frame}
                    ):
                        path_static_to_dynamic[static_path].append(dynamic_path)

    path_static_to_dynamic = {
        static: dynamic
        for static, dynamic in path_static_to_dynamic.items()
        if len(dynamic) > 250
    }

    # TODO: compute how many puts are "used/ignored"

    for path, instances in path_static_to_dynamic.items():
        for src, dst in zip(path[:-1], path[1:]):
            static_dfg_graphviz.get_edge(
                frame_to_id(src),
                frame_to_id(dst),
            ).attr["color"] = "blue"
            static_dfg_graphviz.get_edge(frame_to_id(src), frame_to_id(dst),).attr[
                "label"
            ] += " " + str(len(instances))

    dot_path = Path(trial.output_dir / "dataflow.dot")
    img_path = trial.output_dir / "dataflow.png"
    static_dfg_graphviz.write(dot_path)
    static_dfg_graphviz.draw(img_path, prog="dot")

    # assert len(path_static_to_dynamic) == 6
    for static_path, dynamic_paths in path_static_to_dynamic.items():
        print(
            len(dynamic_paths),
            [
                f"{frame_to_label(frame)} {str(frame)} on {get_topic(frame)}"
                for frame in static_path
            ],
        )
        all_transits = (
            numpy.array(
                [
                    [node.wall_start for node in dynamic_path]
                    + [dynamic_path[-1].wall_stop]
                    for dynamic_path in dynamic_paths
                ]
            )
            / 1e6
        )

        assert (all_transits[1:] - all_transits[:-1] >= 0).all()

        assert (all_transits[:, 1:] - all_transits[:, :-1] >= 0).all()

        latency = all_transits - all_transits[:, 0][:, numpy.newaxis]

        for i in range(len(dynamic_paths[0])):
            m = latency[:, i].mean()
            s = latency[:, i].std()
            print(f"{m:.0f} +/- {s:.0f} (ms)")
            print(frame_to_label(static_path[i]).replace("\\n", "\n"))
        end_latency = latency[:, -1]
        print(
            f"Total latency = {end_latency.mean():.0f} +/- {end_latency.std():.0f}, data[75%] = {numpy.percentile(end_latency, 75):.0f} (ms)"
        )
        duration = all_transits[0, -1] - all_transits[0, 0]
        period = all_transits[1:, 0] - all_transits[:-1, 0]
        print(
            f"Period = {period.mean():.0f} +/- {period.std():.0f}, data[75%] = {numpy.percentile(period, 75):.0f} (ms)"
        )
        rt = all_transits[1:, -1] - all_transits[:-1, 0]
        print(
            f"RT = {rt.mean():.0f} +/- {rt.std():.0f}, data[75%] = {numpy.percentile(rt, 75):.0f} (ms)"
        )
        print()
        # print(numpy.mean(latency, axis=0))
        # print(numpy.std(latency, axis=0))


def get_plugin(frame: StaticFrame) -> Optional[str]:
    """Returns the name of the plugin responsible for calling this static frame (if known). Else None."""
    if not frame.plugin and frame.parent is not None:
        return get_plugin(frame.parent)
    else:
        return frame.plugin


def get_topic(frame: StaticFrame) -> Optional[str]:
    """Returns the topic of the plugin responsible for calling this static frame (if known). Else None."""
    if not frame.topic_name and frame.parent is not None:
        return get_topic(frame.parent)
    else:
        return frame.topic_name


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
