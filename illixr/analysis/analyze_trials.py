"""Munges the data into human-readable outputs.

This should not take into account ILLIXR-specific information.
"""

import collections
import random
from enum import Enum
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
)
import warnings

import anytree  # type: ignore
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import numpy
import pygraphviz  # type: ignore

from .call_tree import DynamicFrame, StaticFrame
from .types import Trial, Trials
from .util import clip


def callgraph(trial: Trial) -> None:
    """Generate a visualization of the callgraph."""
    print(len(trial.call_trees.values()))
    total_time = sum([tree.root.cpu_time for tree in trial.call_trees.values()])
    cpu_timer_calls = sum([tree.calls for tree in trial.call_trees.values()])
    cpu_timer_overhead = cpu_timer_calls * 400
    print(cpu_timer_overhead, total_time, cpu_timer_overhead / total_time if total_time != 0 else "NaN")
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
            if True or static_frame.function_name not in {"get", "put"}:
                topic = get_topic(static_frame)
                graphviz.add_node(
                    id(static_frame),
                    label=str(static_frame) + f"\n{topic}" if topic else "",
                )
                if static_frame.parent is not None:
                    edge_weight = node_weight
                    graphviz.add_edge(
                        id(static_frame.parent),
                        id(static_frame),
                        penwidth=clip((edge_weight) * 20, 0.1, 45),
                    )

    dot_path = Path(trial.output_dir / "callgraph.dot")
    img_path = trial.output_dir / "callgraph.png"
    graphviz.write(dot_path)
    graphviz.draw(img_path, prog="dot")

    for tree in trial.call_trees.values():
        for static_frame in anytree.PreOrderIter(tree.root.static_frame):
            if static_frame.function_name in {
                "_p_one_iteration",
                "callback",
                "cam",
                "IMU",
                "load_camera_data",
            }:
                dynamic_frames = tree.static_to_dynamic[static_frame]
                times = (
                    numpy.array(
                        [dynamic_frame.cpu_time for dynamic_frame in dynamic_frames]
                    )
                    / 1e3
                )
                plugin = get_plugin(static_frame)
                print(
                    f"{plugin} ({tree.thread_id}) {static_frame.function_name} (us): {numpy.mean(times):,.0f} +/- {numpy.std(times):,.0f}, len(data) = {len(times)}, data[75%] < {numpy.percentile(times, 75):,.0f}, data[95%] < {numpy.percentile(times, 95):,.0f}"
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
            if frame.static_frame.function_name in {"put", "get", "callback", "entry", "exit"}:
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
            if frame.static_frame.function_name == "put" and frame.static_frame.topic_name != "1_completion":
                data_id = (frame.static_frame.topic_name, frame.serial_no)
                assert data_id not in data_to_put, f"{data_id}"
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
                elif frame.serial_no != -1:
                    warnings.warn(f"get: {data_id} not found", UserWarning)

    # Add all sync adges (put to callback)
    for tree in trial.call_trees.values():
        # Maps a topic to its callback counter
        # defaults to zero
        topic_to_callback_count: Dict[str, int] = collections.defaultdict(lambda: 0)
        for frame in anytree.PreOrderIter(tree.root):
            if frame.static_frame.function_name == "callback":
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
                    warnings.warn(f"cb: {data_id} from {get_plugin(frame.static_frame)} not found", UserWarning)

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

    def frame_to_label(frame: StaticFrame) -> str:
        top = f"{get_plugin(frame)} {frame.function_name} {get_topic(frame)}"
        stack = "\\n".join(
            f"{'/'.join(frame.file_name.split('/')[-2:])}:{frame.line}:{frame.function_name}"
            for frame in frame.path[2:]
        )
        return f"{top}\n{stack}"

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
                # assert plugin
                plugin_to_nodes[plugin].append(node)
                static_dfg_graphviz.get_node(frame_to_id(node)).attr[
                    "label"
                ] = frame_to_label(node)

    # for plugin, nodes in plugin_to_nodes.items():
    #     static_dfg_graphviz.add_subgraph(map(frame_to_id, nodes), plugin, rank="same", rankdir="TB")
        

    path_static_to_dynamic: Mapping[
        Tuple[StaticFrame, ...], List[Tuple[DynamicFrame, ...]]
    ] = collections.defaultdict(list)

    def is_dst(static_frame: StaticFrame) -> bool:
        return static_frame.function_name == "exit"

    def is_src(static_frame: StaticFrame) -> bool:
        return static_frame.function_name == "entry"

    dynamic_dfg_rev = dynamic_dfg.reverse()

    srcs = set()
    def explore(
        dynamic_frame: DynamicFrame,
        static_path: Tuple[StaticFrame, ...],
        dynamic_path: Tuple[DynamicFrame, ...],
        tabu: Set[DynamicFrame],
        iter: int,
    ) -> Iterator[Tuple[Tuple[StaticFrame, ...], Tuple[DynamicFrame, ...]]]:
        static_path += (dynamic_frame.static_frame,)
        dynamic_path += (dynamic_frame,)
        assert iter < 30
        if is_src(dynamic_frame.static_frame):
            srcs.add(get_plugin(dynamic_frame.static_frame))
            yield (static_path, dynamic_path)
        for next_dynamic_frame in dynamic_dfg_rev[dynamic_frame]:
            if next_dynamic_frame.static_frame not in tabu:
                yield from explore(
                    next_dynamic_frame,
                    static_path,
                    dynamic_path,
                    tabu | {next_dynamic_frame.static_frame},
                    iter + 1,
                )

    for static_dst in static_dfg:
        if is_dst(static_dst):
            for call_tree in trial.call_trees.values():
                for dynamic_dst in call_tree.static_to_dynamic[static_dst]:
                    for static_path, dynamic_path in explore(
                        dynamic_dst, (), (), {dynamic_dst.static_frame}, 0
                    ):
                        path_static_to_dynamic[static_path[::-1]].append(
                            dynamic_path[::-1]
                        )

    for static, dynamic in path_static_to_dynamic.items():
        for src, dst in zip(static[:-1], static[1:]):
            static_dfg_graphviz.get_edge(frame_to_id(src), frame_to_id(dst),).attr[
                "label"
            ] += " " + str(len(dynamic))

    def get_input_times(path: Iterable[DynamicFrame]) -> Iterable[int]:
        return (
            node.wall_start for node in path if node.static_frame.function_name == "put"
        )

    for static_path, dynamic_paths in path_static_to_dynamic.items():
        if "5" in set(map(get_plugin, static_path)):
            print(len(dynamic_paths), tuple(map(get_plugin, static_path)))
    print("look")

    path_static_to_dynamic = {
        static: dynamic
        for static, dynamic in path_static_to_dynamic.items()
        if len(dynamic) > 250
    }


    path_static_to_dynamic_freshest: Mapping[
        Tuple[StaticFrame, ...], List[Tuple[DynamicFrame, ...]]
    ] = collections.defaultdict(list)
    for static_path, dynamic_paths in path_static_to_dynamic.items():
        freshest_inputs_time: Optional[Tuple[int, ...]] = None
        for dynamic_path in dynamic_paths:
            inputs_time = tuple(get_input_times(dynamic_path))
            if freshest_inputs_time is None or all(
                x > y for x, y in zip(inputs_time, freshest_inputs_time)
            ):
                path_static_to_dynamic_freshest[static_path].append(dynamic_path)
                freshest_inputs_time = inputs_time

    path_static_to_dynamic = path_static_to_dynamic_freshest

    # TODO: compute how many puts are "used/ignored"

    for path, instances in path_static_to_dynamic.items():
        for src, dst in zip(path[:-1], path[1:]):
            static_dfg_graphviz.get_edge(
                frame_to_id(src),
                frame_to_id(dst),
            ).attr["color"] = "blue"

    dot_path = Path(trial.output_dir / "dataflow.dot")
    img_path = trial.output_dir / "dataflow.png"
    static_dfg_graphviz.write(dot_path)
    static_dfg_graphviz.draw(img_path, prog="dot")

    # assert len(path_static_to_dynamic) == 6
    for static_path, dynamic_paths in path_static_to_dynamic.items():
        if static_path[-1].topic_name == "vsync":
            continue
        print()
        print(
            len(dynamic_paths),
            " -> ".join(f"{get_plugin(frame)}" for frame in static_path),
        )
        all_transits = (
            numpy.array(
                list(
                    [node.wall_start for node in dynamic_path]
                    + [dynamic_path[-1].wall_stop]
                    for dynamic_path in dynamic_paths
                )
            )
            / 1e6
        )

        import pandas as pd  # type: ignore

        df = pd.DataFrame(all_transits)
        df.columns = [get_plugin(node.static_frame) for node in dynamic_paths[0]] + [
            "end"
        ]
        df.to_csv("all_transits.csv")

        assert (all_transits[1:] - all_transits[:-1] < 0).sum() < len(
            all_transits
        ) * 0.01, "row monotonicity is violated for"

        assert (all_transits[:, 1:] - all_transits[:, :-1] < 0).sum() < len(
            all_transits
        ) * 0.01, "column monotonicity is violated"

        latency = all_transits - all_transits[:, 0][:, numpy.newaxis]

        # data_flow_bar_chart(static_path,dynamic_path, latency)

        def label2(frame: StaticFrame) -> str:
            return f"{get_plugin(frame)}-{frame.function_name}-{get_topic(frame)}"

        for i in range(len(dynamic_paths[0]) - 1):
            l = (latency[:, i+1] - latency[:, i])
            print(f"{label2(dynamic_paths[0][i].static_frame)} -> {label2(dynamic_paths[0][i+1].static_frame)}: {l.mean():,.1f} +/- {l.std():,.1f} (ms)")

        end_latency = latency[:, -1]
        print(
            f"Total latency (ms): {end_latency.mean():,.1f} +/- {end_latency.std():,.1f}, data[25%] = {numpy.percentile(end_latency, 25):,.1f}, data[50%] = {numpy.percentile(end_latency, 50):,.1f}, data[75%] = {numpy.percentile(end_latency, 75):,.1f}"
        )
        duration = all_transits[0, -1] - all_transits[0, 0]
        period = all_transits[1:, -1] - all_transits[:-1, -1]
        print(
            f"Period (ms): {period.mean():,.1f} +/- {period.std():,.1f}, min(data) = {period.min()}, data[50%] = {numpy.percentile(period, 50):,.1f}, data[75%] = {numpy.percentile(period, 75):,.1f}, data[95%] = {numpy.percentile(period, 95):,.1f}, data[99%] = {numpy.percentile(period, 99):,.1f}, max = {period.max()}"
        )
        rt = all_transits[1:, -1] - all_transits[:-1, 0]
        print(
            f"RT (ms): {rt.mean():,.1f} +/- {rt.std():,.1f}, min(data) = {rt.min()}, data[50%] = {numpy.percentile(rt, 50):,.1f}, data[75%] = {numpy.percentile(rt, 75):,.1f}, data[95%] = {numpy.percentile(rt, 95):,.1f}, data[99%] = {numpy.percentile(rt, 99):,.1f}, max = {rt.max()}"
        )
        # print(numpy.mean(latency, axis=0))
        # print(numpy.std(latency, axis=0))


def data_flow_bar_chart(static_path, dynamic_paths, latencies) -> None:
    rowNum = -1.5
    coIndex = 0
    fig, ax = plt.subplots()
    plt.title("Data Flow")
    plt.xlabel("Wall Time")
    plt.ylabel("iteration")
    colors = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"]
    ax.grid(True)
    for y, row in enumerate(latencies):
        for i in range(len(row) - 1):
            plt.broken_barh(
                [(row[i], row[i + 1] - row[i])],
                (y, 1),
                facecolors=colors[i % len(colors)],
            )
    random_num = random.randint(1, 1000000000)
    fig.savefig(f"dataflow_time_{random_num}.png")
    plt.close(fig)


def get_plugin(frame: StaticFrame) -> Optional[str]:
    """Returns the name of the plugin responsible for calling this static frame (if known). Else None."""
    if not frame.plugin and frame.parent is not None:
        return get_plugin(frame.parent)
    else:
        return frame.plugin


def get_topic(frame: StaticFrame) -> Optional[str]:
    """Returns the topic of the plugin responsible for calling this static frame (if known). Else None."""
    if frame.function_name == "callback" and frame.topic_name is None:
        return get_topic(frame.parent.parent)
    else:
        if frame.function_name in {"get", "put"}:
            assert frame.topic_name is not None
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
