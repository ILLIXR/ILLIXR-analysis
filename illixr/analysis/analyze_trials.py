from __future__ import annotations

"""Munges the data into human-readable outputs.

This should not take into account ILLIXR-specific information.
"""

import collections
import random
import contextlib
from enum import Enum
from pathlib import Path
import shutil
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Any,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)
import warnings
import itertools
import io
import gc

import anytree  # type: ignore
import yaml
import networkx as nx  # type: ignore
import numpy as np
import pygraphviz  # type: ignore
import pandas as pd  # type: ignore
from frozendict import frozendict
import charmonium.time_block as ch_time_block
from charmonium.cache import memoize, MemoizedGroup, Memoized
from tqdm import tqdm
import dask.bag
import dask
import multiprocessing

from .call_tree import DynamicFrame, StaticFrame, CallTree
from .util import clip, timeseries, histogram, write_dir, dict_concat, right_pad, summary_stats, chunker, second, omit, undefault_dict, capture_file

use_parallel = True

group = MemoizedGroup(size="40GiB", fine_grain_persistence=True)

def conditions2label(conditions: Mapping[str, Any], rev: bool = False, cpu: bool = True) -> str:
    return "".join([
        conditions["scheduler"],
        f"-{conditions['swap']}" if "swap" in conditions and conditions["scheduler"] == "dynamic" else "",
        f" (rev {conditions['hash']:08x})" if rev else "",
        f" {conditions['cpus']}x{conditions['cpu_freq']:.1f}GHz" if cpu else "",
    ])

def dict_delayed(*keys):
    def decorator(func):
        def inner_func(*args, **kwargs):
            # delayed_key = func(*args, **kwargs)
            delayed_key = dask.delayed(func)(*args, **kwargs)
            if len(keys) == 0:
                return {func.__name__: delayed_key}
            elif len(keys) == 1:
                return {keys[0]: delayed_key}
            else:
                raise ValueError
        inner_func.__name__ = func.__name__
        return inner_func
    return decorator

@memoize(verbose=False, group=group)
def gen_static_to_dynamic(call_trees: Mapping[int, CallTree]) -> Mapping[StaticFrame, List[DynamicFrame]]:
    static_to_dynamic: Mapping[StaticFrame, List[DynamicFrame]] = collections.defaultdict(list)
    for tree in call_trees.values():
        for static_frame, dynamic_frames in tree.static_to_dynamic.items():
            static_to_dynamic[static_frame].extend(dynamic_frames)
    return dict(static_to_dynamic)

def gen_callgraph_plot(
        output_dir: Path,
        call_trees: Mapping[int, CallTree],
        static_to_dynamic: Mapping[StaticFrame, List[DynamicFrame]],
) -> None:
    """Generate a visualization of the callgraph."""
    total_time = sum([tree.root.cpu_time for tree in call_trees.values()])
    cpu_timer_calls = sum([tree.calls for tree in call_trees.values()])
    cpu_timer_overhead = cpu_timer_calls * 400
    # cpu overhead estimate: {cpu_timer_overhead / 1e6:.1f}ms, total_time: {total_time / 1e6:.1f}ms, percent error: {cpu_timer_overhead / total_time * 100 if total_time != 0 else 0:.1f}%
    graphviz = pygraphviz.AGraph(strict=True, directed=True)

    for static_frame, dynamic_frames in static_to_dynamic.items():
        static_frame_time = sum(dynamic_frame.cpu_time for dynamic_frame in dynamic_frames)
        node_weight = static_frame_time / total_time
        if True or static_frame.function_name not in {"get", "put"}:
            graphviz.add_node(
                id(static_frame),
                label=static_frame.plugin_function_topic(),
            )
            if static_frame.parent is not None:
                edge_weight = node_weight
                graphviz.add_edge(
                    id(static_frame.parent),
                    id(static_frame),
                    penwidth=clip((edge_weight) * 20, 0.1, 45),
                )

    dot_path = Path(output_dir / "callgraph.dot")
    img_path = output_dir / "callgraph.png"
    graphviz.write(dot_path)
    graphviz.draw(img_path, prog="dot")

def gen_compute_times(
        output_dir: Path,
        call_trees: Mapping[int, CallTree],
        static_to_dynamic: Mapping[StaticFrame, List[DynamicFrame]],
        conditions: Mapping[str, Any],
) -> Mapping[StaticFrame, Mapping[str, Any]]:
    top_level_fn_names = {
        "callback",
        "cam",
        "IMU",
        "load_camera_data",
        "_p_one_iteration",
    }
    compute_times = {}
    for tree in call_trees.values():
        for static_frame in anytree.PreOrderIter(tree.root.static_frame):
            fn_name_good = static_frame.function_name in top_level_fn_names or static_frame.function_name.startswith("_")
            plugin_name_good = static_frame.plugin != "1"
            if fn_name_good and plugin_name_good:
                dynamic_frames = tree.static_to_dynamic[static_frame]
                cpu_times = np.array([dynamic_frame.cpu_time for dynamic_frame in dynamic_frames])
                wall_times = np.array([dynamic_frame.wall_time for dynamic_frame in dynamic_frames])
                ts = np.array([dynamic_frame.wall_start for dynamic_frame in dynamic_frames])
                compute_times[static_frame] = {
                    "ts": ts,
                    "cpu_time": cpu_times,
                    "wall_time": wall_times,
                    "thread_id": tree.thread_id,
                }
    return compute_times

@memoize(group=group, verbose=False)
def gen_compute_times_plot(
        call_trees: Mapping[int, CallTree],
        compute_times: Mapping[StaticFame, Mapping[str, Any]],
        output_dir: Path,
) -> Mapping[str, Any]:
    total_time = sum([tree.root.cpu_time for tree in call_trees.values()])
    cpu_timer_calls = sum([tree.calls for tree in call_trees.values()])
    cpu_timer_overhead = cpu_timer_calls * 400
    def summarize_compute_times(frame_data: Tuple[StaticFrame, Mapping[str, Any]]) -> Tuple[str, str]:
        frame, data = frame_data
        return (
            frame.plugin,
            " ".join([
                right_pad(frame.plugin_function(' '), 20),
                "tid:",
                right_pad(str(data["thread_id"]), 8),
                "cpu_time:  ",
                right_pad(summary_stats(data["cpu_time" ] / 1e6, digits=2), 95),
                "wall_time: ",
                right_pad(summary_stats(data["wall_time"] / 1e6, digits=2) , 95),
            ]),
        )

    compute_dir = output_dir / "compute_times"
    write_dir({
        output_dir / "compute_times": {
            "summary.txt": "\n".join([
                f"threads: {len(call_trees)}",
                f"cpu overhead estimate: {cpu_timer_overhead / 1e6:.1f}ms, total_time: {total_time / 1e6:.1f}ms, percent error: {cpu_timer_overhead / total_time * 100 if total_time != 0 else 0:.1f}%",
                *map(second, sorted(map(summarize_compute_times, compute_times.items()))),
            ]),
            **{
                f"{frame.plugin_function(' ')}": {
                    "hist.png": histogram(
                        ys=data["cpu_time"] / 1e6,
                        xlabel=f"CPU Time (ms)",
                        title=f"Compute Time of {frame.plugin_function(' ')}"
                    ),
                    "ts.png": timeseries(
                        ts=data["ts"],
                        ys=data["cpu_time"] / 1e6,
                        ylabel=f"CPU Time (ms)",
                        title=f"Compute Time of {frame.plugin_function(' ')}"
                    ),
                }
                for frame, data in compute_times.items()
            },
        },
    })

class EdgeType(Enum):
    program = 1
    async_ = 2
    sync = 3

@ch_time_block.decor()
def gen_dynamic_dfg(
        call_trees: Mapping[int, CallTree],
) -> Mapping[str, Any]:
    dynamic_dfg = nx.DiGraph()

    # Add all program edges
    for tree in call_trees.values():
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
    for tree in call_trees.values():
        for frame in anytree.PreOrderIter(tree.root):
            if frame.static_frame.function_name == "put" and frame.static_frame.topic_name != "1_completion":
                data_id = (frame.static_frame.topic_name, frame.serial_no)
                assert data_id not in data_to_put, f"{data_id}"
                data_to_put[data_id] = frame

    # Add all async adges (put to get)
    for tree in call_trees.values():
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
    for tree in call_trees.values():
        # Maps a topic to its callback counter
        # defaults to zero
        topic_to_callback_count: Dict[str, int] = collections.defaultdict(lambda: 0)
        for frame in anytree.PreOrderIter(tree.root):
            if frame.static_frame.function_name == "callback":
                topic_name = frame.static_frame.topic_name
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
                    warnings.warn(f"cb: {data_id} from {frame.static_frame.plugin} not found", UserWarning)

    return dynamic_dfg


def gen_static_dfg(
        call_trees: Mapping[int, CallTree],
        dynamic_dfg,
) -> Mapping[str, Any]:
    static_dfg = nx.DiGraph()
    for src, dst, edge_attrs in dynamic_dfg.edges(data=True):
        static_dfg.add_edge(
            src.static_frame,
            dst.static_frame,
            **edge_attrs,
        )
    return static_dfg

@ch_time_block.decor()
def gen_dfg_plot(
        static_dfg,
        output_dir: Path,
) -> Mapping[str, Any]:

    plugin_to_nodes: Dict[str, List[StaticFrame]] = collections.defaultdict(list)
    static_dfg_graphviz = pygraphviz.AGraph(strict=True, directed=True)

    def frame_to_id(frame: StaticFrame) -> str:
        return str(id(frame))

    def frame_to_label(frame: StaticFrame) -> str:
        top = f"frame.plugin_function_topic(' ')"
        stack = "\\n".join(str(frame.file_function_line()) for frame in frame.path[2:])
        return f"{top}\n{stack}"

    include_program = True
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
                # assert plugin
                plugin_to_nodes[node.plugin].append(node)
                static_dfg_graphviz.get_node(frame_to_id(node)).attr[
                    "label"
                ] = frame_to_label(node)

    # for plugin, nodes in plugin_to_nodes.items():
    #     static_dfg_graphviz.add_subgraph(map(frame_to_id, nodes), plugin, rank="same", rankdir="TB")
        
    # for static, dynamic in path_static_to_dynamic.items():
    #     for src, dst in zip(static[:-1], static[1:]):
    #         static_dfg_graphviz.get_edge(frame_to_id(src), frame_to_id(dst),).attr[
    #             "label"
    #         ] += " " + str(len(dynamic))

    # for path, instances in path_static_to_dynamic.items():
    #     for src, dst in zip(path[:-1], path[1:]):
    #         static_dfg_graphviz.get_edge(
    #             frame_to_id(src),
    #             frame_to_id(dst),
    #         ).attr["color"] = "blue"

    dot_path = Path(output_dir / "dataflow.dot")
    img_path = output_dir / "dataflow.png"
    static_dfg_graphviz.write(dot_path)
    static_dfg_graphviz.draw(img_path, prog="dot")


@memoize(group=group, verbose=False)
def gen_path_static_to_dynamic(
        static_dfg,
        dynamic_dfg,
        static_to_dynamic,
) -> Any:
    path_static_to_dynamic: Mapping[
        Tuple[StaticFrame, ...], List[Tuple[DynamicFrame, ...]]
    ] = collections.defaultdict(list)

    def is_dst(static_frame: StaticFrame) -> bool:
        return static_frame.function_name == "exit"

    def is_src(static_frame: StaticFrame) -> bool:
        return static_frame.function_name == "entry"

    dynamic_dfg_rev = dynamic_dfg.reverse()

    def explore(
        dynamic_frame: DynamicFrame,
        dynamic_path: Tuple[DynamicFrame, ...],
        tabu: Set[DynamicFrame],
        iter: int,
    ) -> Iterator[Tuple[DynamicFrame, ...]]:
        dynamic_path += (dynamic_frame,)
        assert iter < 30
        if is_src(dynamic_frame.static_frame):
            yield dynamic_path
        for next_dynamic_frame in dynamic_dfg_rev[dynamic_frame]:
            if next_dynamic_frame.static_frame not in tabu:
                yield from explore(
                    next_dynamic_frame,
                    dynamic_path,
                    tabu | {next_dynamic_frame.static_frame},
                    iter + 1,
                )

    for static_dst in static_dfg:
        assert static_dst in static_to_dynamic, f"{static_dst!s} from static_dfg is not found in static_to_dynamic"
        assert static_to_dynamic[static_dst]
        for dynamic_dst in static_to_dynamic[static_dst]:
                for dynamic_path in explore(
                    dynamic_dst, (), {dynamic_dst.static_frame}, 0
                ):
                    dynamic_path = dynamic_path[::-1]
                    static_path = tuple(frame.static_frame for frame in dynamic_path)
                    path_static_to_dynamic[static_path].append(dynamic_path)

    def get_input_times(path: Iterable[DynamicFrame]) -> Iterable[int]:
        return (
            node.wall_start for node in path if node.static_frame.function_name == "put"
        )

    path_static_to_dynamic = {
        static: dynamic
        for static, dynamic in path_static_to_dynamic.items()
        if len(dynamic) > 80
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

    # TODO: compute how many puts are "used/ignored"
    return dict(path_static_to_dynamic_freshest)

@memoize(verbose=False, group=group)
def all_dfg(call_trees, output_dir, static_to_dynamic) -> Mapping[str, Any]:
    dynamic_dfg = gen_dynamic_dfg(call_trees)
    static_dfg = gen_static_dfg(call_trees, dynamic_dfg)
    # gen_dfg_plot(static_dfg, output_dir)
    path_static_to_dynamic = gen_path_static_to_dynamic(static_dfg, dynamic_dfg, static_to_dynamic)
    return path_static_to_dynamic

important_path_signatures: Mapping[str, List[int]] = {
    "CC":     (7,) * 2 +            (3,) * 3 +            (6,) * 4,
    "Render": (7,) * 2 +            (3,) * 3 + (5,) * 2 + (6,) * 6,
    "Cam":    (8,) * 2 + (2,) * 3 + (3,) * 2 +            (6,) * 4,
}

def gen_important_paths(
        path_static_to_dynamic,
) -> Any:
    def path_signature(path: Iterable[StaticFrame]) -> Tuple[int]:
        return tuple(int(frame.plugin) for frame in path)

    def find_path(paths: Iterable[Tuple[StaticFrame, ...]], signature: Tuple[int, ...]) -> Tuple[StaticFrame, ...]:
        for path in paths:
            if path_signature(path) == signature:
                return path
        print(f"{signature} is not found.")
        for path in paths:
            print(path_signature(path))
        raise KeyError()

    return {
        name: find_path(path_static_to_dynamic.keys(), path_signature)
        for name, path_signature in important_path_signatures.items()
    }

# TODO: Move the cutting to a later stage
time_offset = 7

@ch_time_block.decor()
def gen_path_metrics(
        path_static_to_dynamic,
) -> Mapping[str, Any]:
    path_metrics = {}
    for static_path, dynamic_paths in path_static_to_dynamic.items():
        data = np.array([
            [node.wall_start for node in dynamic_path] + [dynamic_path[-1].wall_stop]
            for dynamic_path in dynamic_paths
        ])

        assert (data[1:] - data[:-1] < 0).sum() < len(
            data
        ) * 0.01, "row monotonicity is violated for"

        assert (data[:, 1:] - data[:, :-1] < 0).sum() < len(
            data
        ) * 0.01, "column monotonicity is violated"

        frame_offset = np.argmax(data[:, 0] > time_offset)

        path_metrics[static_path] = {
            "times": data[frame_offset:],
            "rel_time": data[frame_offset:] - data[frame_offset:, 0][:, np.newaxis],
            "latency": data[frame_offset:, -1] - data[frame_offset:, (1 if static_path[0].plugin == 8 else 0)] ,
            "period": data[frame_offset+1:, -1] - data[frame_offset:-1, -1],
            "rt": data[frame_offset+1:, -1] - data[frame_offset:-1, 0],
        }
    return path_metrics

path_metric_names = ["latency", "period", "rt"]

@ch_time_block.decor()
def gen_path_metrics_plot(path_static_to_dynamic, important_paths, path_metrics, output_dir):
    def summarize(static_path: tuple[StaticFrame, ...]) -> str:
        dynamic_paths = path_static_to_dynamic[static_path]
        data = path_metrics[static_path]
        def link_str(i: int) -> str:
            frame_start = dynamic_paths[0][i].static_frame.plugin_function_topic("-")
            frame_end = dynamic_paths[0][i+1].static_frame.plugin_function_topic("-")
            label = f"{frame_start} -> {frame_end} (ms) "
            padding = max(0, 60 - len(label)) * " "
            numbers = summary_stats((data["rel_time"][:, i+1] - data["rel_time"][:, i]) / 1e6)
            return label + padding + numbers
        return "\n".join([
            f"{len(dynamic_paths)}" + " -> ".join(frame.plugin for frame in static_path),
            "\n".join(link_str(i) for i in range(len(dynamic_paths[0]) - 1)),
            "\n".join(f"{summary_stats(data[metric] / 1e6)}" for metric in path_metric_names),
        ])

    chains_dir = output_dir / "chains"
    write_dir({
        output_dir / "chains": {
            f"summary.txt": "\n".join(
                "\n".join([
                    name,
                    summarize(static_path),
                    "",
                ])
                for name, static_path in important_paths.items()
            ),
            **{
                path_name: {
                    **{
                        metric_name: {
                            "hist.png": histogram(
                                ys=path_metrics[path][metric_name] / 1e6,
                                xlabel=f"{metric_name} (ms)",
                                title=f"{metric_name.capitalize()} of {path_name}",
                            ),
                            "ts.png": timeseries(
                                ts=path_metrics[path]["times"][:, 0],
                                ys=path_metrics[path][metric_name] / 1e6,
                                ylabel=f"{metric_name} (ms)",
                                title=f"{metric_name.capitalize()} of {path_name}",
                            ),
                            "data.npy": capture_file(lambda file: np.savetxt(file, path_metrics[path][metric_name])),
                        }
                        for metric_name in path_metric_names
                    },
                    "times.npy": capture_file(lambda file: np.savetxt(file, path_metrics[path]["times"])),
                }
                for path_name, path in important_paths.items()
            },
        },
    })

def gen_project_compute_times(
        conditions: Mapping[str, Any],
        compute_times: Mapping[str, Any],
):
    return {
        frozendict(conditions): {
            frame.plugin_function(" "): {
                metric_name: [data[metric_name]]
                for metric_name in ["cpu_time", "wall_time"]
            }
            for frame, data in compute_times.items()
        },
    }

def gen_project_path_metrics(
        conditions: Mapping[str, Any],
        important_paths: Mapping[str, Any],
        path_metrics,
):
    return {
        frozendict(conditions): {
            path_name: {
                metric_name: [path_metrics[path][metric_name]]
                for metric_name in path_metric_names
            }
            for path_name, path in important_paths.items()
        },
    }
def gen_project_path_times(
        conditions: Mapping[str, Any],
        important_paths: Mapping[str, Any],
        path_metrics,
):
    return {
        frozendict(conditions): {
            path_name: [path_metrics[path]["times"]]
            for path_name, path in important_paths.items()
        },
    }

def collect(trees: List[Optional[CallTree]]):
    return {
        int(tree.thread_id): tree
        for tree in trees
        if tree is not None
    }


def analyze_trial_delayed(metrics: Path, output_dir: Path) -> Mapping[str, dask.Delayed[Any]]:
    delayed = dask.delayed if globals()["use_parallel"] else lambda x: x

    (output_dir / "source").write_text(str(metrics.resolve()))

    config = yaml.load((metrics / "config.yaml").read_text(), Loader=yaml.SafeLoader)
    conditions = config["conditions"]

    call_trees = delayed(collect)([
        delayed(Memoized(CallTree.from_database, verbose=False, group=group))(path)
        for path in (metrics / "frames").iterdir()
    ])
    static_to_dynamic = delayed(gen_static_to_dynamic)(call_trees)
    compute_times = delayed(gen_compute_times)(output_dir, call_trees, static_to_dynamic, conditions)
    path_static_to_dynamic = delayed(all_dfg)(call_trees, output_dir, static_to_dynamic)
    important_paths = delayed(gen_important_paths)(path_static_to_dynamic)
    path_metrics = delayed(gen_path_metrics)(path_static_to_dynamic)
    path_metrics_plot = delayed(gen_path_metrics_plot)(path_static_to_dynamic, important_paths, path_metrics, output_dir)

    compute_times_plot = delayed(gen_compute_times_plot)(call_trees, compute_times, output_dir)
    # callgraph_plot = delayed(gen_callgraph_plot)(output_dir, call_trees, static_to_dynamic)

    proj_compute_times = delayed(gen_project_compute_times)(conditions, compute_times)
    proj_path_metrics = delayed(gen_project_path_metrics)(conditions, important_paths, path_metrics)
    proj_path_times = delayed(gen_project_path_times)(conditions, important_paths, path_metrics)
    return {
        "conditions": conditions,
        "path_metrics_plot": path_metrics_plot,
        "compute_times_plot": compute_times_plot,
        "proj_compute_times": proj_compute_times,
        "proj_path_metrics": proj_path_metrics,
        "proj_path_times": proj_path_times,
        "condition_trials": {frozendict(conditions): [output_dir]},
    }

def combine(projected_trials: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    data = {
        "proj_compute_times": collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list))),
        "proj_path_metrics": collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list))),
        "proj_path_times": collections.defaultdict(lambda: collections.defaultdict(list)),
        "condition_trials": collections.defaultdict(list)
    }
    for trial in projected_trials:
        for condition, equiv_trials in trial["condition_trials"].items():
            data["condition_trials"][condition].extend(equiv_trials)
        for conditions, name_metric_series in trial["proj_compute_times"].items():
            for name, metric_series in name_metric_series.items():
                for metric, series in metric_series.items():
                    data["proj_compute_times"][conditions][name][metric].extend(series)
        for conditions, name_metric_series in trial["proj_path_metrics"].items():
            for name, metric_series in name_metric_series.items():
                for metric, series in metric_series.items():
                    data["proj_path_metrics"][conditions][name][metric].extend(series)
        # for conditions, name_metric_series in trial["proj_path_times"].items():
        #     for name, times in name_metric_series.items():
        #         data["proj_path_times"][conditions][name].extend(times)
    # Undo the defaultdict
    return undefault_dict(data)

@memoize(verbose=False, group=group)
def analyze_trials_projection(metrics_group: Path) -> Mapping[str, Any]:
    compute = dask.compute if globals()["use_parallel"] else lambda *args: args
    ret = combine(compute([
        analyze_trial_delayed(metrics, metrics)
        for metrics in metrics_group
    ])[0])
    gc.collect()
    return ret
