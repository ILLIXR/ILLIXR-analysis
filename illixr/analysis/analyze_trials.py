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
from .util import clip, timeseries, histogram, write_contents, dict_concat, right_pad, summary_stats, chunker, second
import dask.distributed
dask.config.set({"distributed.worker.daemon": False})
client = dask.distributed.Client(
    address=dask.distributed.LocalCluster(
        n_workers=multiprocessing.cpu_count(),
    ),
)
print(client.dashboard_link)

group = MemoizedGroup(size="40GiB", fine_grain_persistence=True)

def conditions2label(conditions: Mapping[str, Any]) -> str:
    label = f"{conditions['scheduler']} {conditions['cpus']}x{conditions['cpu_freq']:.1f}GHz"
    return label

def dict_delayed(*keys):
    def decorator(func):
        def inner_func(*args, **kwargs):
            delayed_key = dask.delayed(func)(*args, **kwargs)
            if len(keys) == 0:
                return {func.__name__: delayed_key}
            elif len(keys) == 1:
                return {keys[0]: delayed_key}
            else:
                raise ValueError
        return inner_func
    return decorator


def gen_config_conditions(metrics: Path, **kwargs: Any) -> Mapping[str, Any]:
    config = yaml.load((metrics / "config.yaml").read_text(), Loader=yaml.SafeLoader)
    return {"config": config, "conditions": config["conditions"]}

def gen_call_trees(metrics: Path, **kwargs: Any) -> Mapping[str, Any]:
    @dask.delayed
    def collect(trees: List[Optional[CallTree]]):
        return {
            int(tree.thread_id): tree
            for tree in trees
            if tree is not None
        }
    return {"call_trees": collect([
        dask.delayed(Memoized(CallTree.from_database, verbose=False, group=group))(path)
        for path in (metrics / "frames").iterdir()
    ])}

@dict_delayed("static_to_dynamic")
@memoize(verbose=False, group=group)
def gen_static_to_dynamic(call_trees: Mapping[int, CallTree], **kwargs) -> Mapping[StaticFrame, List[DynamicFrame]]:
    static_to_dynamic: Mapping[StaticFrame, List[DynamicFrame]] = collections.defaultdict(list)
    for tree in call_trees.values():
        for static_frame, dynamic_frames in tree.static_to_dynamic.items():
            static_to_dynamic[static_frame].extend(dynamic_frames)
    return dict(static_to_dynamic)

@dict_delayed()
@memoize(verbose=False, group=group)# @ch_time_block.decor()
def plot_callgraph(
        output_dir: Path,
        call_trees: Mapping[int, CallTree],
        static_to_dynamic: Mapping[StaticFrame, List[DynamicFrame]],
        **kwargs: Any,
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

@dict_delayed("compute_times")
@memoize(verbose=False, group=group)
def gen_compute_times(
        output_dir: Path,
        call_trees: Mapping[int, CallTree],
        static_to_dynamic: Mapping[StaticFrame, List[DynamicFrame]],
        conditions: Mapping[str, Any],
        **kwargs: Any,
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
            plugin_name_good = static_frame.plugin != "1" or not conditions["scheduler"] in {"static", "dynamic"}
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

@dict_delayed()
@memoize(verbose=False, group=group)# @ch_time_block.decor()
def plot_compute_times(
        call_trees: Mapping[int, CallTree],
        compute_times: Mapping[StaticFame, Mapping[str, Any]],
        output_dir: Path,
        **kwargs: Any,
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
    write_contents({
        compute_dir / "compute_times.txt": "\n".join([
            f"threads: {len(call_trees)}",
            f"cpu overhead estimate: {cpu_timer_overhead / 1e6:.1f}ms, total_time: {total_time / 1e6:.1f}ms, percent error: {cpu_timer_overhead / total_time * 100 if total_time != 0 else 0:.1f}%",
            *map(second, sorted(map(summarize_compute_times, compute_times.items()))),
        ]),
        **{
            compute_dir / "hist" / f"{frame.plugin_function(' ')}.png": histogram(
                ys=data["cpu_time"] / 1e6,
                xlabel=f"CPU Time (ms)",
                title=f"Compute Time of {frame.plugin_function(' ')}"
            )
            for frame, data in compute_times.items()
        },
        **{
            compute_dir / "ts" / f"{frame.plugin_function(' ')}.png": timeseries(
                ts=data["ts"],
                ys=data["cpu_time"] / 1e6,
                ylabel=f"CPU Time (ms)",
                title=f"Compute Time of {frame.plugin_function(' ')}"
            )
            for frame, data in compute_times.items()
        },
    })

class EdgeType(Enum):
    program = 1
    async_ = 2
    sync = 3

@dict_delayed("dynamic_dfg")
@memoize(verbose=False, group=group)# @ch_time_block.decor()
def gen_dynamic_dfg(
        call_trees: Mapping[int, CallTree],
        **kwargs: Any,
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


@dict_delayed("static_dfg")
@memoize(verbose=False, group=group)# @ch_time_block.decor()
def gen_static_dfg(
        call_trees: Mapping[int, CallTree],
        dynamic_dfg,
        **kwargs: Any,
) -> Mapping[str, Any]:
    static_dfg = nx.DiGraph()
    for src, dst, edge_attrs in dynamic_dfg.edges(data=True):
        static_dfg.add_edge(
            src.static_frame,
            dst.static_frame,
            **edge_attrs,
        )
    return static_dfg

@dict_delayed()
@memoize(verbose=False, group=group)# @ch_time_block.decor()
def plot_dfg(
        static_dfg,
        output_dir: Path,
        **kwargs: Any
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


@dict_delayed("path_static_to_dynamic")
@memoize(verbose=False, group=group)# @ch_time_block.decor()
def gen_path_static_to_dynamic(
        static_dfg,
        dynamic_dfg,
        static_to_dynamic,
        **kwargs
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

@dict_delayed("important_paths")
@memoize(verbose=False, group=group)# @ch_time_block.decor()
def gen_important_paths(
        path_static_to_dynamic,
        **kwargs: Any,
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

    important_path_signatures: Mapping[str, List[int]] = {
        "CC":     (7,) * 2 +            (3,) * 3 +            (6,) * 4,
        "Render": (7,) * 2 +            (3,) * 3 + (5,) * 2 + (6,) * 6,
        "Cam":    (8,) * 2 + (2,) * 3 + (3,) * 2 +            (6,) * 4,
    }

    return {
        name: find_path(path_static_to_dynamic.keys(), path_signature)
        for name, path_signature in important_path_signatures.items()
    }

@dict_delayed("path_times")
@memoize(verbose=False, group=group)
def gen_path_times(
        path_static_to_dynamic,
        **kwargs: Any,
) -> Mapping[str, Any]:
    path_times = {}
    for static_path, dynamic_paths in path_static_to_dynamic.items():
        data = (
            np.array(
                list(
                    [node.wall_start for node in dynamic_path]
                    + [dynamic_path[-1].wall_stop]
                    for dynamic_path in dynamic_paths
                )
            )
        )

        assert (data[1:] - data[:-1] < 0).sum() < len(
            data
        ) * 0.01, "row monotonicity is violated for"

        assert (data[:, 1:] - data[:, :-1] < 0).sum() < len(
            data
        ) * 0.01, "column monotonicity is violated"

        path_times[static_path] = {
            "times": data,
            "rel_time": data - data[:, 0][:, np.newaxis],
            "latency": data[:, -1] - data[:, 0],
            "period": data[1:, -1] - data[:-1, -1],
            "rt": data[1:, -1] - data[:-1, 0],
        }
    return path_times

all_metrics = ["latency", "period", "rt"]

@dict_delayed()
@memoize(verbose=False, group=group)# @ch_time_block.decor()
def plot_paths(path_static_to_dynamic, important_paths, path_times, output_dir, **kwargs: Any):
    def summarize(static_path: tuple[StaticFrame, ...]) -> str:
        dynamic_paths = path_static_to_dynamic[static_path]
        data = path_times[static_path]
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
            "\n".join(f"{summary_stats(data[metric] / 1e6)}" for metric in all_metrics),
        ])

    chains_dir = output_dir / "chains"
    write_contents({
        **{
            chains_dir / f"{name}.txt": summarize(static_path)
            for name, static_path in important_paths.items()
        },
        **dict_concat(
            {
                chains_dir / name / "hist"/ f"{metric}.png": histogram(
                    ys=path_times[static_path][metric] / 1e6,
                    xlabel=f"{metric} (ms)",
                    title=f"{metric.capitalize()} of {name}",
                )
                for name, static_path in important_paths.items()
            }
            for metric in all_metrics
        ),
        **dict_concat(
            {
                chains_dir / name / "ts"/ f"{metric}.png": timeseries(
                    ts=path_times[static_path]["times"][:, 0],
                    ys=path_times[static_path][metric] / 1e6,
                    ylabel=f"{metric} (ms)",
                    title=f"{metric.capitalize()} of {name}",
                )
                for name, static_path in important_paths.items()
            }
            for metric in all_metrics
        ),
    })

@dict_delayed("proj_compute_times")
def project_compute_times(
        conditions: Mapping[str, Any],
        compute_times: Mapping[str, Any],
        **kwargs,
) -> Mapping[str, Any]:
    return {
        conditions["cpu_freq"]: {
            frame.plugin_function(" "): {
                metric: data[metric]
                for metric in ["cpu_time", "wall_time"]
            }
            for frame, data in compute_times.items()
        },
    }

@dict_delayed("proj_path_times")
def project_path_times(
        conditions: Mapping[str, Any],
        important_paths: Mapping[str, Any],
        path_times,
        **kwargs,
) -> Mapping[str, Any]:
    return {
        frozendict(conditions): {
            name: {
                metric: path_times[static_path][metric]
                for metric in all_metrics
            }
            for name, static_path in important_paths.items()
        },
    }

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def agg_compute_times(trials: Trials) -> Mapping[str, Any]:
    compute_times: Mapping[str, List[np.array]] = collections.defaultdict(list)
    for trial in trials.each:
        for frame, data in compute_times.items():
            compute_times[frame.plugin_function(" ")].append(data["cpu_time"])
    compute_times_dir = trials.output_dir / "compute_times"

    write_contents({
        compute_times_dir / "summary.txt": "\n".join(
            f"{label} {(len(times))}: {summary_stats(np.concatenate(times) / 1e6)}"
            for label, times in compute_times.items()
        ),
        **{
            compute_times_dir / "hist" / (label + ".png"): histogram(np.concatenate(times) / 1e6, "CPU Time (ms)", f"Compute Times of {label}")
            for label, times in compute_times.items()
        },
    })
    return {"compute_times": dict(compute_times)}

import scipy.stats

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def compare_compute_times(trials: Trials) -> Mapping[str, Any]:
    keys = trials.each[0].output["compute_times"].keys()
    for trial in trials.each:
        keys &= compute_times.keys()
    def diff(pair: Tuple[Trial, Trial]) -> Tuple[float, str]:
        a, b = pair
        a_keys = a.output["compute_times"].keys()
        b_keys = b.output["compute_times"].keys()
        for key in (a_keys & b_keys):
            ks, p = scipy.stats.ks_2samp(a.output["compute_times"][key][1], b.output["compute_times"][key][1])
            return p, f"- {a.output_dir!s} <-> {b.output_dir!s}: {p:.2f} {ks}"

    compute_times = trials.output_dir / "compute_times"
    write_contents({
        compute_times / "consistency.txt": "\n".join(
            f"# {key}\n" + "\n".join(map(second, sorted(map(diff, tqdm(itertools.combinations(trials.each, 2)))))) + "\n"
            for key in keys
        )
    })
    return {}

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def agg_chain_metrics(trials: Trials) -> Mapping[str, Any]:
    f = lambda: collections.defaultdict(lambda: {"latency": [], "period": [], "rt": []})
    important_chain_names = trials.each[0].output["important_paths"].keys()
    conditions = set(frozendict(trial.config["conditions"]) for trial in trials.each)
    chain_metrics: Mapping[str, Mapping[frozendict, Mapping[str, List[np.array]]]] = {
        name: f()
        for name in important_chain_names
    }
    for trial in trials.each:
        conditions = frozendict(trial.config["conditions"])
        for name, static_path in trial.output["important_paths"].items():
            for metric in all_metrics:
                chain_metrics[name][conditions][metric].append(
                    trial.output["path_times"][static_path][metric]
                )

    chains_dir = trials.output_dir / "chains"

    def summarize(static_path: tuple[StaticFrame, ...]) -> str:
        dynamic_paths = trial.output["path_static_to_dynamic"][static_path]
        data = trial.output["path_times"][static_path]
        return "\n".join([
            "\n".join(f"{summary_stats(data[metric] / 1e6)}" for metric in all_metrics),
        ])


    write_contents({
        **{
            chains_dir / f"summary.txt": "\n".join(
                "\n".join([f"# {condition}"] + [
                    "\n".join([
                        " ".join([
                            right_pad("{name} {metric} ({len(np.concat(chain_metrics[name][condition][metric]))})", length=25),
                            print_dist(np.concat(chain_metrics[name][condition][metric]))
                        ])
                        for metric in all_metrics
                    ])
                    for name in important_chain_names
                ] + [""])
                for condition in conditions
            )
        },
        **dict_concat(
                "\n".join([f"# {condition}"] + [
                    "\n".join([
                        " ".join([
                            right_pad("{name} {metric} ({len(np.concat(chain_metrics[name][condition][metric]))})", length=25),
                            print_dist(np.concat(chain_metrics[name][condition][metric]))
                        ])
                        for metric in all_metrics
                    ])
                    for name in important_chain_names
                ] + [""])
                for condition in conditions
            ),
    })
    # TODO: write metrics
    # TODO: plot hist
    return {"chain_metrics": {key: dict(val) for key, val in chain_metrics.items()}}

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def compare_chain_metrics(trials: Trials) -> Mapping[str, Any]:
    conditions = set(
        frozendict({
            key: val
            for key, val in condition
            if key != "scheduler"
        })
        for condition in trials.output["by_conditions"].keys()
    )
    all_schedulers = ["default", "priority", "manual", "static", "dynamic"]
    for condition in conditions:
        schedulers = [
            scheduler
            for scheduler in all_schedulers
            if frozendict(condition.copy(scheduler=scheduler)) in trial.output["by_conditions"]
        ]
        print(f"## {conditions2label(condition)}")
        for sched_a, sched_b in itertools.combinations(schedulers, 2):
            trial_a = trials.output["compute"]
            for name in a.output["important_chains"].keys():
                for metric in all_metrics:
                    a_data = trial_a["path_times"][trial_a.output["important_paths"][name]][metric]
                    b_data = trial_b["path_times"][trial_b.output["important_paths"][name]][metric]
                    np.median(a_data) / np.median(b_data)
                    np.stddev(a_data) / np.stddev(b_data)
    return {}

analyze_trial_fns: List[Callable[[Trial], None]] = [
    gen_config_conditions,
    gen_call_trees,
    gen_static_to_dynamic,
    plot_callgraph,
    gen_compute_times,
    gen_dynamic_dfg,
    gen_static_dfg,
    plot_dfg,
    gen_path_static_to_dynamic,
    gen_important_paths,
    gen_path_times,
    plot_paths,
    plot_compute_times,
    project_compute_times,
    project_path_times,
]

def analyze_trial(metrics: Path, output_dir: Path) -> Mapping[str, dask.Delayed[Any]]:
    dct = {"metrics": metrics, "output_dir": output_dir}
    print(metrics.name)
    for analyze_trial in analyze_trial_fns:
        dct.update(analyze_trial(**dct))
    return dct

# def analyze_trials(all_metrics: List[Path], output_dir: Path) -> None:
#     lst = [analyze_trial(metrics, metrics) for metrics in all_metrics]
#     print("dask.compute")
#     dask.compute(lst)
#     print("dask.compute done")
#     client.shutdown()

def combine(projected_trials: Iterable[Mapping[str, Any]]) -> Mapping[str, Any]:
    key_condition_name_metric_series = {
        "proj_compute_times": collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list))),
        "proj_path_times": collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list))),
    }
    for trial in projected_trials:
        for cpu_freq, name_metric_series in trial["proj_compute_times"].items():
            for name, metric_series in name_metric_series.items():
                for metric, series in metric_series.items():
                    key_condition_name_metric_series["proj_compute_times"][cpu_freq][name][metric].append(series)
        for conditions, name_metric_series in trial["proj_path_times"].items():
            for name, metric_series in name_metric_series.items():
                for metric, series in metric_series.items():
                    key_condition_name_metric_series["proj_path_times"][conditions][name][metric].append(series)
    # Undo the defaultdict
    return {
        key: {
            condition: {
                name: {
                    metric: series
                    for metric, series in metric_series.items()
                }
                for name, metric_series in name_metric_series.items()
            }
            for condition, name_metric in condition_name_metric_series.items()
        }
        for key, condition_name_metric_series in  key_condition_name_metric_series.items()
    }

@memoize(verbose=False, group=group)
def gen_aggregate(all_metrics: List[Path]) -> Mapping[str, Any]:
    return combine(
        combine(dask.compute([
            analyze_trial(metrics, metrics)
            for metrics in metrics_group
        ])[0])
        for metrics_group in chunker(all_metrics, multiprocessing.cpu_count())
    )

def analyze_trials(all_metrics: List[Path], output_dir: Path) -> None:
    aggregate = gen_aggregate(all_metrics)
