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
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import numpy as np
import pygraphviz  # type: ignore
import pandas as pd  # type: ignore
from frozendict import frozendict
import charmonium.time_block as ch_time_block
from charmonium.cache import memoize
import multiprocessing
from tqdm import tqdm
import dask.bag

from .call_tree import DynamicFrame, StaticFrame
from .types import Trial, Trials, conditions2label
from .util import clip
from .read_trials import group


def print_distribution(data: np.array, digits: int = 1) -> str:
    percentiles = [25, 75, 90, 95]
    percentiles_str = " " + " ".join(
        f"[{percentile}%]={np.percentile(data, percentile):,.{digits}f}"
        for percentile in percentiles
    )
    with np.errstate(invalid="ignore"):
        return f"{data.mean():,.{digits}f} +/- {data.std():,.{digits}f} ({data.std() / data.mean() * 100:.0f}%) med={np.median(data):,.{digits}f} count={len(data)}{percentiles_str}"

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def gen_static_to_dynamic(trial: Trial) -> Mapping[str, Any]:
    static_to_dynamic: Mapping[StaticFrame, List[DynamicFrame]] = collections.defaultdict(list)
    for tree in trial.call_trees.values():
        for static_frame, dynamic_frames in tree.static_to_dynamic.items():
            static_to_dynamic[static_frame].extend(dynamic_frames)
    return {"static_to_dynamic": dict(static_to_dynamic)}

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def plot_callgraph(trial: Trial) -> Mapping[str, Any]:
    """Generate a visualization of the callgraph."""
    total_time = sum([tree.root.cpu_time for tree in trial.call_trees.values()])
    cpu_timer_calls = sum([tree.calls for tree in trial.call_trees.values()])
    cpu_timer_overhead = cpu_timer_calls * 400
    # cpu overhead estimate: {cpu_timer_overhead / 1e6:.1f}ms, total_time: {total_time / 1e6:.1f}ms, percent error: {cpu_timer_overhead / total_time * 100 if total_time != 0 else 0:.1f}%
    graphviz = pygraphviz.AGraph(strict=True, directed=True)

    for static_frame, dynamic_frames in trial.output["static_to_dynamic"].items():
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

    dot_path = Path(trial.output_dir / "callgraph.dot")
    img_path = trial.output_dir / "callgraph.png"
    graphviz.write(dot_path)
    graphviz.draw(img_path, prog="dot")
    return {}

def gen_compute_times(trial: Trial) -> Mapping[str, Any]:
    top_level_fn_names = {
        "callback",
        "cam",
        "IMU",
        "load_camera_data",
        "_p_one_iteration",
    }
    output = {"compute_times": {}}
    for tree in trial.call_trees.values():
        for static_frame in anytree.PreOrderIter(tree.root.static_frame):
            fn_name_good = static_frame.function_name in top_level_fn_names or static_frame.function_name.startswith("_")
            plugin_name_good = static_frame.plugin != "1" or not trial.config["conditions"]["scheduler"] in {"static", "dynamic"}
            if fn_name_good and plugin_name_good:
                dynamic_frames = tree.static_to_dynamic[static_frame]
                cpu_times = np.array([dynamic_frame.cpu_time for dynamic_frame in dynamic_frames])
                wall_times = np.array([dynamic_frame.wall_time for dynamic_frame in dynamic_frames])
                ts = np.array([dynamic_frame.wall_start for dynamic_frame in dynamic_frames])
                output["compute_times"][static_frame] = {
                    "ts": ts,
                    "cpu_time": cpu_times,
                    "wall_time": wall_times,
                    "thread_id": tree.thread_id,
                }
    return output

def right_pad(text: str, length: int) -> str:
    return text + " " * max(0, length - len(text))

import pickle
pickle.dumps(StaticFrame)

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def plot_compute_times(trial: Trial) -> Mapping[str, Any]:
    def compute_times(frame_data: Tuple[StaticFrame, Mapping[str, Any]]) -> Tuple[str, str]:
        frame, data = frame_data
        return (
            frame.plugin,
            " ".join([
                right_pad(frame.plugin_function(' '), 30),
                right_pad(f"TID={data['thread_id']}", 10),
                right_pad(print_distribution(data["cpu_time"] / 1e6, digits=2) , 35),
                right_pad(print_distribution(data["wall_time"] / 1e6, digits=2) , 35),
            ]),
        )

    compute_dir = trial.output_dir / "compute_times"
    write_contents({
        compute_dir / "compute_times.txt": "\n".join([
            f"threads: {len(trial.call_trees)}",
            *map(second, sorted(map(compute_times, trial.output["compute_times"].items()))),
        ]),
        **{
            compute_dir / "hist" / f"{frame.plugin_function(' ')}.png": histogram(
                ys=data["cpu_time"] / 1e6,
                xlabel=f"CPU Time (ms)",
                title=f"Compute Time of {frame.plugin_function(' ')}"
            )
            for frame, data in trial.output["compute_times"].items()
        },
        **{
            compute_dir / "ts" / f"{frame.plugin_function(' ')}.png": timeseries(
                ts=data["ts"],
                ys=data["cpu_time"] / 1e6,
                ylabel=f"CPU Time (ms)",
                title=f"Compute Time of {frame.plugin_function(' ')}"
            )
            for frame, data in trial.output["compute_times"].items()
        },
    })
    return {}

class EdgeType(Enum):
    program = 1
    async_ = 2
    sync = 3

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def gen_dfg(trial: Trial) -> Mapping[str, Any]:
    """Generates a visualization of dataflow over Switchboard between plugins."""

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

    static_dfg = nx.DiGraph()
    for src, dst, edge_attrs in dynamic_dfg.edges(data=True):
        static_dfg.add_edge(
            src.static_frame,
            dst.static_frame,
            **edge_attrs,
        )
    return {"dynamic_dfg": dynamic_dfg, "static_dfg": static_dfg}

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def plot_dfg(trial: Trial) -> Mapping[str, Any]:

    plugin_to_nodes: Dict[str, List[StaticFrame]] = collections.defaultdict(list)
    static_dfg_graphviz = pygraphviz.AGraph(strict=True, directed=True)

    def frame_to_id(frame: StaticFrame) -> str:
        return str(id(frame))

    def frame_to_label(frame: StaticFrame) -> str:
        top = f"frame.plugin_function_topic(' ')"
        stack = "\\n".join(str(frame.file_function_line()) for frame in frame.path[2:])
        return f"{top}\n{stack}"

    include_program = True
    for src, dst, edge_attrs in trial.output["static_dfg"].edges(data=True):
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

    dot_path = Path(trial.output_dir / "dataflow.dot")
    img_path = trial.output_dir / "dataflow.png"
    static_dfg_graphviz.write(dot_path)
    static_dfg_graphviz.draw(img_path, prog="dot")
    return {}


@memoize(verbose=False, group=group)# @ch_time_block.decor()
def gen_paths(trial: Trial) -> Mapping[str, Any]:
    path_static_to_dynamic: Mapping[
        Tuple[StaticFrame, ...], List[Tuple[DynamicFrame, ...]]
    ] = collections.defaultdict(list)

    def is_dst(static_frame: StaticFrame) -> bool:
        return static_frame.function_name == "exit"

    def is_src(static_frame: StaticFrame) -> bool:
        return static_frame.function_name == "entry"

    dynamic_dfg_rev = trial.output["dynamic_dfg"].reverse()

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

    for static_dst in trial.output["static_dfg"]:
        assert static_dst in trial.output["static_to_dynamic"], f"{static_dst!s} from static_dfg is not found in static_to_dynamic"
        assert trial.output["static_to_dynamic"][static_dst]
        for dynamic_dst in trial.output["static_to_dynamic"][static_dst]:
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

    path_static_to_dynamic = dict(path_static_to_dynamic_freshest)

    # TODO: compute how many puts are "used/ignored"

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
        "important_paths": {
            name: find_path(path_static_to_dynamic.keys(), path_signature)
            for name, path_signature in important_path_signatures.items()
        },
        "path_static_to_dynamic": path_static_to_dynamic,
    }

@memoize(verbose=False, group=group)
def gen_path_times(trial: Trial) -> Mapping[str, Any]:
    path_times = {}
    for static_path, dynamic_paths in trial.output["path_static_to_dynamic"].items():
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
    return {"path_times": path_times}

Key = TypeVar("Key")
Val = TypeVar("Val")
def dict_concat(dicts: Iterable[Mapping[Key, Val]]) -> Mapping[Key, Val]:
    return {
        key: val
        for dict in dicts
        for key, val in dict.items()
    }

all_metrics = ["latency", "period", "rt"]

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def plot_paths(trial: Trial) -> Mapping[str, Any]:
    def summarize(static_path: tuple[StaticFrame, ...]) -> str:
        dynamic_paths = trial.output["path_static_to_dynamic"][static_path]
        data = trial.output["path_times"][static_path]
        def link_str(i: int) -> str:
            frame_start = dynamic_paths[0][i].static_frame.plugin_function_topic("-")
            frame_end = dynamic_paths[0][i+1].static_frame.plugin_function_topic("-")
            label = f"{frame_start} -> {frame_end} (ms) "
            padding = max(0, 60 - len(label)) * " "
            numbers = print_distribution((data["rel_time"][:, i+1] - data["rel_time"][:, i]) / 1e6)
            return label + padding + numbers
        return "\n".join([
            f"{len(dynamic_paths)}" + " -> ".join(frame.plugin for frame in static_path),
            "\n".join(link_str(i) for i in range(len(dynamic_paths[0]) - 1)),
            "\n".join(f"{print_distribution(data[metric] / 1e6)}" for metric in all_metrics),
        ])

    chains_dir = trial.output_dir / "chains"
    write_contents({
        **{
            chains_dir / f"{name}.txt": summarize(static_path)
            for name, static_path in trial.output["important_paths"].items()
        },
        **dict_concat(
            {
                chains_dir / name / "hist"/ f"{metric}.png": histogram(
                    ys=trial.output["path_times"][static_path][metric] / 1e6,
                    xlabel=f"{metric} (ms)",
                    title=f"{metric.capitalize()} of {name}",
                )
                for name, static_path in trial.output["important_paths"].items()
            }
            for metric in all_metrics
        ),
        **dict_concat(
            {
                chains_dir / name / "ts"/ f"{metric}.png": timeseries(
                    ts=trial.output["path_times"][static_path]["times"][:, 0],
                    ys=trial.output["path_times"][static_path][metric] / 1e6,
                    ylabel=f"{metric} (ms)",
                    title=f"{metric.capitalize()} of {name}",
                )
                for name, static_path in trial.output["important_paths"].items()
            }
            for metric in all_metrics
        ),
    })
    return {}

@memoize(verbose=False, group=group)
def plot_cc_stuff(trials: Trials) -> Mapping[str, Any]:
    datas: Mapping[str, Mapping[str, np.array]] = {}
    max_lat = 0
    for trial in trials.each:
        static_path = trial.output["important_paths"]["CC"]
        ts = trial.output["path_times"][static_path]["times"]
        lat = trial.output["path_times"][static_path]["latency"]
        max_lat = max(np.percentile(lat, 99), max_lat)
        datas[conditions2label(trial.config["conditions"])] = {
            "ts": ts,
            "lat": lat
        }

    for label, data in datas.items():
        plt.rcParams.update({
            "figure.figsize": [8, 6],
            "grid.color": "#AAAAAA",
            "figure.autolayout": True,
	})

        fig, ax = plt.subplots()
        ax.set_title("Rotational MTP")
        ax.set_ylabel("Motion-to-photon latency (ms)", fontsize=20)
        plt.ylim(0, max_lat / 1e6)
        ax.set_xlabel("Time (s)", fontsize=20)
        start = data["ts"][0, 0]
        ax.plot((data["ts"][:, 0] - start) / 1e9, data["lat"] / 1e6, label=label)
        ax.legend(loc="upper right", fontsize=12)
        ax.tick_params("x", labelsize=16)
        ax.tick_params("y", labelsize=16)
        ax.grid(True)
        fig.savefig(trial.output_dir / "mtp.png")
        plt.close(fig)

    fig, ax = plt.subplots()
    ax.set_ylabel("Count", fontsize=20)
    ax.set_xlabel("Motion-to-photon latency (ms)", fontsize=20)
    # ax.set_xlim(0, max_lat)
    for label, data in datas.items():
        ax.hist(data["lat"], bins=25, histtype="step", label=label)
    fig.savefig("mtp.png")
    plt.close(fig)
    return {}

@memoize(verbose=False, group=group)
def group_by_conditions(trials: Trials) -> Mapping[str, Any]:
    by_conditions: Mapping[Conditions, List[Trial]] = collections.defaultdict(list)
    for trial in trials.each:
        by_conditions[frozendict(trial.config["conditions"].items())].append(trial)

    write_contents({
        trials.output_dir / "conditions.txt": "\n".join(
            f"# {conditions2label(condition)}\n" + "\n".join(
                str(trial.output_dir)
                for trial in trials
            )
            for condition, trials in by_conditions.items()
        )
    })
    return {"by_conditions": dict(by_conditions)}

def write_contents(content_map: Mapping[Path, Union[str, bytes]]) -> None:
    for path, content in content_map.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        if isinstance(content, str):
            path.write_text(content)
        else:
            path.write_bytes(content)

import scipy.stats

def histogram(
        ys: np.array,
        xlabel: str,
        title: str,
        bins: int = 50,
        cloud: bool = True,
        logy: bool = True,
        grid: bool = False,
) -> bytes:
    fake_file = io.BytesIO()
    fig, ax = plt.subplots()
    ax.hist(ys, bins=bins, align='mid')
    if cloud:
        ax.plot(ys, np.random.randn(*ys.shape) * (ax.get_ylim()[1] * 0.2) + (ax.get_ylim()[1] * 0.5) * np.ones(ys.shape), linestyle='', marker='.', ms=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Occurrences (count)")
    if grid:
        ax.grid(True, which="major", axis="both")
    if logy:
        ax.set_yscale("log")
    fig.savefig(fake_file)
    plt.close(fig)
    return fake_file.getvalue()

def timeseries(
        ts: np.array,
        ys: np.array,
        title: str,
        ylabel: str,
        series_label: Optional[str] = None,
        grid: bool = False,
) -> bytes:
    fake_file = io.BytesIO()
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Time since start (sec)")
    if grid:
        ax.grid(True, which="major", axis="both")
    if len(ts) == len(ys) + 1:
        ts = ts[:-1]
    if len(ts) == len(ys) - 1:
        ys = ys[:-1]
    ax.plot((ts - ts[0]) / 1e9, ys, label=series_label)
    fig.savefig(fake_file)
    plt.close(fig)
    return fake_file.getvalue()

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def agg_compute_times(trials: Trials) -> Mapping[str, Any]:
    compute_times: Mapping[str, List[np.array]] = collections.defaultdict(list)
    for trial in trials.each:
        for frame, data in trial.output["compute_times"].items():
            compute_times[frame.plugin_function(" ")].append(data["cpu_time"])
    compute_times_dir = trials.output_dir / "compute_times"

    write_contents({
        compute_times_dir / "summary.txt": "\n".join(
            f"{label} {(len(times))}: {print_distribution(np.concatenate(times) / 1e6)}"
            for label, times in compute_times.items()
        ),
        **{
            compute_times_dir / "hist" / (label + ".png"): histogram(np.concatenate(times) / 1e6, "CPU Time (ms)", f"Compute Times of {label}")
            for label, times in compute_times.items()
        },
    })
    return {"compute_times": dict(compute_times)}

A = TypeVar("A")
B = TypeVar("B")
def second(pair: Tuple[A, B]) -> B:
    return pair[1]

@memoize(verbose=False, group=group)# @ch_time_block.decor()
def compare_compute_times(trials: Trials) -> Mapping[str, Any]:
    keys = trials.each[0].output["compute_times"].keys()
    for trial in trials.each:
        keys &= trial.output["compute_times"].keys()
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
            "\n".join(f"{print_distribution(data[metric] / 1e6)}" for metric in all_metrics),
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

# def compare_chain_metrics(all_trials: Trials) -> None:
#     for conditions, trials in all_trials.output["by_conditions"].items():
#         for a, b in itertools.combinations(trials, 2):
#             names = a.output["important_paths"].keys()
#             for name in names:
#                 scipy.stats.ks_2samp(a.output["lat"])

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

analyze_trial_fns: List[Callable[[Trial], None]] = [
    gen_static_to_dynamic,
    plot_callgraph,
    gen_compute_times,
    plot_compute_times,
    gen_dfg,
    plot_dfg,
    gen_paths,
    gen_path_times,
    plot_paths,
]

analyze_trials_fns: List[Callable[[Trials], None]] = [
    group_by_conditions,
    agg_compute_times,
    compare_compute_times,
    agg_chain_metrics,
    plot_cc_stuff,
]

def get_name(function, default: str = "unknown") -> str:
    return getattr(function, "name", getattr(function, "__name__", default))

def analyze_trials(trials: Trials) -> None:
    """Main entrypoint for inter-trial analysis.

    All inter-trial analyses should be started from here.

    """

    for trial in trials.each:
        for analyze_trial_fn in analyze_trial_fns:
            with ch_time_block.ctx(get_name(analyze_trial_fn)):
                output = analyze_trial_fn(trial)
                generated_keys = output.keys()
                if generated_keys:
                    print(f"Generated {' '.join(generated_keys)}")
                trial.output.update(output)

    for analyze_trials_fn in analyze_trials_fns:
        with ch_time_block.ctx(get_name(analyze_trials_fn)):
            output = analyze_trials_fn(trials)
            trials.output.update(output)
