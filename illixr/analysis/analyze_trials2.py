import collections
import itertools
from pathlib import Path
from typing import Any, List, Mapping

import charmonium.time_block as ch_time_block
import numpy as np
import scipy.stats
import charmonium.cache
from frozendict import frozendict
from tqdm import tqdm

from illixr.analysis.analyze_trials import (
    analyze_trials_projection,
    combine,
    conditions2label,
    group,
    important_path_signatures,
    path_metric_names,
)
from illixr.analysis.util import (
    chunker,
    histogram,
    omit,
    right_pad,
    second,
    summary_stats,
    write_dir,
)

use_memoize = False

memoize = charmonium.cache.memoize if use_memoize else (lambda **y: lambda x: x)


@memoize(group=group)
def compare_compute_times(
    output_dir: Path,
    proj_compute_times: Mapping[float, Mapping[str, Mapping[str, List[np.ndarray]]]],
    **kwargs,
) -> Mapping[str, Any]:
    write_dir(
        {
            output_dir
            / "compute_times": {
                "summary.txt": "\n".join(
                    [
                        "\n".join(
                            [
                                conditions2label(conditions),
                                "\n".join(
                                    [
                                        " ".join(
                                            [
                                                right_pad(label, 20),
                                                "len:",
                                                str(len(data["wall_time"])),
                                                "cpu_time:",
                                                right_pad(
                                                    summary_stats(
                                                        np.concatenate(data["cpu_time"])
                                                        / 1e6,
                                                        digits=2,
                                                    ),
                                                    95,
                                                ),
                                                "wall_time:",
                                                right_pad(
                                                    summary_stats(
                                                        np.concatenate(
                                                            data["wall_time"]
                                                        )
                                                        / 1e6,
                                                        digits=2,
                                                    ),
                                                    95,
                                                ),
                                            ]
                                        )
                                        for label, data in label_data.items()
                                    ]
                                ),
                                "",
                            ]
                        )
                        for conditions, label_data in proj_compute_times.items()
                    ]
                ),
                "convergence.txt": "\n".join(
                    [
                        "\n".join(
                            [
                                conditions2label(conditions),
                                "\n".join(
                                    [
                                        "\n".join(
                                            [
                                                f"{label} {metric}:",
                                                " ".join(
                                                    [
                                                        f"{p:.4f}"
                                                        for p, ks in sorted(
                                                            scipy.stats.ks_2samp(a, b)[
                                                                ::-1
                                                            ]
                                                            for a, b in itertools.combinations(
                                                                series, 2
                                                            )
                                                        )
                                                    ]
                                                ),
                                            ]
                                        )
                                        for label, data in label_data.items()
                                        for metric, series in data.items()
                                    ]
                                ),
                                "",
                            ]
                        )
                        for conditions, label_data in proj_compute_times.items()
                    ]
                ),
                **{
                    conditions2label(conditions): {
                        label: {
                            f"{metric}.png": histogram(
                                ys=np.concatenate(series) / 1e6,
                                xlabel=f"{metric.replace('_', ' ').replace('cpu', 'CPU').capitalize()} (ms)",
                                title=f"Compute Times of {label}",
                            )
                            for metric, series in data.items()
                        }
                        for label, data in label_data.items()
                    }
                    for conditions, label_data in proj_compute_times.items()
                },
            },
        }
    )
    return {}


@ch_time_block.decor(print_start=False)
def compare_path_metrics(
    output_dir: Path,
    proj_path_metrics: Mapping[
        frozendict, Mapping[str, Mapping[str, List[np.ndarray]]]
    ],
    **kwargs,
) -> Mapping[str, Any]:
    path_metrics_dir = output_dir / "path_metrics"
    scheduler_conditions = collections.defaultdict(list)
    for conditions in proj_path_metrics.keys():
        sched_dict = frozendict(
            {
                "scheduler": conditions["scheduler"],
                **({"swap": conditions["swap"]} if "swap" in conditions else {}),
            }
        )
        scheduler_conditions[sched_dict].append(conditions)

    def inner_summarize_diff(conditions1, conditions2, name, metric):
        # len_a = len(path_metrics[conditions1][name][metric])
        # len_b = len(path_metrics[conditions2][name][metric])

        # So... I found a case where hash(conditions1) != hash(key) but conditions1 == key.
        # This is possible because: the hash of a frozendict is the hash of its contents, the hash is stored rather than recomputed, and it is not reset after unpickling.
        # hash is not comparable across different runs, but that's exactly the comparison frozendict will make.
        conditions1 = [
            conditions
            for conditions in proj_path_metrics.keys()
            if conditions1 == conditions
        ][0]
        conditions2 = [
            conditions
            for conditions in proj_path_metrics.keys()
            if conditions2 == conditions
        ][0]
        a = np.concatenate(proj_path_metrics[conditions1][name][metric])
        b = np.concatenate(proj_path_metrics[conditions2][name][metric])
        assert conditions1["cpu_freq"] == conditions2["cpu_freq"]
        return " ".join(
            [
                right_pad(f"{conditions1['cpu_freq']}GHz", 8),
                right_pad(metric, 8),
                right_pad(name, 8),
                f"p={scipy.stats.ttest_ind(a, b, equal_var=False)[1]:.4f}",
                f"mean/mean={a.mean() / b.mean():.2f}",
                f"stddev/stddev={a.std() / b.std():.2f}",
                f"med/med={np.median(a) / np.median(b):.2f}",
                *[
                    right_pad(summary_stats(series / 1e6, percentiles=[75]), 80)
                    for series in [a, b]
                ],
            ]
        )

    def summarize_diff(scheduler1, scheduler2):
        conditions1 = {
            frozendict(omit(condition, {"scheduler", "swap"}))
            for condition in scheduler_conditions[scheduler1]
        }
        conditions2 = {
            frozendict(omit(condition, {"scheduler", "swap"}))
            for condition in scheduler_conditions[scheduler2]
        }
        yield f"{scheduler1} / {scheduler2}"
        common_conditions = conditions1 & conditions2
        for condition in common_conditions:
            condition1 = frozendict(**condition, **scheduler1)
            condition2 = frozendict(**condition, **scheduler2)
            yield "\n".join(
                [
                    *[
                        inner_summarize_diff(condition1, condition2, name, metric_name)
                        for name in important_path_signatures.keys()
                        for metric_name in path_metric_names
                    ]
                ]
            )
        yield ""

    write_dir(
        {
            output_dir
            / "path_metrics": {
                "summary.txt": "\n".join(
                    "\n".join(
                        [
                            conditions2label(conditions),
                            *map(
                                second,
                                sorted(
                                    [
                                        (
                                            name,
                                            " ".join(
                                                [
                                                    right_pad(name, 6),
                                                    f"len: {len(data['latency'])}",
                                                    "latency:",
                                                    right_pad(
                                                        summary_stats(
                                                            np.concatenate(
                                                                data["latency"]
                                                            )
                                                            / 1e6,
                                                            percentiles=[75],
                                                        ),
                                                        58,
                                                    ),
                                                    "period:",
                                                    right_pad(
                                                        summary_stats(
                                                            np.concatenate(
                                                                data["period"]
                                                            )
                                                            / 1e6,
                                                            percentiles=[75],
                                                        ),
                                                        58,
                                                    ),
                                                    "rt:",
                                                    right_pad(
                                                        summary_stats(
                                                            np.concatenate(data["rt"])
                                                            / 1e6,
                                                            percentiles=[75],
                                                        ),
                                                        58,
                                                    ),
                                                ]
                                            ),
                                        )
                                        for name, data in name_data.items()
                                    ]
                                ),
                            ),
                            "",
                        ]
                    )
                    for conditions, name_data in proj_path_metrics.items()
                ),
                "convergence.txt": "\n".join(
                    [
                        "\n".join(
                            [
                                conditions2label(conditions),
                                "\n".join(
                                    [
                                        "\n".join(
                                            [
                                                f"{label} {metric}:",
                                                "".join(
                                                    [
                                                        f"{p:.4f} "
                                                        for p in sorted(
                                                            map(
                                                                second,
                                                                (
                                                                    scipy.stats.ks_2samp(
                                                                        a, b
                                                                    )
                                                                    for a, b in itertools.combinations(
                                                                        series, 2
                                                                    )
                                                                ),
                                                            )
                                                        )
                                                    ]
                                                ),
                                            ]
                                        )
                                        for label, data in label_data.items()
                                        for metric, series in data.items()
                                    ]
                                ),
                                "",
                            ]
                        )
                        for conditions, label_data in proj_path_metrics.items()
                    ]
                ),
                "comparison.txt": "\n".join(
                    [
                        "\n".join(summarize_diff(scheduler1, scheduler2))
                        for scheduler1, scheduler2 in itertools.product(
                            scheduler_conditions.keys(), repeat=2
                        )
                    ]
                ),
                "hist": {
                    conditions2label(conditions): {
                        name: {
                            f"{metric}.png": histogram(
                                ys=np.concatenate(series) / 1e6,
                                xlabel=f"{metric} (ms)",
                                title=f"{metric.capitalize()} of {name}",
                            )
                            for metric, series in data.items()
                        }
                        for name, data in name_data.items()
                    }
                    for conditions, name_data in proj_path_metrics.items()
                },
            },
        }
    )
    return {}


def print_trials(
    output_dir: Path, condition_trials: Mapping[frozendict, List[Path]], **kwargs
) -> Mapping[str, Any]:
    write_dir(
        {
            output_dir
            / "trials.txt": "\n".join(
                [
                    str(conditions) + "\n" + "\n".join(map(str, trials)) + "\n"
                    for conditions, trials in condition_trials.items()
                ]
            ),
        }
    )
    return {}


def print_and_ret(x):
    print(x)
    return x


def gen_aggregate(metrics_dirs: List[Path], chunk_size: int) -> Mapping[str, Any]:
    return combine(
        tqdm(
            (
                analyze_trials_projection(print_and_ret(metrics_group))
                for metrics_group in chunker(
                    sorted(metrics_dirs, key=lambda path: path.stat().st_mtime),
                    chunk_size,
                )
            ),
            total=len(metrics_dirs) // chunk_size + int(len(metrics_dirs) % chunk_size),
            unit="chunk",
            disable=True,
        )
    )


aggregate_fns = [
    compare_compute_times,
    compare_path_metrics,
    print_trials,
]


def analyze_trials(metrics_dirs: List[Path], output_dir: Path, chunk_size: int) -> None:
    aggregate = {"output_dir": output_dir, **gen_aggregate(metrics_dirs, chunk_size)}
    for aggregate_fn in aggregate_fns:
        aggregate.update(aggregate_fn(**aggregate))
