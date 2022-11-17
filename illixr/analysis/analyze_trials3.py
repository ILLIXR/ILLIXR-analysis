from collections import defaultdict
import itertools
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import scipy.interpolate
import scipy.ndimage
from tabulate import tabulate
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import charmonium.time_block as ch_time_block
from frozendict import frozendict
from illixr.analysis.analyze_trials import conditions2label

from illixr.analysis.util import omit


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


moving_average2 = lambda xs, window: scipy.ndimage.gaussian_filter(xs, window, order=0, mode='reflect', truncate=4.0)


def analyze_trials3(candidates: List[Path], output_dir: Path, chunk_size: int) -> None:
    # matplotlib.use("Qt5Agg", force=True)

    params = {
        "font.size": 18,
        "figure.figsize": [10, 7.2],
        "font.family": "MankSans-Medium",
        "grid.color": "#AAAAAA",
        "grid.linestyle": "--",
        "grid.linewidth": 1.5,
        "figure.autolayout": True,
    }
    metric_translation = {"rt": "response time"}
    chain_translation = {
        "CC": "Rot MTP",
        "Render": "Trans MTP",
        "Cam": "VIO chain",
    }
    scheduler_translation = {
        "dynamic": "Catan",
        "default": "Static Rates",
        "priority": "Static Rates + Prio",
    }
    scheduler_color = {
        "default": "#990000",
        "priority": "#0066CC",
        "dynamic": "#006400",
        # "dynamic": "#4CDA1D",
        # "dynamic": "#000064",
    }

    plt.rcParams.update(params)
    condition_chain_metric_dup = [
        (
            frozendict(
                yaml.load(
                    (candidate / "config.yaml").read_text(), Loader=yaml.SafeLoader
                )["conditions"]
            ),
            {
                path.stem: {
                    "times": np.loadtxt(path / "times.npy"),
                    "metrics": {
                        metric.stem: np.loadtxt(metric / "data.npy")
                        for metric in path.iterdir()
                        if metric.is_dir()
                    },
                }
                for path in (candidate / "chains").iterdir()
                if path.is_dir()
            },
        )
        for candidate in candidates
    ]

    with ch_time_block.ctx("trials.txt", print_start=True):
        trials_dict = defaultdict(list)
        for candidate in candidates:
            if (candidate / "chains/CC").exists():
                conditions = frozendict(
                    yaml.load(
                        (candidate / "config.yaml").read_text(), Loader=yaml.SafeLoader
                    )["conditions"]
                )
                chain_dict = {
                    chain.name: (np.loadtxt(chain / "times.npy"), np.loadtxt(chain / "rt/data.npy"))
                    for chain in (candidate / "chains").iterdir()
                    if chain.is_dir()
                }
                trials_dict[conditions].append((candidate, chain_dict))

        chains = ["CC", "Render", "Cam"]
        (output_dir / "trials.txt").write_text(
            "\n\n".join((
                f"{conditions2label(conditions)}: {len(data)}\n" + "\n".join(
                    f"{path} rt of " + " ".join(f"{np.average(chain_dict[chain][1], weights=chain_dict[chain][0][:len(chain_dict[chain][1]), -1])/1e6:.1f} +/- {np.std(chain_dict[chain][1])/1e6:.1f}" for chain in chains)
                    for (path, chain_dict) in data
                )
                for conditions, data in trials_dict.items()
            )) + "\n"
        )

        summary_table = [
            [conditions2label(condition)] + [
                (lambda aggregate: f"{np.mean(aggregate)/1e6:.1f} +/- {np.std(aggregate)/1e6:.1f}")([
                    np.average(chain_dict[chain][1], weights=chain_dict[chain][0][:len(chain_dict[chain][1]), -1])
                    for _, chain_dict in data
                ])
                for chain in chains
            ]
            for condition, data in trials_dict.items()
        ]
        (output_dir / "summary.txt").write_text("\n\n".join([
            tabulate(summary_table, headers=[""] + chains, tablefmt="fancygrid"),
            tabulate(summary_table, headers=[""] + chains, tablefmt="latex_booktabs"),
        ]) + "\n")

        schedulers = ["default", "priority", "dynamic"]
        scheduler_labels = [scheduler_translation[scheduler] for scheduler in schedulers]
        conditions = [
            [
                condition
                for condition in trials_dict.keys()
                if condition["scheduler"] == scheduler
            ][0]
            for scheduler in schedulers
        ]

        max_ys = [
            np.max([
                np.mean([
                    np.average(chain_dict[chain][1], weights=chain_dict[chain][0][:len(chain_dict[chain][1]), -1])
                    for _, chain_dict in trials_dict[condition]
                ])
                for condition in conditions
            ])
            for chain in chains
        ]

        grouped_bar_chart2(
            labels=(
                [chain_translation[chain] for chain in chains],
                scheduler_labels,
            ),
            yss=[
                [
                    np.mean([
                        np.average(chain_dict[chain][1], weights=chain_dict[chain][0][:len(chain_dict[chain][1]), -1])
                        for _, chain_dict in trials_dict[condition]
                    ]) / (max_y if False else 1) / 1e6
                    for condition in conditions
                ]
                for chain, max_y in zip(chains, max_ys)
            ],
            errss=[
                [
                    np.std([
                        np.average(chain_dict[chain][1], weights=chain_dict[chain][0][:len(chain_dict[chain][1]), -1])
                        for _, chain_dict in trials_dict[condition]
                    ]) / (max_y if False else 1) / 1e6
                    for condition in conditions
                ]
                for chain, max_y in zip(chains, max_ys)
            ],
            colors=[
                scheduler_color[scheduler]
                for scheduler in schedulers
            ],
            output=output_dir / Path("summary.png"),
            ylabel="Response time (ms)",
        )

        # import sys
        # sys.exit(0)

    condition_chain_metric = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for condition, chain_metrics in condition_chain_metric_dup:
        for path, metrics in chain_metrics.items():
            times = metrics["times"]
            for metric_name, series in metrics["metrics"].items():
                condition_chain_metric[condition][path][metric_name].append(
                    scipy.interpolate.interp1d(
                        times[0 : len(series), -1],
                        series[0 : len(times)],
                        fill_value=np.nan,
                        bounds_error=False,
                    )
                )

    all_conditions = [
        condition
        for condition in condition_chain_metric.keys()
        if condition["cpus"] != 10 and condition["scheduler"] != "manual"
    ]
    all_schedulers = ["default", "priority", "dynamic"]
    all_projected_conditions = sorted(
        list(
            set(
                frozendict(omit(condition, {"cpu_freq"}))
                for condition in all_conditions
            )
        ),
        key=lambda condtion: all_schedulers.index(condition["scheduler"]),
    )
    all_paths = list(condition_chain_metric[all_conditions[0]].keys())
    all_metrics = [
        metric
        for metric in list(
            condition_chain_metric[all_conditions[0]][all_paths[0]].keys()
        )
        if metric != "times"
    ]
    all_cpu_freqs = list(set(condition["cpu_freq"] for condition in all_conditions))

    ts = np.linspace(times.min(), times.max(), 1000)
    tstep_seconds = (ts.max() - ts.min()) / len(ts) * 1e-9
    sigma = 50
    sigma2 = 50
    stride = 10
    window_seconds = 1
    window_frames = int(window_seconds / tstep_seconds)
    interp = True
    outlier_prec = 10
    variance_scale = 0.1

    for cpu_freq in all_cpu_freqs:
        for path in all_paths:
            for metric in all_metrics:
                plt.figure()
                title = f"{chain_translation.get(path, path)} {metric_translation.get(metric, metric)} (average over runs of each scheduler)"
                print(title)
                plt.title(title)
                for projected_condition in all_projected_conditions:
                    condition = frozendict(cpu_freq=cpu_freq, **projected_condition)

                    fns = condition_chain_metric[condition][path][metric]
                    ys = np.average([fn(ts) for fn in fns], axis=0)
                    chopped_ys = moving_average(ys, window_frames) if interp else ys
                    chopped_ts = ts[window_frames//2:-window_frames//2+1] if interp else ts

                    plt.plot(
                        chopped_ts / 1e9,
                        chopped_ys / 1e6,
                        # label=conditions2label(condition),
                        label=f"{scheduler_translation.get(condition['scheduler'], condition['scheduler'])}",
                        color=scheduler_color.get((condition["scheduler"], condition["swap"]), "black"),
                    )
                    plt.xlabel("Time (seconds)")
                    plt.ylabel(
                        f"{metric_translation.get(metric, metric).capitalize()} (ms)"
                    )
                plt.grid(True)
                plt.legend(loc="upper right")
                fname = (
                    output_dir
                    / "path_metrics"
                    / path
                    / metric
                    / str(cpu_freq)
                    / "ts_avg.png"
                )
                fname.parent.mkdir(exist_ok=True, parents=True)
                plt.savefig(fname, bbox_inches="tight")
                plt.close()


    for cpu_freq in all_cpu_freqs:
        for path in all_paths:
            for metric in all_metrics:
                plt.figure()
                title = f"{chain_translation.get(path, path)} {metric_translation.get(metric, metric)} (individual runs of each scheduler)"
                print(title)
                plt.title(title)
                for projected_condition in all_projected_conditions:
                    condition = frozendict(cpu_freq=cpu_freq, **projected_condition)

                    chain_metricss = [
                        chain_metrics
                        for this_condition, chain_metrics in condition_chain_metric_dup
                        if this_condition == condition
                    ]

                    ts_yss = [
                        (chain_metrics[path]["times"], chain_metrics[path]["metrics"][metric])
                        for chain_metrics in chain_metricss
                        if path in chain_metrics
                    ]

                    for ts, ys in ts_yss:
                        ts = ts[0:len(ys), -1]
                        assert ts.max() > 75e9
                        ys = moving_average2(ys[0:len(ts)], sigma)
    
                        plt.plot(
                            ts[::stride] / 1e9,
                            ys[::stride] / 1e6,
                            # label=conditions2label(condition),
                            label=f"{scheduler_translation.get(condition['scheduler'], condition['scheduler'])}",
                            color=scheduler_color.get((condition["scheduler"], condition["swap"]), "black"),
                            alpha=0.4,
                        )

                    # ys = np.zeros(len(ts))
                    # ys[len(ts) // 2] = 5
                    # ys = moving_average2(ys, sigma)
                    # plt.plot(ts / 1e9, ys, color="black", label="window", alpha=0.1)

                    plt.xlabel("Time (seconds)")
                    plt.ylabel(
                        f"{metric_translation.get(metric, metric).capitalize()} (ms)"
                    )
                plt.grid(True)
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc="upper right")
                fname = (
                    output_dir
                    / "path_metrics"
                    / path
                    / metric
                    / str(cpu_freq)
                    / "ts_individual.png"
                )
                fname.parent.mkdir(exist_ok=True, parents=True)
                plt.savefig(fname, bbox_inches="tight")
                plt.close()

    for cpu_freq in all_cpu_freqs:
        for path in all_paths:
            for metric in all_metrics:
                plt.figure()
                title = f"{chain_translation.get(path, path)} {metric_translation.get(metric, metric)} (normal range of each scheduler)"
                print(title)
                plt.title(title)
                for projected_condition in all_projected_conditions:
                    condition = frozendict(cpu_freq=cpu_freq, **projected_condition)

                    chain_metricss = [
                        chain_metrics
                        for this_condition, chain_metrics in condition_chain_metric_dup
                        if this_condition == condition
                    ]

                    fns = []
                    for chain_metrics in chain_metricss:
                        if path in chain_metrics:
                            this_ts, ys = chain_metrics[path]["times"][:, -1], chain_metrics[path]["metrics"][metric]
                            this_ts = this_ts[:len(ys)]
                            fns.append(scipy.interpolate.interp1d(
                                this_ts,
                                ys, 
                                fill_value=np.nan,
                                bounds_error=False,
                            ))

                    yss = np.array([fn(ts) for fn in fns])
                    mean_ys = np.mean(yss, axis=0)
                    stddev_ys = np.std(yss, axis=0)
                    mean_ys = moving_average2(mean_ys, sigma2)
                    stddev_ys = moving_average2(stddev_ys, sigma2)

                    plt.fill_between(
                        ts[::stride] / 1e9,
                        (mean_ys - stddev_ys * variance_scale)[::stride] / 1e6,
                        (mean_ys + stddev_ys * variance_scale)[::stride] / 1e6,
                        # label=conditions2label(condition),
                        label=f"{scheduler_translation.get(condition['scheduler'], condition['scheduler'])}",
                        color=scheduler_color.get((condition["scheduler"], condition["swap"]), "black"),
                        alpha=0.4,
                    )

                    # ys = np.zeros(len(ts))
                    # ys[len(ts) // 2] = 5
                    # ys = moving_average2(ys, sigma)
                    # plt.plot(ts / 1e9, ys, color="black", label="window", alpha=0.1)

                    plt.xlabel("Time (seconds)")
                    plt.ylabel(
                        f"{metric_translation.get(metric, metric).capitalize()} (ms)"
                    )
                plt.grid(True)
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(), loc="upper right")
                fname = (
                    output_dir
                    / "path_metrics"
                    / path
                    / metric
                    / str(cpu_freq)
                    / "ts_ranges.png"
                )
                fname.parent.mkdir(exist_ok=True, parents=True)
                plt.savefig(fname, bbox_inches="tight")
                plt.close()

    for cpu_freq in all_cpu_freqs:
        for path in all_paths:
            for metric in all_metrics:
                plt.figure()
                title = f"{chain_translation.get(path, path)} {metric_translation.get(metric, metric)} (single run of each scheduler)"
                print(title)
                plt.title(title)
                for projected_condition in all_projected_conditions:
                    condition = frozendict(cpu_freq=cpu_freq, **projected_condition)

                    chain_metricss = [
                        chain_metrics
                        for this_condition, chain_metrics in condition_chain_metric_dup
                        if this_condition == condition
                    ]

                    ts_yss = [
                        (chain_metrics[path]["times"][:, -1], chain_metrics[path]["metrics"][metric])
                        for chain_metrics in chain_metricss
                        if path in chain_metrics
                    ]

                    ts, ys = sorted(ts_yss, key=lambda ts_ys: ts_ys[1].mean())[len(ts_yss) // 2]
                    tlimit_sec = 80
                    ts = ts[ts < tlimit_sec * 1e9]

                    ts = ts[:len(ys)]
                    ys = ys[:len(ts)]

                    low_y = np.percentile(ys, outlier_prec)
                    high_y = np.percentile(ys, 100-outlier_prec)
                    nonoutlier_mask = (low_y < ys) & (ys < high_y)
                    ts = ts[nonoutlier_mask]
                    ys = ys[nonoutlier_mask]

                    plt.plot(
                        ts / 1e9,
                        ys / 1e6,
                        # label=conditions2label(condition),
                        label=f"{scheduler_translation.get(condition['scheduler'], condition['scheduler'])}",
                        color=scheduler_color.get((condition["scheduler"], condition["swap"]), "black"),
                    )
                    plt.xlabel("Time (seconds)")
                    plt.ylabel(
                        f"{metric_translation.get(metric, metric).capitalize()} (ms)"
                    )
                plt.grid(True)
                plt.legend(loc="upper right")
                fname = (
                    output_dir
                    / "path_metrics"
                    / path
                    / metric
                    / str(cpu_freq)
                    / "ts_single.png"
                )
                fname.parent.mkdir(exist_ok=True, parents=True)
                plt.savefig(fname, bbox_inches="tight")
                plt.close()

    for cpu_freq in all_cpu_freqs:
        for path in all_paths:
            for metric in all_metrics:
                plt.figure()
                title = f"{chain_translation.get(path, path)} {metric_translation.get(metric, metric)} (median buckets of each scheduler)"
                print(title)
                plt.title(title)
                for projected_condition in all_projected_conditions:
                    condition = frozendict(cpu_freq=cpu_freq, **projected_condition)

                    ts = np.linspace(times.min(), times.max(), 50)
                    t_buckets = [[] for t in ts]
                    chain_metricss = [
                        chain_metrics
                        for this_condition, chain_metrics in condition_chain_metric_dup
                        if this_condition == condition
                    ]

                    for chain_metrics in chain_metricss:
                        if path in chain_metrics:
                            this_ts, ys = chain_metrics[path]["times"][:, -1], chain_metrics[path]["metrics"][metric]
                            for t, y in zip(this_ts, ys):
                                t_bucket = np.searchsorted(ts, t)
                                if 0 <= t_bucket < len(t_buckets):
                                    t_buckets[t_bucket].append(y)

                    ys = np.array([np.median(bucket) for bucket in t_buckets])


                    tlimit_sec = 80
                    ts = ts[ts < tlimit_sec * 1e9]

                    ts = ts[:len(ys)]
                    ys = ys[:len(ts)]

                    plt.plot(
                        ts / 1e9,
                        ys / 1e6,
                        # label=conditions2label(condition),
                        label=f"{scheduler_translation.get(condition['scheduler'], condition['scheduler'])}",
                        color=scheduler_color.get((condition["scheduler"], condition["swap"]), "black"),
                    )
                    plt.xlabel("Time (seconds)")
                    plt.ylabel(
                        f"{metric_translation.get(metric, metric).capitalize()} (ms)"
                    )
                plt.grid(True)
                plt.legend(loc="upper right")
                fname = (
                    output_dir
                    / "path_metrics"
                    / path
                    / metric
                    / str(cpu_freq)
                    / "ts_median.png"
                )
                fname.parent.mkdir(exist_ok=True, parents=True)
                plt.savefig(fname, bbox_inches="tight")
                plt.close()

def grouped_bar_chart(labels: Tuple[List[str], List[str]], yss, errss, colors: List[str], output: Path) -> None:
    assert len(yss) == len(labels[0])
    assert all(len(ys) == len(labels[1]) for ys in yss)
    assert len(errss) == len(labels[0])
    assert all(len(errs) == len(labels[1]) for errs in errss)
    assert len(colors) == len(labels[1])

    bar_width = 1
    margin = 1
    group_width = bar_width * len(labels[1]) + margin

    for group_no, group_label, ys, errs in zip(range(len(labels[0])), labels[0], yss, errss):
        for bar_no, bar_label, y, err, color in zip(range(len(labels[1])), labels[1], ys, errs, colors):
            x = group_no * group_width + bar_no * bar_width
            plt.bar(x, y, width=bar_width, align="edge", yerr=err, color=color)

    plt.xticks(np.arange(len(labels[0])) * group_width + (group_width - margin) / 2, labels[0])

    plt.legend(handles=[
        mpatches.Patch(color=color, label=bar_label)
        for bar_label, color in zip(labels[1], colors)
    ])

    plt.savefig(output)

def grouped_bar_chart2(labels: Tuple[List[str], List[str]], yss, errss, colors: List[str], output: Path, ylabel: str) -> None:
    assert len(yss) == len(labels[0])
    assert all(len(ys) == len(labels[1]) for ys in yss)
    assert len(errss) == len(labels[0])
    assert all(len(errs) == len(labels[1]) for errs in errss)
    assert len(colors) == len(labels[1])

    bar_width = 1
    margin = 1

    axs = []
    for group_no, group_label, ys, errs in zip(range(len(labels[0])), labels[0], yss, errss):
        ax = plt.subplot(1, len(labels[0]), group_no+1)
        for bar_no, bar_label, y, err, color in zip(range(len(labels[1])), labels[1], ys, errs, colors):
            ax.bar(bar_no * bar_width, y, width=bar_width, align="edge", yerr=err, color=color, capsize=10)
        ax.set_title(group_label)
        ax.set_ylabel(ylabel)
        axs.append(ax)
        ax.set_xticks([])

    # axs[1].legend(
    #     handles=[
    #         mpatches.Patch(color=color, label=bar_label)
    #         for bar_label, color in zip(labels[1], colors)
    #     ],
    #     # ncol=len(labels[1]),
    #     loc="lower center",
    # )

    plt.savefig(output)
