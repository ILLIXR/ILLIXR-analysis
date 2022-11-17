from collections import defaultdict
from typing import List
from pathlib import Path
from frozendict import frozendict
import yaml
import numpy as np
import matplotlib
import scipy.interpolate
from illixr.analysis.analyze_trials import conditions2label
from illixr.analysis.util import omit

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def analyze_trials3(candidates: List[Path], output_dir: Path, chunk_size: int) -> None:
    matplotlib.use("Qt5Agg", force=True)
    import matplotlib.pyplot as plt
    params = {
        "font.size": 18,
        "figure.figsize": [10, 7.2],
	"font.family": "MankSans-Medium",
	"grid.color": "#AAAAAA",
	"grid.linestyle": "--",
	"grid.linewidth": 1.5,
	"figure.autolayout": True,
    }
    plt.rcParams.update(params)
    condition_path_metric_dup = [
        (
            frozendict(yaml.load((candidate / "config.yaml").read_text(), Loader=yaml.SafeLoader)["conditions"]),
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

    condition_path_metric = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for condition, path_metrics in condition_path_metric_dup:
        for path, metrics in path_metrics.items():
            times = metrics["times"]
            for metric_name, series in metrics["metrics"].items():
                condition_path_metric[condition][path][metric_name].append(scipy.interpolate.interp1d(
                    times[0:len(series), -1],
                    series[0:len(times)],
                    fill_value=np.nan,
                    bounds_error=False,
                ))

    all_conditions = [
        condition
        for condition in condition_path_metric.keys()
        if condition["cpus"] != 10 and condition["scheduler"] != "manual"
    ]
    all_schedulers = ["default", "priority", "dynamic"]
    all_projected_conditions = sorted(list(set(
        frozendict(omit(condition, {"cpu_freq"})) for condition in all_conditions
    )), key=lambda condtion: all_schedulers.index(condition["scheduler"]))
    all_paths = list(condition_path_metric[all_conditions[0]].keys())
    all_metrics = [
        metric
        for metric in list(condition_path_metric[all_conditions[0]][all_paths[0]].keys())
        if metric != "times"
    ]
    all_cpu_freqs = list(set(condition["cpu_freq"] for condition in all_conditions))

    metric_translation = {
        "rt": "response time"
    }
    path_translation = {
        "CC": "Rotational motion-to-photon",
        "Render": "Translational motion-to-photon",
        "Cam": "Visual Inertial Odometry",
    }
    scheduler_translation = {
        "dynamic": "Catan",
        "default": "Static Rates",
        "priority": "Static Rates + Priorities",
    }
    scheduler_color = {
        "default": "#990000",
        "priority": "#0066CC",
        "dynamic": "#006400",
    }

    ts = np.linspace(times.min(), times.max(), 1000)
    tstep_seconds = (ts.max() - ts.min()) / len(ts) * 1e-9
    window_seconds = 1
    window_frames = int(window_seconds / tstep_seconds)
    interp = True

    for cpu_freq in all_cpu_freqs:
        for path in all_paths:
            for metric in all_metrics:
                plt.figure()
                title = f"{path_translation.get(path, path)} {metric_translation.get(metric, metric)}"
                print(title)
                plt.title(title)
                for projected_condition in all_projected_conditions:
                    condition = frozendict(cpu_freq=cpu_freq, **projected_condition)
                    fns = condition_path_metric[condition][path][metric]
                    ys = np.average([fn(ts) for fn in fns], axis=0)
                    chopped_ys = moving_average(ys, window_frames) if interp else ys
                    chopped_ts = ts[window_frames//2:-window_frames//2+1] if interp else ts
                    plt.plot(
                        chopped_ts / 1e9,
                        chopped_ys / 1e6,
                        label=f"{scheduler_translation.get(condition['scheduler'], condition['scheduler'])}",
                        color=scheduler_color.get(condition["scheduler"], "black"),
                    )
                    plt.xlabel("Time (seconds)")
                    plt.ylabel(f"{metric_translation.get(metric, metric).capitalize()} (ms)")
                plt.grid(True)
                plt.legend(loc="upper right")
                fname = output_dir / "path_metrics" / path / metric / str(cpu_freq) / "ts.png"
                fname.parent.mkdir(exist_ok=True, parents=True)
                plt.savefig(fname, bbox_inches="tight")
                plt.close()
