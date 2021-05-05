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
