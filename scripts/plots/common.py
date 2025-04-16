import math

import glob
import matplotlib.pyplot as plt

import json
import os

from matplotlib.ticker import FormatStrFormatter

from colors import codec_color

MS_SSIM_NORM = lambda v: -10 * math.log10(1.0 - v)

results_paths = list()
dump_path = None


def add_results_path(path):
    global results_paths
    results_paths.append(path)


def set_dump_path(path):
    global dump_path
    try:
        os.mkdir(os.path.dirname(path))
    except FileExistsError:
        pass
    dump_path = path


def codec_style(codec_type):
    if codec_type == "inr":
        return {"marker": ".", "linestyle": "-", "linewidth": 0.75}
    elif codec_type == "traditional":
        return {"marker": "", "linestyle": "-", "linewidth": 0.75}
    elif codec_type == "autoencoder":
        return {"marker": "", "linestyle": "--", "linewidth": 0.75}


fig = None
axes = None


def init(rows=1, cols=2):
    global fig
    global axes

    plt.rc("font", size=8)

    fig, axes = plt.subplots(rows, cols)
    fig.tight_layout()


legend_handles = list()
legend_labels = list()


def plot_metric(
    row,
    col,
    title,
    metric_name,
    display_name,
    xlim,
    ylim,
    xdigits=2,
    ydigits=2,
    bpp_key="bpp",
    normalizer=None,
):
    try:
        if hasattr(axes, "ndim") and axes.ndim == 2:
            ax = axes[row][col]
        else:
            ax = axes[col]
    except TypeError:
        ax = axes

    ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{xdigits}f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{ydigits}f"))

    for result_path in results_paths:
        for full_path in glob.glob(result_path):
            try:
                stats = json.load(open(full_path))
                results = stats["results"]
                codec_name = stats["name"]

                values = results[metric_name]

                if normalizer:
                    values = [normalizer(value) for value in values]

                (line,) = ax.plot(
                    results[bpp_key],
                    values,
                    color=codec_color(codec_name),
                    **codec_style(stats["type"]),
                )
                line.set_label(codec_name)

                if codec_name not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(codec_name)
            except Exception as e:
                print(f"Cannot load {full_path}: {e}")

    # _, xend = ax.get_xlim()
    # _, yend = ax.get_ylim()
    # ax.xaxis.set_ticks(numpy.arange(0.0, xend, 1.0))
    # ax.yaxis.set_ticks(numpy.arange(0.0, yend, 0.1))

    ax.set_xlabel("bits per pixel (bpp)")
    ax.set_ylabel(display_name)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)

    ax.legend(loc="lower right", fontsize=6)

    ax.grid()


# CelebA
# plot_metric(0, "psnr", "PSNR", [0.0, 3.0], [25.0, 40.0])
# plot_metric(1, "ms-ssim", "MS-SSIM", [0.0, 3.0], [0.95, 1.0])


def clear_results():
    global dump_path
    results_paths.clear()
    dump_path = None


def clear_fig():
    global fig
    global axes
    legend_handles.clear()
    legend_labels.clear()
    fig = None
    axes = None
    plt.clf()


def save(height=2.5, width=6.5):
    global dump_path
    fig.tight_layout()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    # fig.legend(legend_handles, legend_labels, loc='center', ncol=3, bbox_to_anchor=(0.5, legend_distance))
    plt.savefig(dump_path, bbox_inches="tight")
