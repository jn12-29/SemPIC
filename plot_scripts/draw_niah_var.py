import argparse
import json
import logging
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from sempic.utils.runtime import RuntimeContext
from sempic.utils.run_storage import allocate_run_dir

LOGGER = logging.getLogger("sempic.plot_scripts.draw_niah_var")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot NIAH adapter-length results from an evaluation result directory."
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing *_result.json files named <doc_len>_<header>_<trailer>_<epoch>_result.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = allocate_run_dir("./logs/draw_niah_var", None)
    with RuntimeContext(
        entrypoint="draw_niah_var",
        run_dir=run_dir,
        config_file=None,
        resolved_config=None,
        config_snapshot_name=None,
        cli_args=vars(args).copy(),
    ):
        records = []
        for fname in os.listdir(args.results_dir):
            if not fname.endswith("_result.json"):
                continue
            parts = fname.replace("_result.json", "").split("_")
            if len(parts) != 4:
                continue
            doc_len, header, trailer, epoch = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            with open(os.path.join(args.results_dir, fname)) as f:
                data = json.load(f)
            result = data["result"]
            records.append({
                "doc_len": doc_len,
                "header": header,
                "trailer": trailer,
                "packet_len": header + trailer,
                "epoch": epoch,
                "f1": result["f1"],
                "overhead": result["num_wrapped_tokens"] / result["num_orig_tokens"] - 1,
            })

        doc_lens = sorted(set(r["doc_len"] for r in records))
        packet_lens = sorted(set(r["packet_len"] for r in records))
        epochs = sorted(set(r["epoch"] for r in records))

        # Match color cycle from reference script
        colors = {8: "C0", 16: "C1", 32: "C2"}
        labels = {8: "4+4", 16: "8+8", 32: "16+16"}
        markers = {8: "o", 16: "s", 32: "^"}

        # overhead is epoch-independent
        overhead = defaultdict(dict)
        for r in records:
            overhead[r["doc_len"]][r["packet_len"]] = r["overhead"]

        num_rows = 2
        num_cols = len(doc_lens)

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(3 * num_cols, 3 * num_rows),
            squeeze=False,
            sharey="row",
        )

        for col, doc_len in enumerate(doc_lens):
            ax_f1 = axes[0, col]
            ax_f1.set_title(f"document len. = {doc_len}", fontsize=14, pad=10)

            for plen in packet_lens:
                subset = sorted(
                    [r for r in records if r["doc_len"] == doc_len and r["packet_len"] == plen],
                    key=lambda x: x["epoch"],
                )
                if not subset:
                    continue
                xs = [r["epoch"] for r in subset]
                ys = [r["f1"] for r in subset]
                ax_f1.plot(
                    xs,
                    ys,
                    color=colors[plen],
                    linestyle="-",
                    marker=markers[plen],
                    markersize=5,
                    linewidth=1.5,
                    label=labels[plen],
                )

            ax_f1.set_ylim(bottom=0, top=0.80)
            ax_f1.yaxis.set_major_locator(MultipleLocator(0.2))
            ax_f1.set_xticks(epochs[::2])
            ax_f1.grid(True, linestyle=":", alpha=0.6)
            ax_f1.set_xlabel("Epoch", fontsize=11)

            if col == 0:
                ax_f1.set_ylabel("F1 Score", fontsize=11)

            ax_oh = axes[1, col]

            x = np.arange(len(packet_lens))
            width = 0.5
            oh_vals = [overhead[doc_len].get(p, 0) for p in packet_lens]
            oh_max = 0.11

            bars = ax_oh.bar(
                x,
                oh_vals,
                width=width,
                color=[colors[p] for p in packet_lens],
                edgecolor="black",
                linewidth=0.6,
            )

            for bar, v in zip(bars, oh_vals):
                ax_oh.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + oh_max * 0.02,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            ax_oh.set_ylim(0, oh_max * 1.3)
            ax_oh.set_xticks(x)
            ax_oh.set_xticklabels([labels[p] for p in packet_lens], fontsize=9)
            ax_oh.grid(True, linestyle=":", alpha=0.6, axis="y")
            ax_oh.set_xlabel("Adapter Length", fontsize=11)

            if col == 0:
                ax_oh.set_ylabel("Storage Overhead", fontsize=11)

        handles, lbls = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            lbls,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=len(packet_lens),
            frameon=True,
            fontsize=12,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig("ablation_niah.pdf", bbox_inches="tight", pad_inches=0.2)
        plt.savefig("ablation_niah.svg", dpi=150, bbox_inches="tight", pad_inches=0.2)
        LOGGER.info("Saved to ablation_niah.pdf and ablation_niah.svg")


if __name__ == "__main__":
    main()
