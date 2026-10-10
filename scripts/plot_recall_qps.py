#!/usr/bin/env python3
"""Plot recall@k against single-thread QPS from a benchmark_ann.py JSON report."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON written by benchmark_ann.py --output")
    parser.add_argument("-o", "--output", type=Path, required=True, help="image path (.png, .svg)")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text(encoding="utf-8"))
    engines = report.get("engines") or {"minivec": {"configurations": report["configurations"]}}
    dataset = report["dataset"]
    env = report["environment"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, result in engines.items():
        points = sorted(result["configurations"], key=lambda c: c["recall_at_k"])
        ax.plot(
            [c["recall_at_k"] for c in points],
            [c["qps_single_thread"] for c in points],
            marker="o",
            label=name,
        )
        for c in points:
            ax.annotate(
                f"ef={c['ef_search']}", (c["recall_at_k"], c["qps_single_thread"]),
                textcoords="offset points", xytext=(4, 4), fontsize=7,
            )
    ax.set_yscale("log")
    ax.set_xlabel(f"Recall@{dataset['k']}")
    ax.set_ylabel("Queries per second (1 thread, log scale)")
    ax.set_title(
        f"{dataset.get('name', 'dataset')}: {dataset['count']:,} vectors, "
        f"M={report['index']['M']}, efC={report['index']['ef_construction']}\n"
        f"{env.get('processor', '')} · {env.get('platform', '')}",
        fontsize=9,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
