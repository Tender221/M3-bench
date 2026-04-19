#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


BOOT_LINE_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+\[([+-]?\d+(?:\.\d+)?),([+-]?\d+(?:\.\d+)?)\]\s+p=([01]?(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+\[([+-]?\d+(?:\.\d+)?),([+-]?\d+(?:\.\d+)?)\]\s+p=([01]?(?:\.\d+)?)\s*$"
)


def load_summary_csv(path: str, metric: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grid_from_summary(rows: List[Dict[str, str]], metric: str) -> Tuple[np.ndarray, List[int], List[int], float]:
    mixed_rows = [row for row in rows if row.get("mode") == "mixed"]
    img_levels = sorted({int(row["cell"].split("_")[0]) for row in mixed_rows})
    txt_levels = sorted({int(row["cell"].split("_")[1]) for row in mixed_rows})
    grid = np.full((len(img_levels), len(txt_levels)), np.nan, dtype=float)
    clean_score = math.nan

    for row in rows:
        cell = row["cell"]
        if cell == "0_0":
            try:
                clean_score = float(row.get(metric, "") or "nan")
            except ValueError:
                clean_score = math.nan
        if row.get("mode") != "mixed":
            continue
        i, j = [int(part) for part in cell.split("_")]
        try:
            value = float(row.get(metric, "") or "nan")
        except ValueError:
            value = math.nan
        grid[img_levels.index(i), txt_levels.index(j)] = value

    return grid, img_levels, txt_levels, clean_score


def parse_bootstrap_txt(path: str) -> List[Dict[str, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = BOOT_LINE_RE.match(line.rstrip("\n"))
            if not match:
                continue
            cell_str = match.group(1)
            if len(cell_str) < 2:
                continue
            i = int(cell_str[:-1]) if len(cell_str) > 2 else int(cell_str[0])
            j = int(cell_str[-1])
            rows.append(
                {
                    "i": i,
                    "j": j,
                    "excess_mean": float(match.group(8)),
                    "interact_mean": float(match.group(12)),
                }
            )
    return rows


def bootstrap_grid(rows: List[Dict[str, float]], key: str) -> Tuple[np.ndarray, List[int], List[int]]:
    img_levels = sorted({int(row["i"]) for row in rows})
    txt_levels = sorted({int(row["j"]) for row in rows})
    grid = np.full((len(img_levels), len(txt_levels)), np.nan, dtype=float)
    for row in rows:
        i = img_levels.index(int(row["i"]))
        j = txt_levels.index(int(row["j"]))
        grid[i, j] = float(row[key])
    return grid, img_levels, txt_levels


def draw_grid(ax, grid: np.ndarray, img_levels: List[int], txt_levels: List[int], title: str, cmap: str, symmetric: bool = False) -> None:
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "Not available", ha="center", va="center")
        ax.set_axis_off()
        return

    if symmetric:
        vmax = max(np.max(np.abs(finite)), 1e-6)
        im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1e-6
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Text severity")
    ax.set_ylabel("Image severity")
    ax.set_xticks(range(len(txt_levels)))
    ax.set_xticklabels(txt_levels)
    ax.set_yticks(range(len(img_levels)))
    ax.set_yticklabels(img_levels)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8, color="black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def draw_summary_text(ax, metric: str, clean_score: float, summary_json: Optional[str]) -> None:
    ax.set_axis_off()
    lines = [f"Metric: {metric}"]
    if not math.isnan(clean_score):
        lines.append(f"Clean score: {clean_score:.4f}")

    if summary_json and Path(summary_json).exists():
        with open(summary_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        summary = payload.get("summary", {})
        for key in ["AMR", "NDS", "ANDS", "SNA", "WNA", "WND", "AIS", "SIA", "WIIA"]:
            if key in summary:
                lines.append(f"{key}: {summary[key]:.4f}")
        hotspot = summary.get("worst_excess_cell")
        if hotspot:
            lines.append(f"Worst excess cell: {hotspot}")
        hotspot = summary.get("worst_interact_cell")
        if hotspot:
            lines.append(f"Worst interaction cell: {hotspot}")

    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--metric", default="f1")
    ap.add_argument("--bootstrap_txt", default=None)
    ap.add_argument("--summary_json", default=None)
    args = ap.parse_args()

    summary_rows = load_summary_csv(args.summary_csv, args.metric)
    mixed_grid, img_levels, txt_levels, clean_score = grid_from_summary(summary_rows, args.metric)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    draw_summary_text(axes[0, 0], args.metric, clean_score, args.summary_json)
    draw_grid(axes[0, 1], mixed_grid, img_levels, txt_levels, f"Mixed {args.metric.upper()} Grid", "YlGnBu", symmetric=False)

    if args.bootstrap_txt and Path(args.bootstrap_txt).exists():
        boot_rows = parse_bootstrap_txt(args.bootstrap_txt)
        excess_grid, ex_img_levels, ex_txt_levels = bootstrap_grid(boot_rows, "excess_mean")
        interact_grid, in_img_levels, in_txt_levels = bootstrap_grid(boot_rows, "interact_mean")
        draw_grid(axes[1, 0], excess_grid, ex_img_levels, ex_txt_levels, "Excess Heatmap", "coolwarm", symmetric=True)
        draw_grid(axes[1, 1], interact_grid, in_img_levels, in_txt_levels, "Interaction Heatmap", "coolwarm", symmetric=True)
    else:
        axes[1, 0].text(0.5, 0.5, "Bootstrap summary not available", ha="center", va="center")
        axes[1, 0].set_axis_off()
        axes[1, 1].set_axis_off()

    fig.suptitle("Evaluation Summary", fontsize=16)
    fig.tight_layout()
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
