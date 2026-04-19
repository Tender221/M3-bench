#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


LINE_RE = re.compile(
    r"^\s*(\d{2})\s+(\d+)\s+"
    r"([+-]?\d*\.\d+)\s+([+-]?\d*\.\d+)\s+([+-]?\d*\.\d+)\s+([+-]?\d*\.\d+)\s+([+-]?\d*\.\d+)\s+"
    r"([+-]?\d*\.\d+)\s+\[([+-]?\d*\.\d+),([+-]?\d*\.\d+)\]\s+p=([01]?\.\d+)\s+"
    r"([+-]?\d*\.\d+)\s+\[([+-]?\d*\.\d+),([+-]?\d*\.\d+)\]\s+p=([01]?\.\d+)\s*$"
)

# groups:
# 1 cell
# 2 n
# 3 S_clean
# 4 S_img
# 5 S_txt
# 6 S_mixed
# 7 S_mul
# 8 excess_mean
# 9 excess_lo
# 10 excess_hi
# 11 excess_p
# 12 interact_mean
# 13 interact_lo
# 14 interact_hi
# 15 interact_p


def parse_bootstrap_txt(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue

            cell = m.group(1)
            i, j = int(cell[0]), int(cell[1])

            rows.append(
                {
                    "cell": cell,
                    "img_sev": i,
                    "txt_sev": j,
                    "n": int(m.group(2)),
                    "S_clean": float(m.group(3)),
                    "S_img": float(m.group(4)),
                    "S_txt": float(m.group(5)),
                    "S_mixed": float(m.group(6)),
                    "S_mul": float(m.group(7)),
                    "excess_mean": float(m.group(8)),
                    "excess_lo": float(m.group(9)),
                    "excess_hi": float(m.group(10)),
                    "excess_p": float(m.group(11)),
                    "interact_mean": float(m.group(12)),
                    "interact_lo": float(m.group(13)),
                    "interact_hi": float(m.group(14)),
                    "interact_p": float(m.group(15)),
                }
            )

    if not rows:
        raise ValueError(f"No valid result rows parsed from: {path}")

    return rows


def build_grid(rows: List[Dict], key: str) -> Tuple[np.ndarray, List[int], List[int], np.ndarray]:
    img_levels = sorted({r["img_sev"] for r in rows})
    txt_levels = sorted({r["txt_sev"] for r in rows})

    grid = np.full((len(img_levels), len(txt_levels)), np.nan, dtype=float)
    signif = np.zeros((len(img_levels), len(txt_levels)), dtype=bool)

    img_idx = {v: k for k, v in enumerate(img_levels)}
    txt_idx = {v: k for k, v in enumerate(txt_levels)}

    for r in rows:
        ii = img_idx[r["img_sev"]]
        jj = txt_idx[r["txt_sev"]]
        grid[ii, jj] = r[key]

        if key == "interact_mean":
            signif[ii, jj] = (r["interact_lo"] > 0) or (r["interact_hi"] < 0)
        elif key == "excess_mean":
            signif[ii, jj] = (r["excess_lo"] > 0) or (r["excess_hi"] < 0)

    return grid, img_levels, txt_levels, signif


def pretty_title(metric_name: str, stem: str) -> str:
    stem = stem.replace("_", " ")
    if metric_name == "interact_mean":
        return f"Interaction Heatmap — {stem}"
    return f"Excess Heatmap — {stem}"


def draw_heatmap(
    grid: np.ndarray,
    img_levels: List[int],
    txt_levels: List[int],
    signif: np.ndarray,
    title: str,
    out_path: str,
    annotate: bool = True,
    mark_signif: bool = True,
):
    finite_vals = grid[np.isfinite(grid)]
    vmax = np.max(np.abs(finite_vals)) if finite_vals.size else 1.0
    vmax = max(vmax, 1e-6)

    # 稍微放大一点，适配 3x3 / 4x4 / 5x5
    fig_w = 1.8 * len(txt_levels) + 2.2
    fig_h = 1.6 * len(img_levels) + 1.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(grid, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="equal")

    ax.set_xticks(np.arange(len(txt_levels)))
    ax.set_yticks(np.arange(len(img_levels)))
    ax.set_xticklabels(txt_levels, fontsize=11)
    ax.set_yticklabels(img_levels, fontsize=11)

    ax.set_xlabel("Text severity", fontsize=12)
    ax.set_ylabel("Image severity", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)

    # 网格线
    ax.set_xticks(np.arange(-0.5, len(txt_levels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(img_levels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 标注数值
    if annotate:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if np.isnan(val):
                    continue
                txt = f"{val:+.4f}"
                if mark_signif and signif[i, j]:
                    txt += " *"
                color = "black" if abs(val) < 0.65 * vmax else "white"
                ax.text(
                    j, i, txt,
                    ha="center", va="center",
                    fontsize=10, color=color, fontweight="semibold"
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_txt", required=True, help="bootstrap_f1.txt")
    ap.add_argument("--out_dir", default=None, help="output dir, default = same folder as input")
    ap.add_argument("--stem", default=None, help="figure name stem, default = input filename stem")
    ap.add_argument("--no_annot", action="store_true", help="disable number annotations")
    ap.add_argument("--no_signif", action="store_true", help="disable significance star")
    args = ap.parse_args()

    rows = parse_bootstrap_txt(args.input_txt)

    in_path = Path(args.input_txt)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem if args.stem else in_path.stem

    # Interaction
    igrid, img_levels, txt_levels, isig = build_grid(rows, "interact_mean")
    iout = out_dir / f"{stem}_interaction_heatmap.png"
    draw_heatmap(
        igrid,
        img_levels,
        txt_levels,
        isig,
        pretty_title("interact_mean", stem),
        str(iout),
        annotate=not args.no_annot,
        mark_signif=not args.no_signif,
    )

    # Excess
    egrid, img_levels2, txt_levels2, esig = build_grid(rows, "excess_mean")
    eout = out_dir / f"{stem}_excess_heatmap.png"
    draw_heatmap(
        egrid,
        img_levels2,
        txt_levels2,
        esig,
        pretty_title("excess_mean", stem),
        str(eout),
        annotate=not args.no_annot,
        mark_signif=not args.no_signif,
    )

    print(f"[OK] wrote: {iout}")
    print(f"[OK] wrote: {eout}")


if __name__ == "__main__":
    main()