#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Interaction / Excess from OpenQA metrics files laid out as:
  root_dir/<i>_<j>/<prefix>.jsonl   (or .json)

Example:
  /root/code/infer/qwen2.5-7b_outputs/okvqa/perturbed_output/jpeg_ocr/2_3/jpeg_ocr.jsonl

It will compute, for each cell (i,j):
  S_img   = S(i,0)
  S_txt   = S(0,j)
  S_mixed = S(i,j)
  S_mul   = S_img * S_txt / S_clean
  Excess  = S_mul - S_mixed
  Interact= Δ_mixed - Δ_img - Δ_txt
where Δ = (S_clean - S)/S_clean
"""

import argparse
import json
import os
import re
from typing import Dict, Any, Tuple, List, Optional


def _read_json_or_jsonl(path: str) -> Dict[str, Any]:
    """
    Supports:
      - JSON file: whole file is a JSON object
      - JSONL file: each line is JSON; we use the LAST non-empty JSON object
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError(f"Empty file: {path}")

    # Try parse as a single JSON object first
    if content.lstrip().startswith("{"):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

    # Fallback: JSONL (take last non-empty line)
    last_obj = None
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            last_obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON line in {path}: {line[:120]} ...") from e

    if not isinstance(last_obj, dict):
        raise ValueError(f"Could not parse JSON object from {path}")
    return last_obj


def _get_metric(d: Dict[str, Any], key: str) -> float:
    # exact match
    if key in d and isinstance(d[key], (int, float)) and not isinstance(d[key], bool):
        return float(d[key])
    # case-insensitive match
    km = {str(k).lower(): k for k in d.keys()}
    k2 = km.get(key.lower())
    if k2 is not None and isinstance(d[k2], (int, float)) and not isinstance(d[k2], bool):
        return float(d[k2])
    raise KeyError(f"Metric '{key}' not found in keys={list(d.keys())}")


def _find_metrics_file(cell_dir: str, prefix: Optional[str]) -> str:
    """
    STRICT mode:
      If prefix is provided, ONLY accept:
        <prefix>._metrics.json or <prefix>._metrics.jsonl
      Otherwise, try to find any *._metrics.json(.l) / *_metrics.json(.l)
    """
    files = sorted(os.listdir(cell_dir))

    # 1) strict: exact prefix metrics
    if prefix:
        for fn in (f"{prefix}._metrics.json", f"{prefix}._metrics.jsonl",
                   f"{prefix}_metrics.json", f"{prefix}_metrics.jsonl"):
            p = os.path.join(cell_dir, fn)
            if os.path.isfile(p):
                return p

        raise FileNotFoundError(
            f"[NO_METRICS_FOUND] Expected metrics file not found in {cell_dir}. "
            f"Tried: {prefix}._metrics.json/.jsonl (and _metrics variants). "
            f"Existing files: {files}"
        )

    # 2) no prefix: any metrics-like file
    metrics_like = [fn for fn in files if re.search(r"(^|[._-])metrics?\.jsonl?$", fn)]
    metrics_like.sort(key=lambda x: (0 if x.endswith(".json") else 1, len(x)))
    if metrics_like:
        return os.path.join(cell_dir, metrics_like[0])

    raise FileNotFoundError(f"[NO_METRICS_FOUND] No metrics-like json/jsonl in {cell_dir}. Existing files: {files}")


def _parse_cells(cells_str: str) -> List[Tuple[int, int]]:
    """
    Accept:
      - "22,23,32,33"
      - "2_2,2_3,3_2,3_3"
      - "2,2;2,3;3,2;3,3"
    """
    s = cells_str.strip()
    if not s:
        return []

    # normalize separators to commas
    s = s.replace(";", ",").replace(" ", "")
    parts = [p for p in s.split(",") if p]

    cells: List[Tuple[int, int]] = []
    for p in parts:
        if "_" in p:
            a, b = p.split("_", 1)
            cells.append((int(a), int(b)))
        else:
            # assume two digits like "23" (also allow more digits: e.g. "10-2"? not supported)
            m = re.fullmatch(r"(\d)(\d)", p)
            if not m:
                raise ValueError(f"Bad cell token '{p}'. Use e.g. 23 or 2_3 or 2,3.")
            cells.append((int(m.group(1)), int(m.group(2))))
    return cells


def _delta(s_clean: float, s: float) -> float:
    if s_clean == 0:
        raise ZeroDivisionError("S_clean is 0; cannot compute relative drop.")
    return (s_clean - s) / s_clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True,
                    help="e.g. /root/code/infer/.../perturbed_output/jpeg_ocr")
    ap.add_argument("--prefix", default=None,
                    help="metrics file prefix inside each cell dir, e.g. jpeg_ocr (expects jpeg_ocr.jsonl). If omitted, auto-detect.")
    ap.add_argument("--metric", default="f1",
                    help="which metric to use as score S (default: f1). Examples: em, f1, relaxed_acc@0.5")
    ap.add_argument("--clean_metrics", default=None,
                    help="optional path to clean metrics JSON/JSONL (S_clean). If omitted, reads from <root_dir>/0_0/")
    ap.add_argument("--cells", default="",
                    help="cells to compute, e.g. '22,23,32,33' or '2_2,2_3,3_2,3_3'. If empty, computes all found (excluding row/col 0-only).")
    ap.add_argument("--out_txt", required=True,
                    help="output txt path")
    args = ap.parse_args()

    root = args.root_dir
    metric = args.metric
    prefix = args.prefix

    # Load S_clean
    if args.clean_metrics:
        clean_d = _read_json_or_jsonl(args.clean_metrics)
        s_clean = _get_metric(clean_d, metric)
        clean_src = args.clean_metrics
    else:
        cell00 = os.path.join(root, "0_0")
        mf00 = _find_metrics_file(cell00, prefix)
        d00 = _read_json_or_jsonl(mf00)
        s_clean = _get_metric(d00, metric)
        clean_src = mf00

    # Decide cells
    cells = _parse_cells(args.cells)
    if not cells:
        # auto-discover: take all <i>_<j> dirs with i>0 and j>0
        found: List[Tuple[int, int]] = []
        for name in os.listdir(root):
            m = re.fullmatch(r"(\d+)_(\d+)", name)
            if not m:
                continue
            i = int(m.group(1))
            j = int(m.group(2))
            if i > 0 and j > 0:
                found.append((i, j))
        found.sort()
        cells = found

    # Cache scores
    score_cache: Dict[Tuple[int, int], float] = {}

    def load_score(i: int, j: int) -> float:
        key = (i, j)
        if key in score_cache:
            return score_cache[key]
        cell_dir = os.path.join(root, f"{i}_{j}")
        mf = _find_metrics_file(cell_dir, prefix)
        d = _read_json_or_jsonl(mf)
        s = _get_metric(d, metric)
        score_cache[key] = s
        return s

    # Compute table rows
    rows = []
    for (i, j) in cells:
        s_img = load_score(i, 0)
        s_txt = load_score(0, j)
        s_mixed = load_score(i, j)

        s_mul = (s_img * s_txt / s_clean) if s_clean != 0 else float("nan")
        excess = s_mul - s_mixed  # <-- matches your sample sign convention
        interact = _delta(s_clean, s_mixed) - _delta(s_clean, s_img) - _delta(s_clean, s_txt)

        rows.append((i, j, s_img, s_txt, s_mixed, s_mul, excess, interact))

    # Write txt
    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write(f"[key] {metric}\n")
        f.write(f"[S_clean(0,0)] {s_clean:.6f}\n")
        f.write(f"[clean_src] {clean_src}\n\n")

        f.write("cell      S_img      S_txt    S_mixed      S_mul     Excess   Interact\n")
        f.write("----------------------------------------------------------------------\n")
        for (i, j, s_img, s_txt, s_mixed, s_mul, excess, interact) in rows:
            cell = f"{i}{j}"
            f.write(f"{cell:<6}  {s_img:>10.6f} {s_txt:>10.6f} {s_mixed:>10.6f} {s_mul:>10.6f} {excess:>10.6f} {interact:>10.6f}\n")


if __name__ == "__main__":
    main()