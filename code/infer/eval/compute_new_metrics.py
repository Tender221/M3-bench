#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_mixed_robustness.py

用途：
1) 从 bootstrap_interaction.py 的 txt 输出中，汇总：
   AMR / NDS / ANDS / SNA / WNA / WND / AIS / SIA / WIIA
2) 或者直接从 root_dir + clean_jsonl + prefix 重新计算每个 cell，再汇总。

推荐优先用法：
- 你已经有 bootstrap_f1.txt 时：直接 --bootstrap_txt
- 你只有 jsonl 预测结果时：用 --root_dir --clean_jsonl --prefix

输出：
- 一个 summary txt
- 一个 summary json
- 一个 per-cell csv（方便你后续画图 / 做总表）

定义：
AMR   = mean(S_mix / S_clean)
NDS   = mean(max(Excess, 0) / S_clean)
ANDS  = mean(Excess / S_clean)
SNA   = proportion of cells with Excess CI lower bound > 0
WNA   = proportion of cells with p(Excess>0) >= p_thresh
WND   = max(Excess / S_clean)

AIS   = mean(Interaction)
SIA   = proportion of cells with Interaction CI lower bound > 0
WIIA  = proportion of cells with p(Interaction>0) >= p_thresh

其中：
Excess = S_mul - S_mix
S_mul  = S_img * S_txt / S_clean
Interaction = Δ_mix - Δ_img - Δ_txt
Δ = (S_clean - S)/S_clean
"""

import argparse
import csv
import json
import os
import re
import string
from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

import numpy as np


# =========================
# Metric helpers (aligned with your current eval)
# =========================
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(s: str) -> List[str]:
    s = normalize_text(s)
    return s.split() if s else []


def extract_pred_text(obj: Dict[str, Any]) -> str:
    for k in ["pred", "prediction", "output", "model_output"]:
        if k in obj and obj[k] is not None:
            return str(obj[k]).strip()
    return ""


def extract_gt_texts(obj: Dict[str, Any]) -> List[str]:
    for k in ["gt_text", "answer", "label", "gt"]:
        if k in obj and obj[k] is not None:
            v = obj[k]
            if isinstance(v, list):
                out = [str(x).strip() for x in v if str(x).strip()]
                return out if out else [""]
            return [str(v).strip()]
    return [""]


def exact_match_norm(pred: str, gt: str) -> bool:
    pn = normalize_text(pred)
    gn = normalize_text(gt)
    return (gn != "") and (pn == gn)


def token_f1(pred: str, gt: str) -> float:
    ptoks = tokenize(pred)
    gtoks = tokenize(gt)
    if not gtoks or not ptoks:
        return 0.0
    common = Counter(ptoks) & Counter(gtoks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(ptoks)
    recall = num_same / len(gtoks)
    return 2 * precision * recall / (precision + recall)


def best_over_gts(pred: str, gts: List[str]) -> Tuple[bool, float]:
    em_any = False
    best_f1 = 0.0
    for gt in gts:
        if exact_match_norm(pred, gt):
            em_any = True
        f1 = token_f1(pred, gt)
        if f1 > best_f1:
            best_f1 = f1
    return em_any, best_f1


def relaxed_hit(best_f1: float, tau: float) -> float:
    return 1.0 if best_f1 >= tau else 0.0


def load_jsonl_map(path: str, id_key: str = "id") -> Dict[str, Dict[str, Any]]:
    mp: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if id_key not in obj:
                raise KeyError(f"Missing id_key='{id_key}' in {path}. keys={list(obj.keys())[:20]}")
            mp[str(obj[id_key])] = obj
    if not mp:
        raise ValueError(f"Empty jsonl: {path}")
    return mp


def align_ids(*maps: Dict[str, Dict[str, Any]]) -> List[str]:
    ids = set(maps[0].keys())
    for mp in maps[1:]:
        ids &= set(mp.keys())
    if not ids:
        raise ValueError("No overlapping ids across inputs.")
    return sorted(ids)


def build_score_arrays(
    mp: Dict[str, Dict[str, Any]],
    ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(ids)
    em_arr = np.zeros(n, dtype=np.float64)
    f1_arr = np.zeros(n, dtype=np.float64)
    r05_arr = np.zeros(n, dtype=np.float64)
    r08_arr = np.zeros(n, dtype=np.float64)

    for k, _id in enumerate(ids):
        obj = mp[_id]
        pred = extract_pred_text(obj)
        gts = extract_gt_texts(obj)
        em, best_f1 = best_over_gts(pred, gts)
        em_arr[k] = 1.0 if em else 0.0
        f1_arr[k] = float(best_f1)
        r05_arr[k] = relaxed_hit(best_f1, 0.5)
        r08_arr[k] = relaxed_hit(best_f1, 0.8)

    return em_arr, f1_arr, r05_arr, r08_arr


def pick_metric_array(metric: str, em: np.ndarray, f1: np.ndarray, r05: np.ndarray, r08: np.ndarray) -> np.ndarray:
    m = metric.lower()
    if m == "em":
        return em
    if m == "f1":
        return f1
    if m in ("relaxed_acc@0.5", "relaxed@0.5", "r@0.5"):
        return r05
    if m in ("relaxed_acc@0.8", "relaxed@0.8", "r@0.8"):
        return r08
    raise ValueError(f"Unknown metric '{metric}'")


# =========================
# Parsing bootstrap txt
# =========================
BOOT_LINE_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+"   # S_clean
    r"([+-]?\d+(?:\.\d+)?)\s+"   # S_img
    r"([+-]?\d+(?:\.\d+)?)\s+"   # S_txt
    r"([+-]?\d+(?:\.\d+)?)\s+"   # S_mixed
    r"([+-]?\d+(?:\.\d+)?)\s+"   # S_mul
    r"([+-]?\d+(?:\.\d+)?)\s+\[([+-]?\d+(?:\.\d+)?),([+-]?\d+(?:\.\d+)?)\]\s+p=([01]?(?:\.\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?)\s+\[([+-]?\d+(?:\.\d+)?),([+-]?\d+(?:\.\d+)?)\]\s+p=([01]?(?:\.\d+)?)\s*$"
)


def parse_bootstrap_txt(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = BOOT_LINE_RE.match(line.rstrip("\n"))
            if not m:
                continue
            cell, n, s_clean, s_img, s_txt, s_mix, s_mul, ex_mean, ex_lo, ex_hi, ex_p, it_mean, it_lo, it_hi, it_p = m.groups()

            cell_str = str(cell)
            if len(cell_str) < 2:
                continue
            i = int(cell_str[:-1]) if len(cell_str) > 2 else int(cell_str[0])
            j = int(cell_str[-1])

            rows.append({
                "cell": cell_str,
                "i": i,
                "j": j,
                "n": int(n),
                "S_clean": float(s_clean),
                "S_img": float(s_img),
                "S_txt": float(s_txt),
                "S_mix": float(s_mix),
                "S_mul": float(s_mul),
                "Excess_mean": float(ex_mean),
                "Excess_lo": float(ex_lo),
                "Excess_hi": float(ex_hi),
                "Excess_p_gt0": float(ex_p),
                "Interact_mean": float(it_mean),
                "Interact_lo": float(it_lo),
                "Interact_hi": float(it_hi),
                "Interact_p_gt0": float(it_p),
            })
    if not rows:
        raise ValueError(f"No bootstrap rows parsed from: {path}")
    return rows


# =========================
# Recompute from root_dir
# =========================
def parse_cells(s: str) -> List[Tuple[int, int]]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for tok in [t.strip() for t in s.replace(";", ",").split(",") if t.strip()]:
        if "_" in tok:
            a, b = tok.split("_", 1)
            out.append((int(a), int(b)))
        else:
            if len(tok) != 2 or not tok.isdigit():
                raise ValueError(f"Bad cell token '{tok}'")
            out.append((int(tok[0]), int(tok[1])))
    return out


def discover_targets(root_dir: str) -> List[Tuple[int, int]]:
    cells = set()
    for name in os.listdir(root_dir):
        m = re.fullmatch(r"(\d+)_(\d+)", name)
        if m:
            cells.add((int(m.group(1)), int(m.group(2))))
    targets = []
    for (i, j) in sorted(cells):
        if i > 0 and j > 0 and (i, 0) in cells and (0, j) in cells:
            targets.append((i, j))
    return targets


def bootstrap_arrays(clean_s: np.ndarray, img_s: np.ndarray, txt_s: np.ndarray, mix_s: np.ndarray,
                     B: int, seed: int):
    n = len(clean_s)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n), endpoint=False)

    S_clean = clean_s[idx].mean(axis=1)
    S_img   = img_s[idx].mean(axis=1)
    S_txt   = txt_s[idx].mean(axis=1)
    S_mix   = mix_s[idx].mean(axis=1)

    eps = 1e-12
    S_clean_safe = np.maximum(S_clean, eps)

    d_mix = (S_clean_safe - S_mix) / S_clean_safe
    d_img = (S_clean_safe - S_img) / S_clean_safe
    d_txt = (S_clean_safe - S_txt) / S_clean_safe
    inter = d_mix - d_img - d_txt

    S_mul = (S_img * S_txt) / S_clean_safe
    excess = S_mul - S_mix

    point = {
        "S_clean": float(clean_s.mean()),
        "S_img": float(img_s.mean()),
        "S_txt": float(txt_s.mean()),
        "S_mix": float(mix_s.mean()),
    }
    point["S_mul"] = point["S_img"] * point["S_txt"] / max(point["S_clean"], eps)
    return inter, excess, point


def summarize(a: np.ndarray):
    return {
        "mean": float(np.mean(a)),
        "lo": float(np.percentile(a, 2.5)),
        "hi": float(np.percentile(a, 97.5)),
        "p_gt0": float(np.mean(a > 0)),
    }


def compute_rows_from_root(
    root_dir: str,
    clean_jsonl: str,
    prefix: str,
    metric: str,
    B: int,
    seed: int,
    id_key: str,
    cells: str,
) -> List[Dict[str, Any]]:
    clean_mp = load_jsonl_map(clean_jsonl, id_key)
    targets = parse_cells(cells) if cells else discover_targets(root_dir)
    if not targets:
        raise ValueError("No valid targets found.")

    rows = []
    for (i, j) in targets:
        img_path = os.path.join(root_dir, f"{i}_0", f"{prefix}.jsonl")
        txt_path = os.path.join(root_dir, f"0_{j}", f"{prefix}.jsonl")
        mix_path = os.path.join(root_dir, f"{i}_{j}", f"{prefix}.jsonl")

        for p in (img_path, txt_path, mix_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(p)

        img_mp = load_jsonl_map(img_path, id_key)
        txt_mp = load_jsonl_map(txt_path, id_key)
        mix_mp = load_jsonl_map(mix_path, id_key)

        ids = align_ids(clean_mp, img_mp, txt_mp, mix_mp)

        c_em, c_f1, c_r05, c_r08 = build_score_arrays(clean_mp, ids)
        i_em, i_f1, i_r05, i_r08 = build_score_arrays(img_mp, ids)
        t_em, t_f1, t_r05, t_r08 = build_score_arrays(txt_mp, ids)
        m_em, m_f1, m_r05, m_r08 = build_score_arrays(mix_mp, ids)

        clean_s = pick_metric_array(metric, c_em, c_f1, c_r05, c_r08)
        img_s   = pick_metric_array(metric, i_em, i_f1, i_r05, i_r08)
        txt_s   = pick_metric_array(metric, t_em, t_f1, t_r05, t_r08)
        mix_s   = pick_metric_array(metric, m_em, m_f1, m_r05, m_r08)

        inter, excess, point = bootstrap_arrays(clean_s, img_s, txt_s, mix_s, B, seed)
        ex = summarize(excess)
        it = summarize(inter)

        rows.append({
            "cell": f"{i}{j}",
            "i": i,
            "j": j,
            "n": len(ids),
            "S_clean": point["S_clean"],
            "S_img": point["S_img"],
            "S_txt": point["S_txt"],
            "S_mix": point["S_mix"],
            "S_mul": point["S_mul"],
            "Excess_mean": ex["mean"],
            "Excess_lo": ex["lo"],
            "Excess_hi": ex["hi"],
            "Excess_p_gt0": ex["p_gt0"],
            "Interact_mean": it["mean"],
            "Interact_lo": it["lo"],
            "Interact_hi": it["hi"],
            "Interact_p_gt0": it["p_gt0"],
        })
    return rows


# =========================
# Aggregate new metrics
# =========================
def aggregate_rows(rows: List[Dict[str, Any]], p_thresh: float = 0.95) -> Dict[str, Any]:
    if not rows:
        raise ValueError("rows is empty")

    # Per-cell derived values
    for r in rows:
        s_clean = max(float(r["S_clean"]), 1e-12)
        r["AMR_cell"] = float(r["S_mix"]) / s_clean
        r["NDS_cell"] = max(float(r["Excess_mean"]), 0.0) / s_clean
        r["ANDS_cell"] = float(r["Excess_mean"]) / s_clean
        r["WND_cell"] = float(r["Excess_mean"]) / s_clean
        r["Interact_norm_cell"] = float(r["Interact_mean"])  # keep raw interaction for now

        r["sig_excess_pos"] = 1 if float(r["Excess_lo"]) > 0 else 0
        r["weak_excess_pos"] = 1 if float(r["Excess_p_gt0"]) >= p_thresh else 0

        r["sig_interact_pos"] = 1 if float(r["Interact_lo"]) > 0 else 0
        r["weak_interact_pos"] = 1 if float(r["Interact_p_gt0"]) >= p_thresh else 0

    n_cells = len(rows)

    amr = float(np.mean([r["AMR_cell"] for r in rows]))
    nds = float(np.mean([r["NDS_cell"] for r in rows]))
    ands = float(np.mean([r["ANDS_cell"] for r in rows]))
    wnd = float(np.max([r["WND_cell"] for r in rows]))
    ais = float(np.mean([float(r["Interact_mean"]) for r in rows]))

    sna = float(np.mean([r["sig_excess_pos"] for r in rows]))
    wna = float(np.mean([r["weak_excess_pos"] for r in rows]))
    sia = float(np.mean([r["sig_interact_pos"] for r in rows]))
    wiia = float(np.mean([r["weak_interact_pos"] for r in rows]))

    worst_excess = max(rows, key=lambda r: r["WND_cell"])
    best_excess = min(rows, key=lambda r: r["ANDS_cell"])
    worst_inter = max(rows, key=lambda r: float(r["Interact_mean"]))
    best_inter = min(rows, key=lambda r: float(r["Interact_mean"]))

    top_hot = sorted(rows, key=lambda r: r["WND_cell"], reverse=True)[:5]

    return {
        "n_cells": n_cells,
        "AMR": amr,
        "NDS": nds,
        "ANDS": ands,
        "SNA": sna,
        "WNA": wna,
        "WND": wnd,
        "AIS": ais,
        "SIA": sia,
        "WIIA": wiia,
        "p_thresh": p_thresh,
        "worst_excess_cell": worst_excess["cell"],
        "worst_excess_value": worst_excess["WND_cell"],
        "best_excess_cell": best_excess["cell"],
        "best_excess_value": best_excess["ANDS_cell"],
        "worst_interact_cell": worst_inter["cell"],
        "worst_interact_value": float(worst_inter["Interact_mean"]),
        "best_interact_cell": best_inter["cell"],
        "best_interact_value": float(best_inter["Interact_mean"]),
        "top_hotspots": [
            {
                "cell": r["cell"],
                "Excess_mean": r["Excess_mean"],
                "Excess_norm": r["WND_cell"],
                "Excess_CI": [r["Excess_lo"], r["Excess_hi"]],
                "Excess_p_gt0": r["Excess_p_gt0"],
                "Interact_mean": r["Interact_mean"],
                "Interact_CI": [r["Interact_lo"], r["Interact_hi"]],
                "Interact_p_gt0": r["Interact_p_gt0"],
            }
            for r in top_hot
        ],
    }


def write_per_cell_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "cell", "i", "j", "n",
        "S_clean", "S_img", "S_txt", "S_mix", "S_mul",
        "Excess_mean", "Excess_lo", "Excess_hi", "Excess_p_gt0",
        "Interact_mean", "Interact_lo", "Interact_hi", "Interact_p_gt0",
        "AMR_cell", "NDS_cell", "ANDS_cell", "WND_cell",
        "sig_excess_pos", "weak_excess_pos", "sig_interact_pos", "weak_interact_pos",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def write_summary_txt(summary: Dict[str, Any], out_txt: str, source_desc: str) -> None:
    os.makedirs(os.path.dirname(out_txt) or ".", exist_ok=True)
    lines = []
    lines.append(f"[source] {source_desc}")
    lines.append(f"[n_cells] {summary['n_cells']}")
    lines.append("")
    lines.append("=== Mixed Robustness Summary ===")
    lines.append(f"AMR   = {summary['AMR']:.6f}   # mean(S_mix / S_clean), larger is better")
    lines.append(f"NDS   = {summary['NDS']:.6f}   # mean(max(Excess,0)/S_clean), smaller is better")
    lines.append(f"ANDS  = {summary['ANDS']:.6f}   # mean(Excess/S_clean), signed")
    lines.append(f"SNA   = {summary['SNA']:.6f}   # fraction with Excess CI lower bound > 0")
    lines.append(f"WNA   = {summary['WNA']:.6f}   # fraction with p(Excess>0) >= {summary['p_thresh']}")
    lines.append(f"WND   = {summary['WND']:.6f}   # max(Excess/S_clean)")
    lines.append("")
    lines.append("=== Interaction Summary ===")
    lines.append(f"AIS   = {summary['AIS']:.6f}   # mean Interaction")
    lines.append(f"SIA   = {summary['SIA']:.6f}   # fraction with Interaction CI lower bound > 0")
    lines.append(f"WIIA  = {summary['WIIA']:.6f}   # fraction with p(Interaction>0) >= {summary['p_thresh']}")
    lines.append("")
    lines.append("=== Extreme Cells ===")
    lines.append(f"worst_excess_cell   = {summary['worst_excess_cell']}  value={summary['worst_excess_value']:.6f}")
    lines.append(f"best_excess_cell    = {summary['best_excess_cell']}  value={summary['best_excess_value']:.6f}")
    lines.append(f"worst_interact_cell = {summary['worst_interact_cell']}  value={summary['worst_interact_value']:.6f}")
    lines.append(f"best_interact_cell  = {summary['best_interact_cell']}  value={summary['best_interact_value']:.6f}")
    lines.append("")
    lines.append("=== Top Excess Hotspots ===")
    for x in summary["top_hotspots"]:
        lines.append(
            f"cell={x['cell']}  "
            f"Excess={x['Excess_mean']:+.6f} [{x['Excess_CI'][0]:+.6f},{x['Excess_CI'][1]:+.6f}] p={x['Excess_p_gt0']:.3f}  "
            f"Interact={x['Interact_mean']:+.6f} [{x['Interact_CI'][0]:+.6f},{x['Interact_CI'][1]:+.6f}] p={x['Interact_p_gt0']:.3f}"
        )

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()

    # Mode A: read existing bootstrap txt
    ap.add_argument("--bootstrap_txt", default="", help="Existing bootstrap txt from bootstrap_interaction.py")

    # Mode B: recompute from prediction jsonl grid
    ap.add_argument("--root_dir", default="", help="root dir containing i_j subdirs")
    ap.add_argument("--clean_jsonl", default="", help="clean prediction jsonl")
    ap.add_argument("--prefix", default="", help="prefix jsonl name inside each cell dir")
    ap.add_argument("--metric", default="f1", help="em|f1|relaxed_acc@0.5|relaxed_acc@0.8")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--id_key", default="id")
    ap.add_argument("--cells", default="", help="optional subset, e.g. 22,23,24")

    # Aggregation control
    ap.add_argument("--p_thresh", type=float, default=0.95, help="threshold for weak-positive area")
    ap.add_argument("--out_txt", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)

    args = ap.parse_args()

    use_bootstrap = bool(args.bootstrap_txt.strip())
    use_root = bool(args.root_dir.strip())

    if not use_bootstrap and not use_root:
        raise ValueError("Provide either --bootstrap_txt or --root_dir + --clean_jsonl + --prefix")

    if use_bootstrap:
        rows = parse_bootstrap_txt(args.bootstrap_txt)
        source_desc = f"bootstrap_txt={args.bootstrap_txt}"
    else:
        if not args.clean_jsonl or not args.prefix:
            raise ValueError("When using --root_dir, you also need --clean_jsonl and --prefix")
        rows = compute_rows_from_root(
            root_dir=args.root_dir,
            clean_jsonl=args.clean_jsonl,
            prefix=args.prefix,
            metric=args.metric,
            B=args.B,
            seed=args.seed,
            id_key=args.id_key,
            cells=args.cells,
        )
        source_desc = f"root_dir={args.root_dir}, clean_jsonl={args.clean_jsonl}, prefix={args.prefix}, metric={args.metric}"

    summary = aggregate_rows(rows, p_thresh=args.p_thresh)

    # add metadata
    out_obj = {
        "source": source_desc,
        "summary": summary,
        "rows": rows,
    }

    write_per_cell_csv(rows, args.out_csv)
    write_summary_txt(summary, args.out_txt, source_desc)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"Wrote txt : {args.out_txt}")
    print(f"Wrote json: {args.out_json}")
    print(f"Wrote csv : {args.out_csv}")


if __name__ == "__main__":
    main()