#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, string
from collections import Counter
from typing import Dict, Any, List, Tuple
import numpy as np

# ======= Copied/Aligned with your eval_metrics_openqa.py =======
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
                out = [str(x).strip() for x in v if str(x).strip() != ""]
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

# relaxed acc uses F1 >= tau
def relaxed_hit(best_f1: float, tau: float) -> float:
    return 1.0 if best_f1 >= tau else 0.0

# ======= Fast bootstrap helpers =======
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

def build_score_arrays(mp: Dict[str, Dict[str, Any]], ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    returns arrays aligned with ids:
      em_arr, f1_arr, r05_arr, r08_arr
    """
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
    mean = float(np.mean(a))
    lo = float(np.percentile(a, 2.5))
    hi = float(np.percentile(a, 97.5))
    ppos = float(np.mean(a > 0))
    return mean, lo, hi, ppos

def parse_cells(s: str) -> List[Tuple[int, int]]:
    s = (s or "").strip()
    if not s:
        return []
    out=[]
    for tok in [t.strip() for t in s.replace(";", ",").split(",") if t.strip()]:
        if "_" in tok:
            a,b = tok.split("_", 1)
            out.append((int(a), int(b)))
        else:
            if len(tok)!=2 or not tok.isdigit():
                raise ValueError(f"Bad cell token '{tok}'")
            out.append((int(tok[0]), int(tok[1])))
    return out

def discover_targets(root_dir: str) -> List[Tuple[int,int]]:
    cells=set()
    for name in os.listdir(root_dir):
        m=re.fullmatch(r"(\d+)_(\d+)", name)
        if m:
            cells.add((int(m.group(1)), int(m.group(2))))
    targets=[]
    for (i,j) in sorted(cells):
        if i>0 and j>0 and (i,0) in cells and (0,j) in cells:
            targets.append((i,j))
    return targets

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True)
    ap.add_argument("--prefix", default="jpeg_ocr")
    ap.add_argument("--clean_jsonl", required=True, help="clean1000.jsonl (per-example with id/pred/gt_text or gt list)")
    ap.add_argument("--cells", default="", help="e.g. 22,23,24,32,33,... empty->auto scan")
    ap.add_argument("--metric", default="f1", help="em|f1|relaxed_acc@0.5|relaxed_acc@0.8")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--id_key", default="id")
    ap.add_argument("--out_txt", required=True)
    args=ap.parse_args()

    clean_mp = load_jsonl_map(args.clean_jsonl, args.id_key)

    targets = parse_cells(args.cells)
    if not targets:
        targets = discover_targets(args.root_dir)
    if not targets:
        raise ValueError("No targets found (need i>0,j>0 with deps i_0 and 0_j).")

    lines=[]
    lines.append(f"[root_dir] {args.root_dir}")
    lines.append(f"[clean_jsonl] {args.clean_jsonl}")
    lines.append(f"[metric] {args.metric}  [B] {args.B}  [seed] {args.seed}")
    lines.append("")
    lines.append("cell  n   S_clean   S_img    S_txt   S_mixed   S_mul    Excess(mean [lo,hi] p>0)     Interact(mean [lo,hi] p>0)")
    lines.append("---------------------------------------------------------------------------------------------------------------------")

    for (i,j) in targets:
        img_path = os.path.join(args.root_dir, f"{i}_0", f"{args.prefix}.jsonl")
        txt_path = os.path.join(args.root_dir, f"0_{j}", f"{args.prefix}.jsonl")
        mix_path = os.path.join(args.root_dir, f"{i}_{j}", f"{args.prefix}.jsonl")
        for p in (img_path, txt_path, mix_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(p)

        img_mp = load_jsonl_map(img_path, args.id_key)
        txt_mp = load_jsonl_map(txt_path, args.id_key)
        mix_mp = load_jsonl_map(mix_path, args.id_key)

        ids = align_ids(clean_mp, img_mp, txt_mp, mix_mp)
        n = len(ids)

        c_em, c_f1, c_r05, c_r08 = build_score_arrays(clean_mp, ids)
        i_em, i_f1, i_r05, i_r08 = build_score_arrays(img_mp, ids)
        t_em, t_f1, t_r05, t_r08 = build_score_arrays(txt_mp, ids)
        m_em, m_f1, m_r05, m_r08 = build_score_arrays(mix_mp, ids)

        clean_s = pick_metric_array(args.metric, c_em, c_f1, c_r05, c_r08)
        img_s   = pick_metric_array(args.metric, i_em, i_f1, i_r05, i_r08)
        txt_s   = pick_metric_array(args.metric, t_em, t_f1, t_r05, t_r08)
        mix_s   = pick_metric_array(args.metric, m_em, m_f1, m_r05, m_r08)

        inter, excess, point = bootstrap_arrays(clean_s, img_s, txt_s, mix_s, args.B, args.seed)

        im, ilo, ihi, ip = summarize(inter)
        em, elo, ehi, ep = summarize(excess)

        cell = f"{i}{j}"
        lines.append(
            f"{cell:>4}  {n:>4} "
            f"{point['S_clean']:>8.6f} {point['S_img']:>8.6f} {point['S_txt']:>8.6f} {point['S_mix']:>8.6f} {point['S_mul']:>8.6f}  "
            f"{em:+.6f} [{elo:+.6f},{ehi:+.6f}] p={ep:.3f}    "
            f"{im:+.6f} [{ilo:+.6f},{ihi:+.6f}] p={ip:.3f}"
        )

    os.makedirs(os.path.dirname(args.out_txt) or ".", exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote: {args.out_txt}")

if __name__ == "__main__":
    main()