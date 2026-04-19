#!/usr/bin/env python3
# eval_metrics_openqa.py
import json
import re
import argparse
import string
from collections import Counter
from typing import Dict, Any, Tuple, List

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def normalize_text(s: str) -> str:
    """
    Deterministic normalize (SQuAD-style):
    - lowercase
    - remove punctuation
    - remove articles (a/an/the)
    - collapse whitespace
    """
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
    """
    Support gt as str or list[str]
    """
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

def best_over_gts(pred: str, gts: List[str]) -> Tuple[bool, float, str]:
    """
    For multiple GTs:
    - EM: any match
    - F1: max
    Return (em_any, best_f1, best_gt_for_f1)
    """
    em_any = False
    best_f1 = 0.0
    best_gt = gts[0] if gts else ""
    for gt in gts:
        if exact_match_norm(pred, gt):
            em_any = True
        f1 = token_f1(pred, gt)
        if f1 > best_f1:
            best_f1 = f1
            best_gt = gt
    return em_any, best_f1, best_gt

def derive_out_path(pred_path: str, suffix: str) -> str:
    """
    Robust output naming even if file not endswith .jsonl
    """
    if pred_path.endswith(".jsonl"):
        return pred_path[:-5] + suffix
    return pred_path + suffix

def compute_metrics(pred_path: str, tau_list: List[float]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    total = 0
    em_sum = 0
    f1_sum = 0.0
    relaxed_hits = {tau: 0 for tau in tau_list}
    wrong = []

    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)

            pred = extract_pred_text(obj)
            gts = extract_gt_texts(obj)

            em, f1, best_gt = best_over_gts(pred, gts)
            em_sum += 1 if em else 0
            f1_sum += f1

            for tau in tau_list:
                if f1 >= tau:
                    relaxed_hits[tau] += 1

            # wrong sampling (store all non-EM or low-F1 cases)
            if (not em) or (tau_list and f1 < max(tau_list)):
                wrong.append({
                    "id": obj.get("id", ""),
                    "question": obj.get("question", ""),
                    "image_path": obj.get("image_path", ""),
                    "pred": pred,
                    "gt_text": best_gt,
                    "all_gt_texts": gts,
                    "pred_norm": normalize_text(pred),
                    "gt_norm": normalize_text(best_gt),
                    "em": bool(em),
                    "f1": round(float(f1), 6),
                })

    em = (em_sum / total) if total else 0.0
    f1 = (f1_sum / total) if total else 0.0

    metrics: Dict[str, Any] = {
        "total": total,
        "em": em,
        "f1": f1,
    }
    for tau in tau_list:
        metrics[f"relaxed_acc@{tau:g}"] = (relaxed_hits[tau] / total) if total else 0.0

    return metrics, wrong

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", type=str, required=True, help="Path to predictions jsonl")
    ap.add_argument("--tau", type=str, default="0.5,0.8", help="Comma-separated thresholds, e.g., 0.5,0.8; use '' to disable")
    ap.add_argument("--save_wrong", action="store_true", help="Save wrong cases to *_wrong.json")
    ap.add_argument("--save_metrics", action="store_true", help="Save metrics to *_metrics.json")
    args = ap.parse_args()

    tau_list: List[float] = []
    if args.tau.strip():
        tau_list = sorted(set(float(x) for x in args.tau.split(",") if x.strip() != ""))

    metrics, wrong = compute_metrics(args.pred_path, tau_list)

    print(f"Total: {metrics['total']}")
    print(f"Normalized EM: {metrics['em']:.4f}")
    print(f"Token F1:      {metrics['f1']:.4f}")
    for tau in tau_list:
        k = f"relaxed_acc@{tau:g}"
        print(f"Relaxed Acc@{tau:g}: {metrics[k]:.4f}")

    if args.save_metrics:
        out_metrics = derive_out_path(args.pred_path, "_metrics.json")
        with open(out_metrics, "w", encoding="utf-8") as wf:
            json.dump(metrics, wf, ensure_ascii=False, indent=2)
        print(f"Metrics saved to: {out_metrics}")

    if args.save_wrong and wrong:
        out_wrong = derive_out_path(args.pred_path, "_wrong.json")
        with open(out_wrong, "w", encoding="utf-8") as wf:
            json.dump(wrong, wf, ensure_ascii=False, indent=2)
        print(f"Wrong samples saved to: {out_wrong}")

if __name__ == "__main__":
    main()