import json
import re
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# ----------------------------
# 1) Normalization (与你首页一致)
# ----------------------------
_ARTICLES = {"a", "an", "the"}
_PUNC_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")

def normalize_answer(s: str) -> str:
    s = (s or "").lower()
    s = _PUNC_RE.sub(" ", s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    s = " ".join(tokens)
    s = _SPACE_RE.sub(" ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    s = normalize_answer(s)
    return s.split() if s else []

# ----------------------------
# 2) EM / Token-F1 (SQuAD style)
# ----------------------------
def exact_match(pred: str, gts: List[str]) -> int:
    p = normalize_answer(pred)
    return int(any(p == normalize_answer(gt) for gt in gts))

def token_f1(pred: str, gts: List[str]) -> float:
    p_tokens = tokenize(pred)
    if not gts:
        return 0.0

    def f1_one(gt: str) -> float:
        g_tokens = tokenize(gt)
        if len(p_tokens) == 0 and len(g_tokens) == 0:
            return 1.0
        if len(p_tokens) == 0 or len(g_tokens) == 0:
            return 0.0
        common = defaultdict(int)
        for t in p_tokens:
            common[t] += 1
        num_same = 0
        for t in g_tokens:
            if common[t] > 0:
                num_same += 1
                common[t] -= 1
        if num_same == 0:
            return 0.0
        precision = num_same / len(p_tokens)
        recall = num_same / len(g_tokens)
        return 2 * precision * recall / (precision + recall)

    return max(f1_one(gt) for gt in gts)

def relaxed_acc(pred: str, gts: List[str], tau: float = 0.5) -> int:
    return int(token_f1(pred, gts) >= tau)

# ----------------------------
# 3) Read jsonl
# ----------------------------
def read_pred_jsonl(path: str,
                   qid_key: str = "question_id",
                   pred_key: str = "pred",
                   answers_key: str = "answers") -> Dict[str, Tuple[str, List[str]]]:
    """
    returns: {qid: (pred, gts)}
    gts will be List[str]
    """
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            qid = str(obj[qid_key])
            pred = obj.get(pred_key, "")
            ans = obj.get(answers_key, [])
            if isinstance(ans, str):
                gts = [ans]
            else:
                gts = [str(a) for a in ans]
            out[qid] = (pred, gts)
    return out

# ----------------------------
# 4) Compute dataset-level score S
# ----------------------------
def compute_scores(records: Dict[str, Tuple[str, List[str]]]) -> Dict[str, float]:
    ems, f1s, acc05, acc08 = [], [], [], []
    for qid, (pred, gts) in records.items():
        ems.append(exact_match(pred, gts))
        f1s.append(token_f1(pred, gts))
        acc05.append(relaxed_acc(pred, gts, tau=0.5))
        acc08.append(relaxed_acc(pred, gts, tau=0.8))
    n = max(1, len(ems))
    return {
        "EM": sum(ems) / n,
        "F1": sum(f1s) / n,
        "Acc@0.5": sum(acc05) / n,
        "Acc@0.8": sum(acc08) / n,
        "N": len(ems),
    }

# ----------------------------
# 5) Interaction (additive) + Excess (multiplicative)
# ----------------------------
def relative_drop(S_clean: float, S_pert: float) -> float:
    # Δ = (S_c - S_p)/S_c
    if S_clean <= 0:
        return float("nan")
    return (S_clean - S_pert) / S_clean

def additive_interaction(S_clean: float, S_img: float, S_txt: float, S_mixed: float) -> float:
    d_img = relative_drop(S_clean, S_img)
    d_txt = relative_drop(S_clean, S_txt)
    d_mix = relative_drop(S_clean, S_mixed)
    return d_mix - (d_img + d_txt)

def multiplicative_expected(S_clean: float, S_img: float, S_txt: float) -> float:
    # S_mul = (S_img * S_txt)/S_clean
    if S_clean <= 0:
        return float("nan")
    return (S_img * S_txt) / S_clean

def multiplicative_excess(S_clean: float, S_img: float, S_txt: float, S_mixed: float) -> float:
    S_mul = multiplicative_expected(S_clean, S_img, S_txt)
    return S_mul - S_mixed

# ----------------------------
# 6) Bootstrap CI (按样本qid重采样)
# ----------------------------
def bootstrap_ci(
    qids: List[str],
    clean: Dict[str, Tuple[str, List[str]]],
    img: Dict[str, Tuple[str, List[str]]],
    txt: Dict[str, Tuple[str, List[str]]],
    mixed: Dict[str, Tuple[str, List[str]]],
    metric: str = "F1",
    n_boot: int = 2000,
    seed: int = 0
) -> Dict[str, Any]:
    rng = random.Random(seed)

    def score_subset(src: Dict[str, Tuple[str, List[str]]], subset: List[str]) -> float:
        # compute only chosen qids
        ems, f1s, a05, a08 = [], [], [], []
        for qid in subset:
            pred, gts = src[qid]
            ems.append(exact_match(pred, gts))
            f1s.append(token_f1(pred, gts))
            a05.append(relaxed_acc(pred, gts, tau=0.5))
            a08.append(relaxed_acc(pred, gts, tau=0.8))
        n = max(1, len(subset))
        if metric == "EM":
            return sum(ems) / n
        if metric == "F1":
            return sum(f1s) / n
        if metric == "Acc@0.5":
            return sum(a05) / n
        if metric == "Acc@0.8":
            return sum(a08) / n
        raise ValueError(f"Unknown metric: {metric}")

    stats_add = []
    stats_exc = []

    for _ in range(n_boot):
        sample = [qids[rng.randrange(len(qids))] for _ in range(len(qids))]

        S_c = score_subset(clean, sample)
        S_i = score_subset(img, sample)
        S_t = score_subset(txt, sample)
        S_m = score_subset(mixed, sample)

        stats_add.append(additive_interaction(S_c, S_i, S_t, S_m))
        stats_exc.append(multiplicative_excess(S_c, S_i, S_t, S_m))

    stats_add.sort()
    stats_exc.sort()

    def pct(a: List[float], p: float) -> float:
        idx = int(round((len(a) - 1) * p))
        return a[idx]

    return {
        "metric": metric,
        "additive_interaction_mean": sum(stats_add) / len(stats_add),
        "additive_interaction_ci95": (pct(stats_add, 0.025), pct(stats_add, 0.975)),
        "multiplicative_excess_mean": sum(stats_exc) / len(stats_exc),
        "multiplicative_excess_ci95": (pct(stats_exc, 0.025), pct(stats_exc, 0.975)),
        "n_boot": n_boot,
    }

# ----------------------------
# 7) Main glue
# ----------------------------
def evaluate_four_conditions(clean_path: str, img_path: str, txt_path: str, mixed_path: str,
                             metric: str = "F1", n_boot: int = 2000) -> None:
    clean = read_pred_jsonl(clean_path)
    img = read_pred_jsonl(img_path)
    txt = read_pred_jsonl(txt_path)
    mixed = read_pred_jsonl(mixed_path)

    # Align qids intersection to be safe
    qids = sorted(set(clean) & set(img) & set(txt) & set(mixed))
    assert len(qids) > 0, "No overlapping question_id across files."

    def submap(src):
        return {qid: src[qid] for qid in qids}

    clean, img, txt, mixed = map(submap, [clean, img, txt, mixed])

    S_clean = compute_scores(clean)[metric]
    S_img = compute_scores(img)[metric]
    S_txt = compute_scores(txt)[metric]
    S_mixed = compute_scores(mixed)[metric]

    d_img = relative_drop(S_clean, S_img)
    d_txt = relative_drop(S_clean, S_txt)
    d_mix = relative_drop(S_clean, S_mixed)

    inter_add = additive_interaction(S_clean, S_img, S_txt, S_mixed)
    S_mul = multiplicative_expected(S_clean, S_img, S_txt)
    exc = multiplicative_excess(S_clean, S_img, S_txt, S_mixed)

    print("=== Scores (metric =", metric, ") ===")
    print("N:", len(qids))
    print(f"S_clean = {S_clean:.4f}")
    print(f"S_img   = {S_img:.4f}   Δ_img   = {d_img:.4f}")
    print(f"S_txt   = {S_txt:.4f}   Δ_txt   = {d_txt:.4f}")
    print(f"S_mixed = {S_mixed:.4f} Δ_mixed = {d_mix:.4f}")
    print()
    print("=== Additive interaction ===")
    print(f"Interaction_add = {inter_add:.4f}")
    print()
    print("=== Multiplicative null & excess ===")
    print(f"S_mul_expected  = {S_mul:.4f}")
    print(f"Excess = S_mul - S_mixed = {exc:.4f}  ( >0 => supra-multiplicative collapse )")
    print()

    ci = bootstrap_ci(qids, clean, img, txt, mixed, metric=metric, n_boot=n_boot, seed=0)
    print("=== Bootstrap 95% CI ===")
    print("Additive interaction mean:", f"{ci['additive_interaction_mean']:.4f}",
          "CI95:", tuple(f"{x:.4f}" for x in ci["additive_interaction_ci95"]))
    print("Multiplicative excess mean:", f"{ci['multiplicative_excess_mean']:.4f}",
          "CI95:", tuple(f"{x:.4f}" for x in ci["multiplicative_excess_ci95"]))

# Example:
# evaluate_four_conditions(
#     clean_path="/path/clean.jsonl",
#     img_path="/path/img.jsonl",
#     txt_path="/path/txt.jsonl",
#     mixed_path="/path/mixed.jsonl",
#     metric="F1",
#     n_boot=2000
# )