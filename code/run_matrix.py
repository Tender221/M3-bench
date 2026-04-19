#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INFER_DIR = SCRIPT_DIR / "infer"
EVAL_DIR = INFER_DIR / "eval"
ROUTE_DIR = SCRIPT_DIR / "route"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_cmd(cmd: List[str], *, dry_run: bool = False, cwd: Optional[Path] = None, desc: Optional[str] = None) -> None:
    printable = " ".join(shlex.quote(part) for part in cmd)
    if desc:
        print(f"\n[{desc}]\n>>> {printable}")
    else:
        print(f"\n>>> {printable}")
    if dry_run:
        return

    start = time.time()
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)
    elapsed = time.time() - start
    if desc:
        print(f"[done] {desc} ({elapsed:.1f}s)")
    else:
        print(f"[done] ({elapsed:.1f}s)")


def maybe_link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src.resolve(), dst)
    except Exception:
        import shutil

        shutil.copy2(src, dst)


def progress_iter(items, *, total: Optional[int] = None, desc: str = "Progress"):
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=desc, dynamic_ncols=True, leave=True)


def read_metrics_if_exists(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def metrics_path_candidates_from_pred(pred: Path) -> List[Path]:
    return [pred.with_name(pred.stem + "_metrics.json"), pred.with_name(pred.stem + "._metrics.json")]


def resolve_metrics_path_from_pred(pred: Path) -> Path:
    for candidate in metrics_path_candidates_from_pred(pred):
        if candidate.exists():
            return candidate
    return pred.with_name(pred.stem + "_metrics.json")


def default_infer_script_for_model(model_name: str) -> str:
    key = (model_name or "").strip().lower().replace("-", "").replace("_", "")
    mapping = {
        "qwen25": INFER_DIR / "infer_openqa_qwen25.py",
        "qwen2.5": INFER_DIR / "infer_openqa_qwen25.py",
        "qwen3": INFER_DIR / "infer_openqa_qwen3.py",
        "llava15": INFER_DIR / "infer_openqa_llava.py",
        "llava": INFER_DIR / "infer_openqa_llava.py",
        "blip2": INFER_DIR / "infer_openqa_blip2.py",
    }
    return str(mapping.get(key, INFER_DIR / "infer_openqa_qwen25.py"))


def shared_cell_root(shared_data_root: Path, i: int, j: int) -> Path:
    return shared_data_root / "clean" if i == 0 and j == 0 else shared_data_root / "cells" / f"{i}_{j}"


def shared_cell_data_dir(shared_data_root: Path, i: int, j: int) -> Path:
    return shared_cell_root(shared_data_root, i, j) / "data"


def perturb_tag(img_perturb: str, i: int, txt_perturb: str, j: int) -> str:
    return f"{img_perturb}_is{i}__{txt_perturb}_ts{j}"


def generated_jsonl_paths(shared_data_root: Path, img_perturb: str, txt_perturb: str, i: int, j: int) -> Dict[str, Path]:
    tag = perturb_tag(img_perturb, i, txt_perturb, j)
    base = shared_cell_data_dir(shared_data_root, i, j)
    return {
        "clean": base / f"clean_s0_{tag}.jsonl",
        "img_only": base / f"img_only_{tag}.jsonl",
        "txt_only": base / f"txt_only_{tag}.jsonl",
        "mixed": base / f"mixed_{tag}.jsonl",
    }


def result_cell_root(exp_root: Path, i: int, j: int) -> Path:
    return exp_root / "clean" if i == 0 and j == 0 else exp_root / "cells" / f"{i}_{j}"


def result_cell_pred_dir(exp_root: Path, i: int, j: int) -> Path:
    return result_cell_root(exp_root, i, j) / "preds"


def pred_path(exp_root: Path, i: int, j: int, infer_mode: str, prefix: str) -> Path:
    return result_cell_pred_dir(exp_root, i, j) / f"{prefix}__{infer_mode}.jsonl"


def build_needed_cells(severities: List[int]) -> List[Tuple[int, int, str]]:
    cells: List[Tuple[int, int, str]] = [(0, 0, "clean")]
    positive = [value for value in severities if value != 0]
    for i in positive:
        cells.append((i, 0, "img_only"))
    for j in positive:
        cells.append((0, j, "txt_only"))
    for i in positive:
        for j in positive:
            cells.append((i, j, "mixed"))
    return cells


def validate_openqa_jsonl(jsonl_path: Path, max_lines: int = 5) -> None:
    required = {"id", "question", "image_path"}
    seen = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            missing = [key for key in required if key not in obj]
            if missing:
                raise ValueError(f"Schema check failed for {jsonl_path}: missing keys {missing} in sample {seen}")
            seen += 1
            if seen >= max_lines:
                break
    if seen == 0:
        raise ValueError(f"Schema check failed: {jsonl_path} has no valid JSONL samples")


def prepare_shared_data_for_cell(args, shared_data_root: Path, i: int, j: int, infer_mode: str) -> Path:
    data_dir = shared_cell_data_dir(shared_data_root, i, j)
    ensure_dir(data_dir)
    cache_root = Path(args.shared_cache_root) if args.shared_cache_root else (shared_data_root / "_cache")
    ensure_dir(cache_root)
    jsonls = generated_jsonl_paths(shared_data_root, args.img_perturb, args.txt_perturb, i, j)
    selected_jsonl = jsonls[infer_mode]

    need_generate = not selected_jsonl.exists()
    if need_generate or not args.skip_existing:
        cmd = [
            args.python_bin,
            args.make_script,
            "--input_jsonl", str(args.base_jsonl),
            "--out_dir", str(data_dir),
            "--shared_cache_root", str(cache_root),
            "--img_perturb", args.img_perturb,
            "--txt_perturb", args.txt_perturb,
            "--img_severity", str(i),
            "--txt_severity", str(j),
            "--target_mode", infer_mode,
            "--seed", str(args.seed),
            "--max_samples", str(args.max_samples),
            "--save_format", args.save_format,
            "--calib_cache_dir", str(cache_root / "calibration"),
        ]
        if args.skip_existing:
            cmd.append("--skip_existing")
        if args.auto_calibrate:
            cmd.append("--auto_calibrate")
        if args.use_calibration:
            cmd.append("--use_calibration")
        cmd.extend(["--calib_samples", str(args.calib_samples), "--calib_metric", args.calib_metric])
        run_cmd(cmd, dry_run=args.dry_run, desc=f"prepare data cell=({i},{j}) mode={infer_mode}")
    else:
        print(f"[skip] prepare data cell=({i},{j}) mode={infer_mode}")

    return selected_jsonl


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True, help="Experiment result directory")
    ap.add_argument("--shared_data_root", default=None, help="Shared perturbed data directory")
    ap.add_argument("--shared_cache_root", default=None, help="Optional global cache root shared across perturbation pairs")
    ap.add_argument("--shared_image_root", default=None, help="Compatibility placeholder for older batch scripts")
    ap.add_argument("--dataset", required=True, help="Dataset name or custom dataset key")
    ap.add_argument("--base_jsonl", default=None, help="Base OpenQA jsonl used to generate perturbed data")
    ap.add_argument("--dataset_config", default=None, help="Optional custom dataset config JSON for inference scripts")
    ap.add_argument("--img_perturb", required=True)
    ap.add_argument("--txt_perturb", required=True)
    ap.add_argument("--severities", default="0,2,3,4")
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--model_name", default="model")

    ap.add_argument("--prepare_data_only", action="store_true")
    ap.add_argument("--reuse_data_only", action="store_true")

    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--make_script", default=str(ROUTE_DIR / "make_perturbed_jsonl.py"))
    ap.add_argument("--infer_script", default=None)
    ap.add_argument("--eval_script", default=str(EVAL_DIR / "eval_metrics_openqa.py"))
    ap.add_argument("--bootstrap_script", default=str(EVAL_DIR / "bootstrap_interaction.py"))
    ap.add_argument("--excess_script", default=str(EVAL_DIR / "excess_interaction.py"))
    ap.add_argument("--compute_new_metrics_script", default=str(EVAL_DIR / "compute_new_metrics.py"))
    ap.add_argument("--heatmap_script", default=str(EVAL_DIR / "heatmap.py"))
    ap.add_argument("--summary_figure_script", default=str(EVAL_DIR / "summary_figure.py"))

    ap.add_argument("--model_path", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--realworldqa_root", default=None)
    ap.add_argument("--parquet_paths", nargs="*", default=None)
    ap.add_argument("--image_save_dir", default=None)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=1000)
    ap.add_argument("--save_format", default="jpg", choices=["jpg", "png"])
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--auto_calibrate", action="store_true")
    ap.add_argument("--use_calibration", action="store_true")
    ap.add_argument("--calib_samples", type=int, default=64)
    ap.add_argument("--calib_metric", default="psnr", choices=["psnr", "ssim"])

    ap.add_argument("--tau", default="0.5,0.8")
    ap.add_argument("--metric", default="f1")
    ap.add_argument("--bootstrap_B", type=int, default=1000)
    ap.add_argument("--run_bootstrap", action="store_true")
    ap.add_argument("--run_excess", action="store_true")
    ap.add_argument("--run_new_metrics", action="store_true")
    ap.add_argument("--run_heatmap", action="store_true")
    ap.add_argument("--run_summary_figure", action="store_true")
    ap.add_argument("--new_metrics_root", default=None)
    ap.add_argument("--photo_root", default=None)
    ap.add_argument("--heatmap_stem", default=None)

    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.infer_script:
        args.infer_script = default_infer_script_for_model(args.model_name)

    exp_root = Path(args.exp_root)
    shared_data_root = Path(args.shared_data_root) if args.shared_data_root else exp_root / "shared_data"

    ensure_dir(exp_root)
    ensure_dir(exp_root / "reports")
    ensure_dir(exp_root / "bootstrap_ready")
    ensure_dir(shared_data_root)

    severities = sorted({int(value.strip()) for value in args.severities.split(",") if value.strip()})
    if 0 not in severities:
        severities = [0] + severities

    if not args.reuse_data_only and args.base_jsonl:
        validate_openqa_jsonl(Path(args.base_jsonl))
    if args.base_jsonl is None and not args.reuse_data_only and not args.prepare_data_only:
        print("[warn] --base_jsonl is not set. This is fine only if your shared perturbed data already exists.")

    prefix = args.prefix or f"{args.img_perturb}_{args.txt_perturb}"
    print("\n=== Run Matrix Summary ===")
    print(f"dataset            : {args.dataset}")
    print(f"dataset_config     : {args.dataset_config or '-'}")
    print(f"model_name         : {args.model_name}")
    print(f"infer_script       : {args.infer_script}")
    print(f"shared_data_root   : {shared_data_root}")
    print(f"exp_root           : {exp_root}")
    print(f"severities         : {severities}")
    print(f"prefix             : {prefix}")
    print(f"dry_run            : {args.dry_run}")

    dump_json(shared_data_root / "manifest.json", {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "base_jsonl": args.base_jsonl,
        "img_perturb": args.img_perturb,
        "txt_perturb": args.txt_perturb,
        "severities": severities,
        "prefix": prefix,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "shared_data_root": str(shared_data_root),
        "shared_cache_root": str(Path(args.shared_cache_root)) if args.shared_cache_root else None,
    })
    dump_json(exp_root / "manifest.json", {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "model_name": args.model_name,
        "exp_root": str(exp_root),
        "shared_data_root": str(shared_data_root),
        "shared_cache_root": str(Path(args.shared_cache_root)) if args.shared_cache_root else None,
        "infer_script": args.infer_script,
        "eval_script": args.eval_script,
        "bootstrap_script": args.bootstrap_script,
        "excess_script": args.excess_script,
        "compute_new_metrics_script": args.compute_new_metrics_script,
        "heatmap_script": args.heatmap_script,
        "summary_figure_script": args.summary_figure_script,
        "prefix": prefix,
    })

    plan_items = build_needed_cells(severities)
    total_cells = len(plan_items)

    for index, (i, j, infer_mode) in enumerate(progress_iter(plan_items, total=total_cells, desc="PrepareData"), start=1):
        print(f"\n=== Data {index}/{total_cells}: ({i}, {j}) mode={infer_mode} ===")
        selected_jsonl = generated_jsonl_paths(shared_data_root, args.img_perturb, args.txt_perturb, i, j)[infer_mode]
        if args.reuse_data_only:
            if not selected_jsonl.exists():
                raise FileNotFoundError(f"Missing shared data: {selected_jsonl}")
            print(f"[reuse] {selected_jsonl}")
        elif args.base_jsonl:
            prepare_shared_data_for_cell(args, shared_data_root, i, j, infer_mode)
        elif not selected_jsonl.exists():
            raise FileNotFoundError(f"Missing shared data and no --base_jsonl was provided: {selected_jsonl}")

    if args.prepare_data_only:
        print("\nDone prepare_data_only.")
        print(f"[summary] shared_data_root: {shared_data_root}")
        return

    summary_rows: List[Dict[str, object]] = []
    for index, (i, j, infer_mode) in enumerate(progress_iter(plan_items, total=total_cells, desc="InferEval"), start=1):
        print(f"\n=== Result {index}/{total_cells}: ({i}, {j}) mode={infer_mode} ===")
        selected_jsonl = generated_jsonl_paths(shared_data_root, args.img_perturb, args.txt_perturb, i, j)[infer_mode]
        if not selected_jsonl.exists():
            raise FileNotFoundError(f"Shared data not found: {selected_jsonl}")

        preds_dir = result_cell_pred_dir(exp_root, i, j)
        ensure_dir(preds_dir)
        pred_out = pred_path(exp_root, i, j, infer_mode, prefix)
        metric_out = resolve_metrics_path_from_pred(pred_out)

        if not pred_out.exists() or not args.skip_existing:
            cmd = [
                args.python_bin,
                args.infer_script,
                "--dataset", args.dataset,
                "--jsonl_path", str(selected_jsonl),
                "--output_path", str(pred_out),
                "--max_new_tokens", str(args.max_new_tokens),
            ]
            if args.dataset_config:
                cmd.extend(["--dataset_config", args.dataset_config])
            if args.model_path:
                cmd.extend(["--model_path", args.model_path])
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            if args.realworldqa_root:
                cmd.extend(["--realworldqa_root", args.realworldqa_root])
            if args.image_save_dir:
                cmd.extend(["--image_save_dir", args.image_save_dir])
            if args.parquet_paths:
                cmd.extend(["--parquet_paths", *args.parquet_paths])
            run_cmd(cmd, dry_run=args.dry_run, desc=f"infer cell=({i},{j}) mode={infer_mode}")
        else:
            print(f"[skip] infer cell=({i},{j}) mode={infer_mode}")

        metric_out = resolve_metrics_path_from_pred(pred_out)
        if not metric_out.exists() or not args.skip_existing:
            cmd = [args.python_bin, args.eval_script, "--pred_path", str(pred_out), "--tau", args.tau, "--save_metrics"]
            run_cmd(cmd, dry_run=args.dry_run, desc=f"eval cell=({i},{j}) mode={infer_mode}")
        else:
            print(f"[skip] eval cell=({i},{j}) mode={infer_mode}")

        metric_out = resolve_metrics_path_from_pred(pred_out)
        metrics = read_metrics_if_exists(metric_out) if not args.dry_run else {}
        if infer_mode != "clean" and not args.dry_run:
            boot_dir = exp_root / "bootstrap_ready" / f"{i}_{j}"
            maybe_link_or_copy(pred_out, boot_dir / f"{prefix}.jsonl")
            if metric_out.exists():
                maybe_link_or_copy(metric_out, boot_dir / f"{prefix}_metrics.json")

        print(f"[cell done] ({i}, {j}) mode={infer_mode} | pred={pred_out} | metrics={metric_out}")
        summary_rows.append({
            "cell": f"{i}_{j}",
            "mode": infer_mode,
            "pred_path": str(pred_out),
            "metrics_path": str(metric_out),
            "em": metrics.get("em", ""),
            "f1": metrics.get("f1", ""),
            "relaxed_acc@0.5": metrics.get("relaxed_acc@0.5", ""),
            "relaxed_acc@0.8": metrics.get("relaxed_acc@0.8", ""),
        })

    summary_csv = exp_root / "reports" / "metrics_summary.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cell", "mode", "pred_path", "metrics_path", "em", "f1", "relaxed_acc@0.5", "relaxed_acc@0.8"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n[OK] metrics summary -> {summary_csv}")

    clean_pred = pred_path(exp_root, 0, 0, "clean", prefix)
    clean_metric = resolve_metrics_path_from_pred(clean_pred)
    bootstrap_txt = None
    excess_txt = None
    new_metrics_json = None

    if args.run_excess:
        excess_txt = exp_root / "reports" / f"excess_{args.metric}.txt"
        cmd = [
            args.python_bin,
            args.excess_script,
            "--root_dir", str(exp_root / "bootstrap_ready"),
            "--prefix", prefix,
            "--metric", args.metric,
            "--clean_metrics", str(clean_metric),
            "--out_txt", str(excess_txt),
        ]
        run_cmd(cmd, dry_run=args.dry_run, desc=f"excess summary metric={args.metric}")

    if args.run_bootstrap:
        bootstrap_txt = exp_root / "reports" / f"bootstrap_{args.metric}.txt"
        cmd = [
            args.python_bin,
            args.bootstrap_script,
            "--root_dir", str(exp_root / "bootstrap_ready"),
            "--prefix", prefix,
            "--clean_jsonl", str(clean_pred),
            "--metric", args.metric,
            "--B", str(args.bootstrap_B),
            "--seed", str(args.seed),
            "--out_txt", str(bootstrap_txt),
        ]
        run_cmd(cmd, dry_run=args.dry_run, desc=f"bootstrap summary metric={args.metric}")

    run_new_metrics = args.run_new_metrics or args.run_bootstrap
    run_heatmap = args.run_heatmap or args.run_bootstrap
    run_summary_figure = args.run_summary_figure or args.run_bootstrap

    new_metrics_root = Path(args.new_metrics_root) if args.new_metrics_root else exp_root / "reports" / "new_metrics"
    photo_root = Path(args.photo_root) if args.photo_root else exp_root / "reports" / "figures"

    if run_new_metrics and bootstrap_txt is not None:
        ensure_dir(new_metrics_root)
        out_txt = new_metrics_root / f"summary_{args.metric}.txt"
        out_json = new_metrics_root / f"summary_{args.metric}.json"
        out_csv = new_metrics_root / f"per_cell_{args.metric}.csv"
        new_metrics_json = out_json
        cmd = [
            args.python_bin,
            args.compute_new_metrics_script,
            "--bootstrap_txt", str(bootstrap_txt),
            "--out_txt", str(out_txt),
            "--out_json", str(out_json),
            "--out_csv", str(out_csv),
        ]
        run_cmd(cmd, dry_run=args.dry_run, desc=f"new metrics summary metric={args.metric}")

    if run_heatmap and bootstrap_txt is not None:
        ensure_dir(photo_root)
        stem = args.heatmap_stem or f"bootstrap_{args.metric}"
        cmd = [
            args.python_bin,
            args.heatmap_script,
            "--input_txt", str(bootstrap_txt),
            "--out_dir", str(photo_root),
            "--stem", stem,
        ]
        run_cmd(cmd, dry_run=args.dry_run, desc=f"heatmap metric={args.metric}")

    if run_summary_figure:
        ensure_dir(photo_root)
        summary_figure = photo_root / f"summary_{args.metric}.png"
        cmd = [
            args.python_bin,
            args.summary_figure_script,
            "--summary_csv", str(summary_csv),
            "--out_png", str(summary_figure),
            "--metric", args.metric,
        ]
        if bootstrap_txt is not None:
            cmd.extend(["--bootstrap_txt", str(bootstrap_txt)])
        if new_metrics_json is not None:
            cmd.extend(["--summary_json", str(new_metrics_json)])
        run_cmd(cmd, dry_run=args.dry_run, desc=f"summary figure metric={args.metric}")

    print("\nDone.")
    print(f"[summary] shared_data_root: {shared_data_root}")
    print(f"[summary] result_root: {exp_root}")
    print(f"[summary] metrics_summary: {summary_csv}")
    if bootstrap_txt is not None:
        print(f"[summary] bootstrap_txt: {bootstrap_txt}")
    if excess_txt is not None:
        print(f"[summary] excess_txt: {excess_txt}")
    if run_new_metrics:
        print(f"[summary] new_metrics_root: {new_metrics_root}")
    if run_heatmap or run_summary_figure:
        print(f"[summary] figures_root: {photo_root}")


if __name__ == "__main__":
    main()

