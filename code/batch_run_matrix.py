#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output"
PYTHON_BIN = sys.executable
RUN_MATRIX = SCRIPT_DIR / "run_matrix.py"

DEFAULT_RUN_BOOTSTRAP = True
DEFAULT_RUN_EXCESS = True
DEFAULT_BOOTSTRAP_B = 1000
DEFAULT_METRIC = "f1"
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_MAX_SAMPLES = 1000
DEFAULT_SEVERITIES = [0, 1, 2, 3, 4, 5]
DEFAULT_SAVE_FORMAT = "jpg"
DEFAULT_TAU = "0.5,0.8"
DEFAULT_SKIP_EXISTING = True


@dataclass
class DatasetSpec:
    name: str
    output_dirname: str
    base_jsonl: str
    dataset_config: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)


@dataclass
class ModelSpec:
    key: str
    name: str
    exp_subdir: str
    infer_script: str
    model_path: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)


DATASETS: Dict[str, DatasetSpec] = {
    "okvqa": DatasetSpec(
        name="okvqa",
        output_dirname="okvqa",
        base_jsonl=str(PROJECT_ROOT / "dataset" / "OK-VQA" / "okvqa_val_openqa.jsonl"),
    ),
    "gqa": DatasetSpec(
        name="gqa",
        output_dirname="gqa",
        base_jsonl=str(PROJECT_ROOT / "dataset" / "GQA" / "gqa_val_openqa.jsonl"),
    ),
}

MODELS: Dict[str, ModelSpec] = {
    "qwen25": ModelSpec(
        key="qwen25",
        name="Qwen2.5-VL-7B",
        exp_subdir="Qwen2.5-VL-7B",
        infer_script=str(SCRIPT_DIR / "infer" / "infer_openqa_qwen25.py"),
        model_path=str(SCRIPT_DIR / "infer" / "models" / "Qwen2.5-VL-7B"),
    ),
    "qwen3": ModelSpec(
        key="qwen3",
        name="Qwen3-VL-8B",
        exp_subdir="Qwen3-VL-8B",
        infer_script=str(SCRIPT_DIR / "infer" / "infer_openqa_qwen3.py"),
        model_path=str(SCRIPT_DIR / "infer" / "models" / "Qwen3-VL-8B-Instruct"),
    ),
    "llava15": ModelSpec(
        key="llava15",
        name="LLaVA-1.5-7B",
        exp_subdir="LLaVA-1.5-7B",
        infer_script=str(SCRIPT_DIR / "infer" / "infer_openqa_llava.py"),
        model_path=str(SCRIPT_DIR / "llava-1.5-7b-hf"),
    ),
    "blip2": ModelSpec(
        key="blip2",
        name="BLIP2-opt-2.7B",
        exp_subdir="BLIP2-opt-2.7B",
        infer_script=str(SCRIPT_DIR / "infer" / "infer_openqa_blip2.py"),
        model_path=str(SCRIPT_DIR / "infer" / "models" / "blip2-opt-2.7b"),
    ),
}

PAIRS: List[Tuple[str, str]] = [
    ("jpeg", "ocr"),
    ("jpeg", "whitespace_punc"),
    ("gaussian_blur", "typo"),
    ("gaussian_blur", "whitespace_punc"),
    ("gaussian_noise", "keyboard"),
    ("brightness", "ocr"),
]

SELECT_MODELS = ["qwen25", "qwen3", "llava15", "blip2"]
SELECT_DATASETS = ["okvqa", "gqa"]
SELECT_PAIRS = [f"{a}_{b}" for a, b in PAIRS]


def quote_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_one(cmd: List[str], dry_run: bool = False) -> None:
    print("\n>>>", quote_cmd(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def csv_has_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with open(path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    return len(lines) >= 2


def file_has_content(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def experiment_complete(exp_root: Path, need_bootstrap: bool, need_excess: bool, metric: str) -> bool:
    summary_csv = exp_root / "reports" / "metrics_summary.csv"
    bootstrap_txt = exp_root / "reports" / f"bootstrap_{metric}.txt"
    excess_txt = exp_root / "reports" / f"excess_{metric}.txt"
    if not csv_has_rows(summary_csv):
        return False
    if need_bootstrap and not file_has_content(bootstrap_txt):
        return False
    if need_excess and not file_has_content(excess_txt):
        return False
    return True


def build_shared_root(output_root: Path, dataset_dirname: str, pair_name: str) -> Path:
    return output_root / dataset_dirname / "shared" / f"{pair_name}_grid"


def build_exp_root(output_root: Path, dataset_dirname: str, model_subdir: str, pair_name: str) -> Path:
    return output_root / dataset_dirname / model_subdir / f"{pair_name}_grid"


def load_user_datasets(path: Optional[str]) -> Dict[str, DatasetSpec]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("datasets", payload)
    result = {}
    for key, spec in items.items():
        result[key] = DatasetSpec(
            name=spec.get("name", key),
            output_dirname=spec.get("output_dirname", key),
            base_jsonl=spec.get("base_jsonl", ""),
            dataset_config=spec.get("dataset_config"),
            extra_args=list(spec.get("extra_args", [])),
        )
    return result


def load_user_models(path: Optional[str]) -> Dict[str, ModelSpec]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("models", payload)
    result = {}
    for key, spec in items.items():
        result[key] = ModelSpec(
            key=spec.get("key", key),
            name=spec.get("name", key),
            exp_subdir=spec.get("exp_subdir", spec.get("name", key)),
            infer_script=spec["infer_script"],
            model_path=spec.get("model_path"),
            extra_args=list(spec.get("extra_args", [])),
        )
    return result


def build_prepare_cmd(
    output_root: Path,
    dataset: DatasetSpec,
    img_perturb: str,
    txt_perturb: str,
    pair_name: str,
    shared_cache_root: Optional[Path],
    severities: List[int],
    max_samples: int,
    save_format: str,
    seed: int,
    dry_run: bool,
) -> List[str]:
    shared_root = build_shared_root(output_root, dataset.output_dirname, pair_name)
    cmd = [
        PYTHON_BIN,
        str(RUN_MATRIX),
        "--shared_data_root", str(shared_root),
        "--exp_root", str(shared_root / "_prepare_only_dummy_exp"),
        "--dataset", dataset.name,
        "--base_jsonl", dataset.base_jsonl,
        "--img_perturb", img_perturb,
        "--txt_perturb", txt_perturb,
        "--severities", ",".join(map(str, severities)),
        "--prefix", pair_name,
        "--max_samples", str(max_samples),
        "--save_format", save_format,
        "--seed", str(seed),
        "--prepare_data_only",
    ]
    if shared_cache_root is not None:
        cmd.extend(["--shared_cache_root", str(shared_cache_root)])
    if dataset.dataset_config:
        cmd.extend(["--dataset_config", dataset.dataset_config])
    if DEFAULT_SKIP_EXISTING:
        cmd.append("--skip_existing")
    cmd.extend(dataset.extra_args)
    if dry_run:
        cmd.append("--dry_run")
    return cmd


def build_model_cmd(
    output_root: Path,
    dataset: DatasetSpec,
    model: ModelSpec,
    img_perturb: str,
    txt_perturb: str,
    pair_name: str,
    shared_cache_root: Optional[Path],
    severities: List[int],
    max_samples: int,
    save_format: str,
    seed: int,
    metric: str,
    max_new_tokens: int,
    bootstrap_B: int,
    run_bootstrap: bool,
    run_excess: bool,
    tau: str,
    dry_run: bool,
) -> List[str]:
    shared_root = build_shared_root(output_root, dataset.output_dirname, pair_name)
    exp_root = build_exp_root(output_root, dataset.output_dirname, model.exp_subdir, pair_name)
    cmd = [
        PYTHON_BIN,
        str(RUN_MATRIX),
        "--shared_data_root", str(shared_root),
        "--exp_root", str(exp_root),
        "--dataset", dataset.name,
        "--img_perturb", img_perturb,
        "--txt_perturb", txt_perturb,
        "--severities", ",".join(map(str, severities)),
        "--prefix", pair_name,
        "--model_name", model.name,
        "--infer_script", model.infer_script,
        "--max_new_tokens", str(max_new_tokens),
        "--max_samples", str(max_samples),
        "--save_format", save_format,
        "--seed", str(seed),
        "--tau", tau,
        "--metric", metric,
        "--bootstrap_B", str(bootstrap_B),
        "--reuse_data_only",
    ]
    if shared_cache_root is not None:
        cmd.extend(["--shared_cache_root", str(shared_cache_root)])
    if dataset.dataset_config:
        cmd.extend(["--dataset_config", dataset.dataset_config])
    if model.model_path:
        cmd.extend(["--model_path", model.model_path])
    if DEFAULT_SKIP_EXISTING:
        cmd.append("--skip_existing")
    if run_bootstrap:
        cmd.append("--run_bootstrap")
    if run_excess:
        cmd.append("--run_excess")
    cmd.extend(dataset.extra_args)
    cmd.extend(model.extra_args)
    if dry_run:
        cmd.append("--dry_run")
    return cmd


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only_models", default=None)
    ap.add_argument("--only_datasets", default=None)
    ap.add_argument("--only_pairs", default=None)
    ap.add_argument("--models_config", default=None, help="Path to a custom model registry JSON")
    ap.add_argument("--datasets_config", default=None, help="Path to a custom dataset registry JSON")
    ap.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--severities", default=",".join(map(str, DEFAULT_SEVERITIES)))
    ap.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    ap.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--metric", default=DEFAULT_METRIC)
    ap.add_argument("--bootstrap_B", type=int, default=DEFAULT_BOOTSTRAP_B)
    ap.add_argument("--tau", default=DEFAULT_TAU)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shared_cache_root", default=None,
                    help="Optional global cache root shared across different perturbation pairs")
    ap.add_argument("--no_bootstrap", action="store_true")
    ap.add_argument("--no_excess", action="store_true")
    ap.add_argument("--save_format", default=DEFAULT_SAVE_FORMAT, choices=["jpg", "png"])
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_cli()
    output_root = Path(args.output_root)

    datasets = dict(DATASETS)
    datasets.update(load_user_datasets(args.datasets_config))
    models = dict(MODELS)
    models.update(load_user_models(args.models_config))

    selected_models = args.only_models.split(",") if args.only_models else SELECT_MODELS
    selected_datasets = args.only_datasets.split(",") if args.only_datasets else SELECT_DATASETS
    selected_pairs = args.only_pairs.split(",") if args.only_pairs else SELECT_PAIRS
    pair_map = {f"{a}_{b}": (a, b) for a, b in PAIRS}

    severities = sorted({int(value.strip()) for value in args.severities.split(",") if value.strip()})
    if 0 not in severities:
        severities = [0] + severities

    run_bootstrap = DEFAULT_RUN_BOOTSTRAP and (not args.no_bootstrap)
    run_excess = DEFAULT_RUN_EXCESS and (not args.no_excess)

    print("=" * 88)
    print("Batch runner summary")
    print(f"models        : {selected_models}")
    print(f"datasets      : {selected_datasets}")
    print(f"pairs         : {selected_pairs}")
    print(f"severity      : {severities}")
    print(f"output_root   : {output_root}")
    print(f"dry_run       : {args.dry_run}")
    print("=" * 88)

    if not RUN_MATRIX.exists():
        raise FileNotFoundError(f"run_matrix.py not found: {RUN_MATRIX}")

    for dataset_key in selected_datasets:
        if dataset_key not in datasets:
            raise KeyError(f"Unknown dataset: {dataset_key}")
        dataset = datasets[dataset_key]
        if not dataset.base_jsonl and not dataset.dataset_config:
            raise ValueError(f"dataset[{dataset_key}] must provide base_jsonl or dataset_config")

        for pair_name in selected_pairs:
            if pair_name not in pair_map:
                raise KeyError(f"Unknown perturbation pair: {pair_name}")
            img_perturb, txt_perturb = pair_map[pair_name]
            shared_root = build_shared_root(output_root, dataset.output_dirname, pair_name)
            manifest = shared_root / "manifest.json"
            if manifest.exists():
                print(f"[skip prepare] shared data seems present: {shared_root}")
                continue
            print(f"[prepare] dataset={dataset.output_dirname} pair={pair_name}")
            cmd = build_prepare_cmd(
                output_root=output_root,
                dataset=dataset,
                img_perturb=img_perturb,
                txt_perturb=txt_perturb,
                pair_name=pair_name,
                shared_cache_root=Path(args.shared_cache_root) if args.shared_cache_root else None,
                severities=severities,
                max_samples=args.max_samples,
                save_format=args.save_format,
                seed=args.seed,
                dry_run=args.dry_run,
            )
            run_one(cmd, dry_run=args.dry_run)

    for dataset_key in selected_datasets:
        dataset = datasets[dataset_key]
        for model_key in selected_models:
            if model_key not in models:
                raise KeyError(f"Unknown model: {model_key}")
            model = models[model_key]

            for pair_name in selected_pairs:
                img_perturb, txt_perturb = pair_map[pair_name]
                exp_root = build_exp_root(output_root, dataset.output_dirname, model.exp_subdir, pair_name)
                if experiment_complete(exp_root, need_bootstrap=run_bootstrap, need_excess=run_excess, metric=args.metric):
                    print(f"[skip done] dataset={dataset.output_dirname} model={model.key} pair={pair_name}")
                    continue
                print(f"[run] dataset={dataset.output_dirname} model={model.key} pair={pair_name}")
                cmd = build_model_cmd(
                    output_root=output_root,
                    dataset=dataset,
                    model=model,
                    img_perturb=img_perturb,
                    txt_perturb=txt_perturb,
                    pair_name=pair_name,
                    shared_cache_root=Path(args.shared_cache_root) if args.shared_cache_root else None,
                    severities=severities,
                    max_samples=args.max_samples,
                    save_format=args.save_format,
                    seed=args.seed,
                    metric=args.metric,
                    max_new_tokens=args.max_new_tokens,
                    bootstrap_B=args.bootstrap_B,
                    run_bootstrap=run_bootstrap,
                    run_excess=run_excess,
                    tau=args.tau,
                    dry_run=args.dry_run,
                )
                run_one(cmd, dry_run=args.dry_run)

    print("\nAll tasks have been processed.")


if __name__ == "__main__":
    main()
