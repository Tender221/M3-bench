import importlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional

from data.realworldqa_openqa import load_realworldqa_openqa


def _to_gt_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None:
        return [""]
    return [str(value)]


def _normalize_sample(
    obj: Dict[str, Any],
    *,
    dataset_name: str,
    id_field: str = "id",
    question_field: str = "question",
    answer_field: str = "gt_text",
    image_field: str = "image_path",
    meta_fields: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    image_path = obj.get(image_field, "") or ""
    if meta_fields:
        meta = {key: obj.get(key) for key in meta_fields}
    else:
        skip_keys = {id_field, question_field, answer_field, image_field}
        meta = {k: v for k, v in obj.items() if k not in skip_keys}
    meta["dataset"] = dataset_name
    return {
        "id": str(obj.get(id_field, "")),
        "question": obj.get(question_field, ""),
        "gt_text": _to_gt_list(obj.get(answer_field, "")),
        "image_path": image_path,
        "has_image": bool(image_path) and os.path.exists(image_path),
        "meta": meta,
    }


def load_jsonl_openqa(
    jsonl_path: str,
    limit: Optional[int] = None,
    *,
    dataset_name: str = "custom",
    id_field: str = "id",
    question_field: str = "question",
    answer_field: str = "gt_text",
    image_field: str = "image_path",
    meta_fields: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            samples.append(
                _normalize_sample(
                    obj,
                    dataset_name=dataset_name,
                    id_field=id_field,
                    question_field=question_field,
                    answer_field=answer_field,
                    image_field=image_field,
                    meta_fields=meta_fields,
                )
            )
            if limit and len(samples) >= limit:
                break
    return samples


def _load_python_loader(loader_spec: str):
    if ":" not in loader_spec:
        raise ValueError("dataset loader must be in the form 'module.submodule:function_name'")
    module_name, func_name = loader_spec.split(":", 1)
    module = importlib.import_module(module_name)
    loader = getattr(module, func_name, None)
    if loader is None:
        raise AttributeError(f"Loader function not found: {loader_spec}")
    return loader


def _load_dataset_config(dataset_config: str, dataset_name: str) -> Dict[str, Any]:
    with open(dataset_config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "datasets" in config:
        key = dataset_name or config.get("default_dataset")
        if not key:
            raise ValueError("dataset_config contains multiple datasets; please also pass --dataset")
        if key not in config["datasets"]:
            raise KeyError(f"Dataset '{key}' not found in {dataset_config}")
        spec = dict(config["datasets"][key])
        spec.setdefault("dataset_name", key)
        return spec

    spec = dict(config)
    spec.setdefault("dataset_name", dataset_name or spec.get("name") or "custom")
    return spec


def _load_custom_dataset_from_spec(
    spec: Dict[str, Any],
    *,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    spec_type = (spec.get("type") or "jsonl").lower()
    dataset_name = spec.get("dataset_name") or spec.get("name") or "custom"

    if spec_type == "jsonl":
        jsonl_path = spec.get("jsonl_path")
        if not jsonl_path:
            raise ValueError("jsonl dataset spec requires 'jsonl_path'")
        return load_jsonl_openqa(
            jsonl_path=jsonl_path,
            limit=limit,
            dataset_name=dataset_name,
            id_field=spec.get("id_field", "id"),
            question_field=spec.get("question_field", "question"),
            answer_field=spec.get("answer_field", "gt_text"),
            image_field=spec.get("image_field", "image_path"),
            meta_fields=spec.get("meta_fields"),
        )

    if spec_type == "python":
        loader = _load_python_loader(spec["loader"])
        kwargs = dict(spec.get("kwargs", {}))
        if limit is not None and "limit" not in kwargs:
            kwargs["limit"] = limit
        records = loader(**kwargs)
        if limit is not None:
            records = records[:limit]
        return [
            _normalize_sample(
                record,
                dataset_name=dataset_name,
                id_field=spec.get("id_field", "id"),
                question_field=spec.get("question_field", "question"),
                answer_field=spec.get("answer_field", "gt_text"),
                image_field=spec.get("image_field", "image_path"),
                meta_fields=spec.get("meta_fields"),
            )
            for record in records
        ]

    raise ValueError(f"Unsupported dataset spec type: {spec_type}")


def load_dataset_openqa(
    dataset: str,
    limit: Optional[int] = None,
    realworldqa_root: Optional[str] = None,
    parquet_paths: Optional[list] = None,
    image_save_dir: Optional[str] = None,
    jsonl_path: Optional[str] = None,
    dataset_config: Optional[str] = None,
) -> List[Dict[str, Any]]:
    dataset_name = (dataset or "").strip() or "custom"
    dataset_key = dataset_name.lower()

    if dataset_config:
        spec = _load_dataset_config(dataset_config, dataset_name)
        return _load_custom_dataset_from_spec(spec, limit=limit)

    if dataset_key == "realworldqa":
        if realworldqa_root is None:
            raise ValueError("realworldqa_root is required for dataset=realworldqa")
        if image_save_dir is None:
            raise ValueError("image_save_dir is required for dataset=realworldqa")

        samples = load_realworldqa_openqa(
            data_root=realworldqa_root,
            parquet_paths=parquet_paths,
            image_save_dir=image_save_dir,
            limit=limit,
        )
        return [_normalize_sample(sample, dataset_name="realworldqa") for sample in samples]

    if jsonl_path:
        return load_jsonl_openqa(
            jsonl_path=jsonl_path,
            limit=limit,
            dataset_name=dataset_key,
        )

    raise ValueError(
        f"Unknown dataset '{dataset_name}'. Supported: realworldqa or any dataset provided via --jsonl_path / --dataset_config"
    )
