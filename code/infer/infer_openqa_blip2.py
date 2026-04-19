#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from data.load_dataset import load_dataset_openqa
from runtime_utils import progress_bar, print_kv_block, print_output_block


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(SCRIPT_DIR / "models" / "blip2-opt-2.7b")
DEFAULT_DATA_ROOT = str(SCRIPT_DIR.parent.parent / "dataset" / "RealworldQA" / "data")
DEFAULT_IMAGE_SAVE_DIR = str(SCRIPT_DIR.parent.parent / "dataset" / "RealworldQA" / "images_cache")
DEFAULT_OUTPUT_PATH = str(SCRIPT_DIR / "outputs" / "blip2_openqa.jsonl")


def init_model_and_processor(model_path: str, dtype: str = "fp16"):
    print(f"Loading BLIP2 model: {model_path}")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(dtype.lower(), torch.float16)

    processor = Blip2Processor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def build_prompt(sample: dict) -> str:
    question = (sample.get("question") or "").strip()
    return (
        "Answer the following question with a short, direct answer. "
        "Do not include any explanation.\n"
        f"Question: {question}\nAnswer:"
    )


def load_rgb_image(image_path: Optional[str]) -> Optional[Image.Image]:
    if not image_path or not os.path.exists(image_path):
        return None
    return Image.open(image_path).convert("RGB")


def infer_single_sample(model, processor, sample: dict, max_new_tokens: int) -> str:
    image = load_rgb_image(sample.get("image_path"))
    prompt = build_prompt(sample)
    if image is None:
        raise ValueError(f"BLIP2 requires an image, but image not found: {sample.get('image_path', '')}")

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    target_device = model.device if hasattr(model, "device") else next(model.parameters()).device
    for key, value in inputs.items():
        if torch.is_tensor(value):
            inputs[key] = value.to(target_device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    answer_ids = generated_ids[:, input_len:]
    pred = processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()

    lower_pred = pred.lower()
    question = (sample.get("question") or "").strip().lower()
    if question and question in lower_pred:
        pos = lower_pred.rfind(question)
        pred = pred[pos + len(question):].strip(" :\n\t") or pred
    return pred.strip()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--realworldqa_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--parquet_paths", type=str, nargs="*", default=None)
    ap.add_argument("--image_save_dir", type=str, default=DEFAULT_IMAGE_SAVE_DIR)
    ap.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--dataset", type=str, default="realworldqa")
    ap.add_argument("--jsonl_path", type=str, default=None, help="Path to a processed OpenQA jsonl dataset")
    ap.add_argument("--dataset_config", type=str, default=None, help="Path to a custom dataset config JSON")
    return ap.parse_args()


def main():
    args = parse_args()
    print_kv_block(
        "BLIP2 OpenQA Inference",
        {
            "dataset": args.dataset,
            "dataset_config": args.dataset_config or "-",
            "model_path": args.model_path,
            "output_path": args.output_path,
            "dtype": args.dtype,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
        },
    )

    try:
        model, processor = init_model_and_processor(args.model_path, args.dtype)
    except Exception as exc:
        print(f"Model loading failed: {exc}")
        sys.exit(1)

    samples = load_dataset_openqa(
        dataset=args.dataset,
        limit=args.limit,
        realworldqa_root=args.realworldqa_root,
        parquet_paths=args.parquet_paths,
        image_save_dir=args.image_save_dir,
        jsonl_path=args.jsonl_path,
        dataset_config=args.dataset_config,
    )
    if not samples:
        print("No samples loaded. Check dataset arguments.")
        sys.exit(1)

    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    fail_count = 0

    bar = progress_bar(samples, total=len(samples), desc="BLIP2 OpenQA", unit="sample")
    write_log = getattr(bar, "write", print)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for sample in bar:
            try:
                pred = infer_single_sample(model, processor, sample, args.max_new_tokens)
            except Exception as exc:
                write_log(f"Inference failed for sample {sample.get('id', 'NA')}: {exc}")
                pred = ""
                fail_count += 1

            out = {
                "id": sample.get("id", ""),
                "question": sample.get("question", ""),
                "gt_text": sample.get("gt_text", ""),
                "pred": pred,
                "image_path": sample.get("image_path", ""),
                "has_image": bool(sample.get("has_image", False)),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print_output_block(
        "Inference Outputs",
        {
            "samples": len(samples),
            "failed": fail_count,
            "succeeded": len(samples) - fail_count,
            "predictions": args.output_path,
        },
    )


if __name__ == "__main__":
    main()
