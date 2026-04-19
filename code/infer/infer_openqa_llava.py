#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from types import ModuleType

try:
    from huggingface_hub import is_offline_mode
except ImportError:
    def is_offline_mode() -> bool:
        return os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"

    if "huggingface_hub" not in sys.modules:
        sys.modules["huggingface_hub"] = ModuleType("huggingface_hub")
    sys.modules["huggingface_hub"].is_offline_mode = is_offline_mode

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from data.load_dataset import load_dataset_openqa
from runtime_utils import progress_bar, print_kv_block, print_output_block


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(SCRIPT_DIR.parent / "llava-1.5-7b-hf")
DEFAULT_DATA_ROOT = str(SCRIPT_DIR.parent.parent / "dataset" / "RealworldQA" / "data")
DEFAULT_IMAGE_SAVE_DIR = str(SCRIPT_DIR.parent.parent / "dataset" / "RealworldQA" / "images_cache")
DEFAULT_OUTPUT_PATH = str(SCRIPT_DIR / "outputs" / "llava15_openqa.jsonl")
TORCH_DTYPE = torch.float16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def init_model_and_processor(model_path: str):
    print(f"Loading LLaVA model: {model_path}")
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=TORCH_DTYPE,
            low_cpu_mem_usage=True,
            local_files_only=True,
            trust_remote_code=True,
        ).to(DEVICE)
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        return model, processor
    except Exception as exc:
        print(f"Model loading failed: {exc}")
        sys.exit(1)


def build_openqa_prompt(sample: dict) -> dict:
    text_prompt = (
        "Answer the following question with a short, direct answer.\n"
        "Do not include any explanation.\n\n"
        f"Question: {sample['question']}"
    )
    message = {"role": "user", "content": []}
    if sample.get("image_path") and os.path.exists(sample["image_path"]):
        message["content"].append({"type": "image", "url": Path(sample["image_path"]).resolve().as_uri()})
    message["content"].append({"type": "text", "text": text_prompt})
    return message


def infer_single_sample(model, processor, sample: dict, max_new_tokens: int) -> str:
    has_image = bool(sample.get("image_path")) and os.path.exists(sample["image_path"])
    raw_image = Image.open(sample["image_path"]).convert("RGB") if has_image else None
    conversation = [build_openqa_prompt(sample)]

    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=[prompt],
        images=raw_image if has_image else None,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE, TORCH_DTYPE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    pred = processor.decode(
        generated_ids_trimmed[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()
    return pred


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--realworldqa_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--parquet_paths", type=str, nargs="*", default=None)
    ap.add_argument("--image_save_dir", type=str, default=DEFAULT_IMAGE_SAVE_DIR)
    ap.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--dataset", type=str, default="realworldqa")
    ap.add_argument("--jsonl_path", type=str, default=None, help="Path to a processed OpenQA jsonl dataset")
    ap.add_argument("--dataset_config", type=str, default=None, help="Path to a custom dataset config JSON")
    return ap.parse_args()


def main():
    args = parse_args()
    print_kv_block(
        "LLaVA-1.5 OpenQA Inference",
        {
            "dataset": args.dataset,
            "dataset_config": args.dataset_config or "-",
            "model_path": args.model_path,
            "output_path": args.output_path,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
        },
    )

    model, processor = init_model_and_processor(args.model_path)
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

    bar = progress_bar(samples, total=len(samples), desc="LLaVA-1.5 OpenQA", unit="sample")
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
