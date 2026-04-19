#!/usr/bin/env python3
import argparse
import gc
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
from PIL import Image, ImageFile
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from data.load_dataset import load_dataset_openqa
from runtime_utils import progress_bar, print_kv_block, print_output_block


ImageFile.LOAD_TRUNCATED_IMAGES = True
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(SCRIPT_DIR / "models" / "Qwen3-VL-8B-Instruct")
DEFAULT_DATA_ROOT = str(SCRIPT_DIR.parent.parent / "dataset" / "RealworldQA" / "data")
DEFAULT_IMAGE_SAVE_DIR = str(SCRIPT_DIR.parent.parent / "dataset" / "RealworldQA" / "images_cache")
DEFAULT_OUTPUT_PATH = str(SCRIPT_DIR / "outputs" / "qwen3_openqa.jsonl")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resize_image_for_vl(image: Image.Image, max_side: int = 1024) -> Image.Image:
    width, height = image.size
    if max(width, height) <= max_side:
        return image
    scale = max_side / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.BICUBIC)


def init_model_and_processor(model_path: str, use_flash_attention: bool, min_pixels: int, max_pixels: int):
    print(f"Loading Qwen3-VL model: {model_path}")
    try:
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        return model, processor
    except Exception as exc:
        print(f"Model loading failed: {exc}")
        sys.exit(1)


def build_openqa_prompt(sample: dict) -> list:
    text_prompt = (
        "Answer the following question with a short, direct answer.\n"
        "Do not include any explanation.\n\n"
        f"Question: {sample['question']}"
    )
    messages = [{"role": "user", "content": []}]
    if sample.get("image_path") and os.path.exists(sample["image_path"]):
        messages[0]["content"].append({"type": "image", "image": sample["image_path"]})
    messages[0]["content"].append({"type": "text", "text": text_prompt})
    return messages


def infer_single_sample(model, processor, sample: dict, max_new_tokens: int, image_max_side: int) -> str:
    has_image = bool(sample.get("image_path")) and os.path.exists(sample["image_path"])
    messages = build_openqa_prompt(sample)

    if has_image:
        raw_image = Image.open(sample["image_path"]).convert("RGB")
        raw_image = resize_image_for_vl(raw_image, max_side=image_max_side)
        for content in messages[0]["content"]:
            if content["type"] == "image":
                content["image"] = raw_image

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        target_device = next(model.parameters()).device
        inputs = {key: value.to(target_device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            use_cache=True,
            pad_token_id=(
                processor.tokenizer.pad_token_id
                if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is not None
                else model.generation_config.pad_token_id
            ),
            eos_token_id=(
                processor.tokenizer.eos_token_id
                if hasattr(processor, "tokenizer") and processor.tokenizer.eos_token_id is not None
                else model.generation_config.eos_token_id
            ),
        )

    input_ids = inputs["input_ids"]
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    pred = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return pred


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--realworldqa_root", type=str, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--parquet_paths", type=str, nargs="*", default=None)
    ap.add_argument("--image_save_dir", type=str, default=DEFAULT_IMAGE_SAVE_DIR)
    ap.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--use_flash_attention", action="store_true")
    ap.add_argument("--dataset", type=str, default="realworldqa")
    ap.add_argument("--jsonl_path", type=str, default=None, help="Path to a processed OpenQA jsonl dataset")
    ap.add_argument("--dataset_config", type=str, default=None, help="Path to a custom dataset config JSON")
    ap.add_argument("--image_max_side", type=int, default=1024)
    ap.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    ap.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    return ap.parse_args()


def main():
    args = parse_args()
    print_kv_block(
        "Qwen3-VL OpenQA Inference",
        {
            "dataset": args.dataset,
            "dataset_config": args.dataset_config or "-",
            "model_path": args.model_path,
            "output_path": args.output_path,
            "limit": args.limit,
            "max_new_tokens": args.max_new_tokens,
            "image_max_side": args.image_max_side,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "flash_attention": args.use_flash_attention,
        },
    )

    model, processor = init_model_and_processor(
        model_path=args.model_path,
        use_flash_attention=args.use_flash_attention,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
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
    oom_count = 0

    bar = progress_bar(samples, total=len(samples), desc="Qwen3-VL OpenQA", unit="sample")
    write_log = getattr(bar, "write", print)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for sample in bar:
            pred = ""
            try:
                pred = infer_single_sample(
                    model=model,
                    processor=processor,
                    sample=sample,
                    max_new_tokens=args.max_new_tokens,
                    image_max_side=args.image_max_side,
                )
            except torch.cuda.OutOfMemoryError as exc:
                oom_count += 1
                fail_count += 1
                write_log(f"CUDA OOM on sample {sample.get('id', 'NA')}: {exc}")
                cleanup_cuda()
            except Exception as exc:
                fail_count += 1
                write_log(f"Inference failed for sample {sample.get('id', 'NA')}: {exc}")
                cleanup_cuda()

            out = {
                "id": sample.get("id", ""),
                "question": sample.get("question", ""),
                "gt_text": sample.get("gt_text", ""),
                "pred": pred,
                "image_path": sample.get("image_path", ""),
                "has_image": bool(sample.get("has_image", False)),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            cleanup_cuda()
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(failed=fail_count, oom=oom_count)

    print_output_block(
        "Inference Outputs",
        {
            "samples": len(samples),
            "failed": fail_count,
            "oom": oom_count,
            "succeeded": len(samples) - fail_count,
            "predictions": args.output_path,
        },
    )


if __name__ == "__main__":
    main()
