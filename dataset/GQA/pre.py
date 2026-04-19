#!/usr/bin/env python3
import os
import io
import json
import glob
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

def decode_image_bytes_to_jpg(jpeg_bytes: bytes, out_path: str):
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img.save(out_path, format="JPEG", quality=95)

def iter_parquets(path_or_glob: str):
    """
    支持：
      - 具体文件：/a/b/x.parquet
      - 目录：/a/b/val_balanced_images
      - glob：/a/b/val_balanced_images/*.parquet
    """
    if os.path.isdir(path_or_glob):
        files = sorted(glob.glob(os.path.join(path_or_glob, "*.parquet")))
    elif "*" in path_or_glob or "?" in path_or_glob or "[" in path_or_glob:
        files = sorted(glob.glob(path_or_glob))
    else:
        files = [path_or_glob]
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {path_or_glob}")
    return files

def build_image_index(img_parquet_path_or_dir: str):
    img_bytes = {}
    parquet_files = iter_parquets(img_parquet_path_or_dir)

    for p in parquet_files:
        df = pd.read_parquet(p)
        # 你这里 images parquet 是 ['id', 'image']
        if "id" not in df.columns or "image" not in df.columns:
            raise ValueError(f"Unexpected columns in {p}: {list(df.columns)}")

        df["id"] = df["id"].astype(str)

        for _, row in df.iterrows():
            image_id = row["id"]
            im = row["image"]
            b = im["bytes"] if isinstance(im, dict) else im.bytes
            img_bytes[image_id] = b

    return img_bytes, parquet_files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ins_parquet", required=True, help="val_balanced_instructions parquet path")
    ap.add_argument("--img_parquets", required=True,
                    help="val_balanced_images: dir OR glob OR single parquet")
    ap.add_argument("--out_dir", required=True, help="output root dir")
    ap.add_argument("--images_subdir", default="images_cache", help="subdir for decoded images")
    ap.add_argument("--out_jsonl", default="gqa_val_balanced_openqa.jsonl", help="jsonl filename under out_dir")
    ap.add_argument("--overwrite_images", action="store_true", help="overwrite existing jpgs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    images_dir = os.path.join(args.out_dir, args.images_subdir)
    os.makedirs(images_dir, exist_ok=True)
    out_jsonl_path = os.path.join(args.out_dir, args.out_jsonl)

    # 1) load instructions
    ins_df = pd.read_parquet(args.ins_parquet)
    ins_df["imageId"] = ins_df["imageId"].astype(str)
    ins_df["id"] = ins_df["id"].astype(str)

    # 2) load all image shards
    img_bytes, used_files = build_image_index(args.img_parquets)
    print(f"Loaded {len(img_bytes)} images from {len(used_files)} parquet shards:")
    for p in used_files:
        print("  -", p)

    # 3) write images + jsonl
    n_missing = 0
    n_written_images = 0

    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for _, r in tqdm(ins_df.iterrows(), total=len(ins_df), desc="Processing GQA val_balanced"):
            qid = str(r["id"])
            image_id = str(r["imageId"])
            question = r["question"]
            gt = r["answer"]  # ✅ 用短答案做 EM 最稳

            b = img_bytes.get(image_id, None)
            if b is None:
                n_missing += 1
                continue

            img_path = os.path.join(images_dir, f"{image_id}.jpg")
            if args.overwrite_images or (not os.path.exists(img_path)):
                decode_image_bytes_to_jpg(b, img_path)
                n_written_images += 1

            rec = {
                "id": qid,
                "image_id": image_id,
                "question": question,
                "gt_text": gt,
                "image_path": img_path,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Done.")
    print(f"JSONL: {out_jsonl_path}")
    print(f"Images dir: {images_dir}")
    print(f"Images written this run: {n_written_images}")
    print(f"Missing images for samples: {n_missing} (最好为 0)")

if __name__ == "__main__":
    main()