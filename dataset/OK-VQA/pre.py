#!/usr/bin/env python3
# prep_okvqa_from_parquet.py
import os
import json
import argparse
import hashlib
from typing import Any, Dict, List, Optional, Set
import glob

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def to_list_str(x: Any) -> List[str]:
    """
    answers in your parquet is numpy.ndarray; convert robustly to list[str]
    """
    if x is None:
        return []
    # numpy array
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            x = x.tolist()
    except Exception:
        pass

    if isinstance(x, list):
        out = []
        for v in x:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                out.append(s)
        return out
    # scalar
    s = str(x).strip()
    return [s] if s else []

def try_get_image_bytes(image_obj: Any) -> Optional[bytes]:
    """
    Your 'image' column is a dict. We try common layouts:
    - {'bytes': b'...'}
    - {'data': b'...'}
    - {'image_bytes': b'...'}
    - {'path': '...'}  (fallback: read file)
    - {'array': ...}   (rare; would need PIL; skip)
    """
    if image_obj is None:
        return None

    # sometimes already bytes
    if isinstance(image_obj, (bytes, bytearray)):
        return bytes(image_obj)

    if isinstance(image_obj, dict):
        for k in ["bytes", "data", "image_bytes", "raw", "buffer"]:
            v = image_obj.get(k, None)
            if isinstance(v, (bytes, bytearray)):
                return bytes(v)

        # some HF datasets store as {"path": "...", "bytes": ...} or only path
        p = image_obj.get("path", None)
        if isinstance(p, str) and os.path.exists(p):
            with open(p, "rb") as f:
                return f.read()

    return None

def read_parquet_rows(path: str) -> List[Dict[str, Any]]:
    import pandas as pd
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")

def make_image_name(question_id: str) -> str:
    # stable filename; question_id is string per your schema
    return f"{question_id}.jpg"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", type=str, required=True, help="Directory containing OK-VQA parquet shards")
    ap.add_argument("--pattern", type=str, default="val2014-*.parquet", help="Glob pattern for shards")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dir (will create images_cache/)")
    ap.add_argument("--split", type=str, default="val", help="Output split label")
    ap.add_argument("--overwrite_images", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0, help="Debug limit; 0=all")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    images_cache = os.path.join(args.out_dir, "images_cache")
    ensure_dir(images_cache)

    shard_paths = sorted(glob.glob(os.path.join(args.parquet_dir, args.pattern)))
    if not shard_paths:
        raise RuntimeError(f"No parquet shards found: {os.path.join(args.parquet_dir, args.pattern)}")

    out_jsonl = os.path.join(args.out_dir, f"okvqa_{args.split}_openqa.jsonl")

    seen: Set[str] = set()
    written = 0
    total = 0
    images_written = 0
    images_missing = 0

    with open(out_jsonl, "w", encoding="utf-8") as wf:
        for shard in shard_paths:
            rows = read_parquet_rows(shard)
            if args.max_samples and args.max_samples > 0:
                rows = rows[:args.max_samples]

            for obj in rows:
                total += 1
                qid = str(obj.get("question_id", "")).strip()
                if not qid:
                    # fallback stable id
                    qid = sha1(str(obj.get("question", "")) + str(total))[:16]

                if qid in seen:
                    continue
                seen.add(qid)

                question = str(obj.get("question", "")).strip()
                if not question:
                    continue

                answers = to_list_str(obj.get("answers", None))
                # keep list for max-over-GT evaluation
                gt_text: Any = answers if len(answers) > 0 else ""

                # write image
                img_obj = obj.get("image", None)
                img_bytes = try_get_image_bytes(img_obj)
                img_name = make_image_name(qid)
                img_path = os.path.join(images_cache, img_name)

                if img_bytes is not None:
                    if args.overwrite_images or (not os.path.exists(img_path)):
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                    images_written += 1
                else:
                    # still write sample, but mark missing image path for debugging
                    images_missing += 1

                rec = {
                    "id": qid,
                    "question": question,
                    "gt_text": gt_text,
                    "image_path": img_path,
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"[OK] shards: {len(shard_paths)}")
    print(f"[OK] total_rows_read: {total}")
    print(f"[OK] written_after_dedup: {written}")
    print(f"[OK] images_written: {images_written}")
    print(f"[OK] images_missing_bytes: {images_missing}")
    print(f"[OK] JSONL: {out_jsonl}")
    print(f"[OK] images_cache: {images_cache}")

    if images_missing > 0:
        print("[WARN] Some rows had no decodable image bytes from the 'image' dict.")
        print("       If images_missing is large, paste one 'image' field example and I’ll adapt the decoder.")

if __name__ == "__main__":
    main()