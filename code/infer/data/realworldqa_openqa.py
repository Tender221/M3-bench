#!/usr/bin/env python3
import os
import uuid
from typing import List, Dict, Optional

import pandas as pd
from PIL import Image
from io import BytesIO


def _find_parquets(data_root: str) -> List[str]:
    paths = []
    for d in [data_root, os.path.join(data_root, "data")]:
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith(".parquet"):
                    paths.append(os.path.join(d, fn))
    paths = sorted(list(set(paths)))
    if not paths:
        raise FileNotFoundError(f"No parquet found under: {data_root} or {data_root}/data")
    return paths


def _to_pil_image(x) -> Optional[Image.Image]:
    """
    兼容 parquet 常见存图方式：
    - bytes / bytearray
    - dict 里带 bytes（{"bytes": ...} / {"data": ...}）
    - 已经是 PIL.Image
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        try:
            return Image.open(BytesIO(x)).convert("RGB")
        except Exception:
            return None
    if isinstance(x, dict):
        for k in ["bytes", "data", "image", "value"]:
            if k in x and isinstance(x[k], (bytes, bytearray)):
                try:
                    return Image.open(BytesIO(x[k])).convert("RGB")
                except Exception:
                    return None
    return None


def load_realworldqa_openqa(
    data_root: Optional[str] = None,
    parquet_paths: Optional[List[str]] = None,
    image_save_dir: str = "./realworldqa_images",
    limit: Optional[int] = None,
) -> List[Dict]:
    if parquet_paths is None:
        if not data_root:
            raise ValueError("Either data_root or parquet_paths must be provided.")
        parquet_paths = _find_parquets(data_root)

    os.makedirs(image_save_dir, exist_ok=True)

    samples: List[Dict] = []
    for p in parquet_paths:
        df = pd.read_parquet(p)

        # 尽量容错列名
        q_col = "question" if "question" in df.columns else ("prompt" if "prompt" in df.columns else None)
        a_col = "answer" if "answer" in df.columns else ("answers" if "answers" in df.columns else ("label" if "label" in df.columns else None))
        i_col = "image" if "image" in df.columns else None

        if q_col is None or a_col is None:
            raise KeyError(f"Missing question/answer columns in {p}. columns={list(df.columns)}")

        for _, row in df.iterrows():
            sid = f"realworldqa_{uuid.uuid4().hex[:10]}"
            question = str(row[q_col]).strip() if pd.notna(row[q_col]) else ""
            gt_text = str(row[a_col]).strip() if pd.notna(row[a_col]) else ""

            pil = _to_pil_image(row[i_col]) if (i_col and i_col in df.columns) else None
            image_path = ""
            if pil is not None:
                image_path = os.path.join(image_save_dir, f"{sid}.jpg")
                pil.save(image_path, "JPEG", quality=95)

            samples.append({
                "id": sid,
                "question": question,
                "gt_text": gt_text,
                "image_path": image_path,
                "has_image": bool(pil is not None),
            })

            if limit is not None and len(samples) >= limit:
                return samples

    return samples