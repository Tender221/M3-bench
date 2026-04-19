#!/usr/bin/env python3
"""
ScienceQA数据集加载脚本（适配真实json格式：数字答案+纯字符串选项）
"""
# ====================== 环境兼容修复 ======================
import sys
from types import ModuleType

try:
    from huggingface_hub import is_offline_mode
except ImportError:
    def is_offline_mode() -> bool:
        import os
        return os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    
    if "huggingface_hub" not in sys.modules:
        sys.modules["huggingface_hub"] = ModuleType("huggingface_hub")
    sys.modules["huggingface_hub"].is_offline_mode = is_offline_mode
# ====================== 兼容修复结束 ======================

import json
import os
import pandas as pd

def load_scienceqa(
    split: str = "test",  # 可选：test/train/val
    image_root: str = "/root/dataset/scienceqa/tutorial/",
    ann_path: str = "/root/dataset/scienceqa/tutorial/text/problems.json"
) -> list:
    """
    加载ScienceQA数据集（适配真实标注格式）
    - choices: 纯字符串列表 ["West Virginia", "Louisiana"...]
    - answer: 数字索引（0/1/2/3）→ 转为字母A/B/C/D + 保留数字索引
    """
    samples = []
    
    # 1. 读取标注文件
    print(f"📖 Reading annotation file: {ann_path}")
    with open(ann_path, 'r', encoding='utf-8') as f:
        problems = json.load(f)
    
    # 字母映射表（数字0→A，1→B，2→C，3→D...）
    idx_to_label = ["A", "B", "C", "D", "E", "F"]
    
    # 2. 遍历所有问题，筛选指定分片
    for pid, problem in problems.items():
        # 跳过非目标分片的样本
        if problem.get("split") != split:
            continue
        
        # 3. 拼接图片路径（适配你的格式）
        img_name = problem.get("image", "")  # 从json中取image字段（如"image.png"）
        img_dir = os.path.join(image_root, split, pid)  # tutorial/test/1/
        img_path = os.path.join(img_dir, img_name) if img_name else ""
        abs_img_path = os.path.abspath(img_path) if os.path.exists(img_path) else ""
        
        # 4. 处理选项（纯字符串列表→添加A/B/C标签）
        raw_choices = problem.get("choices", [])
        choices = []  # 最终格式：["A. West Virginia", "B. Louisiana"...]
        choices_raw = []  # 原始纯字符串列表（备用）
        if isinstance(raw_choices, list) and len(raw_choices) > 0:
            choices_raw = [opt.strip() for opt in raw_choices if opt.strip()]
            # 给每个选项添加A/B/C标签（适配prompt）
            for idx, opt_text in enumerate(choices_raw):
                if idx < len(idx_to_label):  # 防止索引越界
                    choices.append(f"{idx_to_label[idx]}. {opt_text}")
        
        # 5. 处理答案（数字索引→字母+保留原始数字）
        raw_answer = problem.get("answer", -1)  # 原始答案是数字（0/1/2/3）
        gt_label = ""  # 字母答案（A/B/C）
        gt_idx = raw_answer  # 数字答案（0/1/2）
        if 0 <= raw_answer < len(choices_raw):
            gt_label = idx_to_label[raw_answer]
        
        # 6. 构造统一格式样本
        sample = {
            "id": pid,                  # 样本ID
            "image_path": abs_img_path, # 图片绝对路径
            "question": problem.get("question", ""),  # 英文问题
            "choices": choices,         # 带A/B标签的选项（用于prompt）
            "choices_raw": choices_raw, # 原始纯字符串选项（备用）
            "gt": gt_label,             # 字母答案（A/B/C，用于评测）
            "gt_idx": gt_idx            # 数字答案（0/1/2，备用）
        }
        samples.append(sample)
    
    # 可选：读取parquet文件
    parquet_path = f"/root/dataset/scienceqa/data/{split}.parquet"
    if os.path.exists(parquet_path):
        print(f"📊 Reading parquet file: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    
    print(f"✅ Loaded ScienceQA {split} split successfully, total {len(samples)} samples")
    return samples

# 测试用
if __name__ == "__main__":
    test_samples = load_scienceqa(split="train")
    if test_samples:
        print("\n📌 Example sample (first one):")
        print(f"ID: {test_samples[0]['id']}")
        print(f"Image path: {test_samples[0]['image_path']}")
        print(f"Question: {test_samples[0]['question']}")
        print(f"Choices (with label): {test_samples[0]['choices']}")
        print(f"GT label (A/B/C): {test_samples[0]['gt']}")
        print(f"GT index (0/1/2): {test_samples[0]['gt_idx']}")