import json

def load_gqa_openqa(jsonl_path, limit=None):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append({
                "id": obj["id"],
                "question": obj["question"],
                "gt_text": obj["gt_text"],
                "image_path": obj["image_path"],
                "has_image": True
            })
            if limit and len(samples) >= limit:
                break
    return samples