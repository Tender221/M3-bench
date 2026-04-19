#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, random, re, io, math
from typing import Dict, Any, Iterable, List, Tuple, Optional
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

# ============================================================
# Utils
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def dump_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def maybe_link_or_copy(src: str, dst: str):
    ensure_dir(os.path.dirname(dst))
    if os.path.lexists(dst):
        os.unlink(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except Exception:
        import shutil
        shutil.copy2(src, dst)

def stable_int_seed(*parts) -> int:
    """Stable seed via md5 (process-independent)."""
    import hashlib
    s = "||".join(map(str, parts)).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)

def pick_sign_from_seed(seed: int) -> int:
    """Deterministic +/-1 from seed."""
    return -1 if (seed % 2 == 0) else 1

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)

# ============================================================
# Quality metrics for calibration
#   - default: PSNR (no deps)
#   - optional: SSIM if scikit-image is available
# ============================================================
def to_uint8_rgb(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.uint8)

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return float(np.mean((a - b) ** 2))

def psnr(a: np.ndarray, b: np.ndarray, maxv: float = 255.0) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return 1e9
    return 10.0 * math.log10((maxv * maxv) / m)

def try_import_ssim():
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        return ssim_fn
    except Exception:
        return None

_SSIM_FN = try_import_ssim()

def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    """
    Uses skimage if available. Returns mean SSIM over channels.
    """
    if _SSIM_FN is None:
        raise RuntimeError("SSIM requested but scikit-image not installed.")
    a_f = a.astype(np.float32) / 255.0
    b_f = b.astype(np.float32) / 255.0
    # channel_axis supported in newer skimage; fallback to per-channel if needed
    try:
        val = _SSIM_FN(a_f, b_f, channel_axis=2, data_range=1.0)
        return float(val)
    except TypeError:
        vals = []
        for c in range(3):
            vals.append(_SSIM_FN(a_f[..., c], b_f[..., c], data_range=1.0))
        return float(np.mean(vals))

# ============================================================
# Image perturbations (natural)
# severity in {0,1,2,3,4,5}
# ============================================================
# Default parameter maps (will be overridden if --use_calibration)
DEFAULT_BLUR_RADIUS = {0:0.0, 1:1.0, 2:2.0, 3:4.0, 4:6.0, 5:8.0}
DEFAULT_NOISE_SIGMA = {0:0.0, 1:5.0, 2:10.0, 3:20.0, 4:30.0, 5:45.0}  # pixel-space 0-255
DEFAULT_JPEG_QUALITY = {0:95, 1:70, 2:40, 3:20, 4:10, 5:5}
DEFAULT_BRIGHT_DELTA = {0:0.0, 1:0.15, 2:0.30, 3:0.45, 4:0.60, 5:0.75}
DEFAULT_CONTRAST_DELTA = {0:0.0, 1:0.15, 2:0.30, 3:0.45, 4:0.60, 5:0.80}

def img_gaussian_blur(img: Image.Image, severity: int, blur_radius_map: Dict[int, float]) -> Image.Image:
    r = float(blur_radius_map[int(severity)])
    return img if r <= 0 else img.filter(ImageFilter.GaussianBlur(radius=r))

def img_gaussian_noise(img: Image.Image, severity: int, seed: int, noise_sigma_map: Dict[int, float]) -> Image.Image:
    sigma = float(noise_sigma_map[int(severity)])
    if sigma <= 0:
        return img
    rng = np.random.default_rng(seed)
    arr = np.array(img).astype(np.float32)
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def img_jpeg(img: Image.Image, severity: int, jpeg_quality_map: Dict[int, int]) -> Image.Image:
    q = int(jpeg_quality_map[int(severity)])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()

def img_brightness(img: Image.Image, severity: int, seed: int, bright_delta_map: Dict[int, float]) -> Image.Image:
    if severity == 0:
        return img
    d = float(bright_delta_map[int(severity)])
    sign = pick_sign_from_seed(seed)
    factor = 1.0 + sign * d
    factor = clamp(factor, 0.05, 3.0)
    return ImageEnhance.Brightness(img).enhance(factor)

def img_contrast(img: Image.Image, severity: int, seed: int, contrast_delta_map: Dict[int, float]) -> Image.Image:
    if severity == 0:
        return img
    d = float(contrast_delta_map[int(severity)])
    sign = pick_sign_from_seed(seed + 1337)  # decorrelate
    factor = 1.0 + sign * d
    factor = clamp(factor, 0.05, 3.5)
    return ImageEnhance.Contrast(img).enhance(factor)

# ============================================================
# Text perturbations (natural, low semantic change)
# severity in {0,1,2,3,4,5}
# ============================================================
P_MAP = {0:0.0, 1:0.02, 2:0.05, 3:0.10, 4:0.15, 5:0.22}

def txt_typo(s: str, severity: int, seed: int) -> str:
    p = P_MAP[int(severity)]
    if p == 0 or not s:
        return s
    rng = random.Random(seed)
    chars = list(s)
    i = 0
    while i < len(chars):
        if chars[i].isalnum() and rng.random() < p:
            op = rng.choice(["del", "swap", "rep", "ins"])
            if op == "del":
                chars.pop(i)
                continue
            if op == "swap" and i + 1 < len(chars):
                chars[i], chars[i+1] = chars[i+1], chars[i]
                i += 2
                continue
            if op == "rep":
                chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz0123456789")
            if op == "ins":
                chars.insert(i, rng.choice("abcdefghijklmnopqrstuvwxyz0123456789"))
                i += 1
        i += 1
    return "".join(chars)

def txt_ocr_confusion(s: str, severity: int, seed: int) -> str:
    p = P_MAP[int(severity)]
    if p == 0 or not s:
        return s
    rng = random.Random(seed)
    mapping = {
        "0":"O", "O":"0",
        "1":"l", "l":"1",
        "5":"S", "S":"5",
        "2":"Z", "Z":"2"
    }
    out = []
    for ch in s:
        if ch in mapping and rng.random() < p:
            out.append(mapping[ch])
        else:
            out.append(ch)
    return "".join(out)

def txt_keyboard_neighbor(s: str, severity: int, seed: int) -> str:
    p = P_MAP[int(severity)]
    if p == 0 or not s:
        return s
    rng = random.Random(seed)
    neigh = {
        "q":"wa", "w":"qes", "e":"wrd", "r":"etf", "t":"ryg", "y":"tuh", "u":"yij", "i":"uok", "o":"ipl", "p":"o",
        "a":"qws", "s":"awedx", "d":"serfc", "f":"drtgcv", "g":"ftyhbv", "h":"gyujbn", "j":"huikmn", "k":"jiol", "l":"kop",
        "z":"asx", "x":"zsdc", "c":"xdfv", "v":"cfgb", "b":"vghn", "n":"bhjm", "m":"njk"
    }
    out = []
    for ch in s:
        low = ch.lower()
        if low in neigh and rng.random() < p:
            rep = rng.choice(neigh[low])
            out.append(rep.upper() if ch.isupper() else rep)
        else:
            out.append(ch)
    return "".join(out)

def txt_whitespace_punc(s: str, severity: int, seed: int) -> str:
    if severity == 0 or not s:
        return s
    rng = random.Random(seed)

    space_jitter = {1:0.15, 2:0.25, 3:0.35, 4:0.45, 5:0.60}[int(severity)]
    comma_ins    = {1:0.02, 2:0.05, 3:0.08, 4:0.12, 5:0.18}[int(severity)]

    # (1) space jitter
    s = re.sub(r"\s+", lambda m: "  " if rng.random() < space_jitter else " ", s).strip()

    # (2) drop punctuation from sev2; stronger at sev4/5
    if severity >= 2:
        drop_all = (severity >= 4)
        if drop_all:
            s = re.sub(r"[^\w\s]", " ", s)
        else:
            # mild: drop most punctuation but keep '?'
            s = re.sub(r"[!\"#$%&'()*+,\-./:;<=>@\[\]^_`{|}~]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

    # (3) insert light commas from sev3
    if severity >= 3:
        tokens = s.split()
        out = []
        for t in tokens:
            out.append(t)
            if rng.random() < comma_ins:
                out.append(",")
        s = " ".join(out).replace(" ,", ",")
    return s

def txt_casing(s: str, severity: int, seed: int) -> str:
    if severity == 0 or not s:
        return s
    rng = random.Random(seed)
    if severity == 1:
        return s.lower()
    if severity == 2:
        return s.upper()
    if severity == 3:
        p = 0.50
    elif severity == 4:
        p = 0.70
    else:
        p = 0.85

    out = []
    for ch in s:
        if ch.isalpha() and rng.random() < p:
            out.append(ch.upper())
        else:
            out.append(ch.lower() if ch.isalpha() else ch)
    return "".join(out)

TXT_PERTURB = {
    "typo": txt_typo,
    "ocr": txt_ocr_confusion,
    "keyboard": txt_keyboard_neighbor,
    "whitespace_punc": txt_whitespace_punc,
    "casing": txt_casing,
}

# ============================================================
# Save image helper (avoid "jpeg dilution" + allow PNG)
# ============================================================
def save_image(
    im: Image.Image,
    out_path: str,
    img_perturb: str,
    severity: int,
    save_format: str,
    jpeg_quality_map: Dict[int, int]
) -> str:
    """
    Key rule:
      - if perturb == jpeg: save JPEG at the *same* quality used by the perturb severity
      - otherwise: save as PNG by default (if save_format=png), or JPEG(95) if jpg
    """
    fmt = save_format.lower()
    if fmt not in ("jpg", "jpeg", "png"):
        raise ValueError("--save_format must be jpg or png")

    if fmt == "png":
        if not out_path.lower().endswith(".png"):
            out_path = os.path.splitext(out_path)[0] + ".png"
        im.save(out_path, format="PNG", optimize=True)
        return out_path

    # jpg/jpeg
    if not (out_path.lower().endswith(".jpg") or out_path.lower().endswith(".jpeg")):
        out_path = os.path.splitext(out_path)[0] + ".jpg"

    if img_perturb == "jpeg":
        q = int(jpeg_quality_map[int(severity)])
        im.save(out_path, format="JPEG", quality=q, optimize=True)
    else:
        im.save(out_path, format="JPEG", quality=95, optimize=True)
    return out_path


def canonical_clean_jsonl_path(cache_root: str, seed: int, max_samples: int) -> str:
    return os.path.join(cache_root, "jsonl", "clean", f"seed{seed}_n{max_samples}", "clean.jsonl")


def canonical_img_only_jsonl_path(cache_root: str, img_perturb: str, img_severity: int, seed: int, max_samples: int) -> str:
    return os.path.join(
        cache_root,
        "jsonl",
        "img_only",
        f"{img_perturb}_is{img_severity}_seed{seed}_n{max_samples}",
        "img_only.jsonl",
    )


def canonical_txt_only_jsonl_path(cache_root: str, txt_perturb: str, txt_severity: int, seed: int, max_samples: int) -> str:
    return os.path.join(
        cache_root,
        "jsonl",
        "txt_only",
        f"{txt_perturb}_ts{txt_severity}_seed{seed}_n{max_samples}",
        "txt_only.jsonl",
    )


def shared_image_dir(cache_root: str, img_perturb: str, img_severity: int, seed: int) -> str:
    return os.path.join(cache_root, "images", img_perturb, f"is{img_severity}", f"seed{seed}")


def shared_text_cache_path(cache_root: str, txt_perturb: str, txt_severity: int, seed: int, max_samples: int) -> str:
    return os.path.join(
        cache_root,
        "text",
        txt_perturb,
        f"ts{txt_severity}",
        f"seed{seed}_n{max_samples}",
        "questions.jsonl",
    )


def load_jsonl_rows(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    for obj in load_jsonl(path):
        rows.append(obj)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def load_text_cache(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    cache = {}
    for obj in load_jsonl(path):
        cache[str(obj["id"])] = obj["question"]
    return cache


def write_text_cache(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    dump_jsonl(path, rows)

# ============================================================
# Auto calibration (image perturb only)
# ============================================================
def default_targets(metric: str) -> Dict[int, float]:
    """
    Targets for severity levels 1-5.
    - For PSNR: higher is closer to clean. Here we set a descending ladder.
    - For SSIM: closer to 1 is closer to clean. We set descending ladder.
    """
    if metric == "ssim":
        # mild -> strong
        return {1:0.92, 2:0.86, 3:0.78, 4:0.68, 5:0.58}
    # psnr
    return {1:35.0, 2:30.0, 3:25.0, 4:20.0, 5:15.0}

def evaluate_metric(clean_im: Image.Image, pert_im: Image.Image, metric: str) -> float:
    a = to_uint8_rgb(clean_im)
    b = to_uint8_rgb(pert_im)
    if metric == "ssim":
        return ssim_rgb(a, b)
    return psnr(a, b)

def pick_best_param(
    clean_images: List[Image.Image],
    make_perturb_fn,
    candidates: List[float],
    metric: str,
    target: float,
    seed: int,
    bidirectional: bool = False
) -> float:
    """
    Finds candidate param that makes mean metric closest to target.
    For brightness/contrast, we can evaluate both + and - directions (bidirectional=True)
    and average the metric to avoid sign bias.
    """
    best_p = candidates[0]
    best_err = float("inf")

    for p in candidates:
        vals = []
        for idx, cim in enumerate(clean_images):
            # deterministic per image for reproducibility
            s = stable_int_seed(seed, idx, p)
            if bidirectional:
                # evaluate both signs by flipping seed parity
                im_pos = make_perturb_fn(cim, p, s | 1)      # odd => +1
                im_neg = make_perturb_fn(cim, p, s & ~1)     # even => -1
                v = 0.5 * (evaluate_metric(cim, im_pos, metric) + evaluate_metric(cim, im_neg, metric))
            else:
                pim = make_perturb_fn(cim, p, s)
                v = evaluate_metric(cim, pim, metric)
            vals.append(v)

        mean_v = float(np.mean(vals))
        err = abs(mean_v - target)
        if err < best_err:
            best_err = err
            best_p = p

    return float(best_p)

def auto_calibrate_image_maps(
    img_perturb: str,
    clean_images: List[Image.Image],
    metric: str,
    seed: int,
    targets: Optional[Dict[int, float]] = None
) -> Dict[str, Dict[int, float]]:
    """
    Returns a dict containing the calibrated map for the selected perturbation.
    Output format:
      {
        "metric": "...",
        "targets": {...},
        "maps": {
           "blur_radius": {0:...,1:...,..},
           ...
        }
      }
    """
    if metric == "ssim" and _SSIM_FN is None:
        print("[WARN] scikit-image not found; falling back to PSNR calibration.")
        metric = "psnr"

    if targets is None:
        targets = default_targets(metric)

    payload = {
        "img_perturb": img_perturb,
        "metric": metric,
        "targets": {str(k): float(v) for k, v in targets.items()},
        "maps": {}
    }

    if img_perturb == "gaussian_blur":
        # candidate radius grid
        candidates = [0.5, 0.8, 1.0, 1.3, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
        out = {0:0.0}
        def make_fn(cim, radius, s):
            return cim.filter(ImageFilter.GaussianBlur(radius=float(radius)))
        for sev in [1,2,3,4,5]:
            out[sev] = pick_best_param(clean_images, make_fn, candidates, metric, targets[sev], seed + 11*sev)
        payload["maps"]["blur_radius"] = {str(k): float(v) for k, v in out.items()}

    elif img_perturb == "gaussian_noise":
        candidates = [2, 3, 5, 7, 10, 14, 20, 28, 35, 45, 60]
        out = {0:0.0}
        def make_fn(cim, sigma, s):
            rng = np.random.default_rng(int(s))
            arr = np.array(cim).astype(np.float32)
            noise = rng.normal(0, float(sigma), arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        for sev in [1,2,3,4,5]:
            out[sev] = pick_best_param(clean_images, make_fn, candidates, metric, targets[sev], seed + 17*sev)
        payload["maps"]["noise_sigma"] = {str(k): float(v) for k, v in out.items()}

    elif img_perturb == "jpeg":
        candidates = [95, 85, 75, 65, 55, 45, 35, 28, 22, 18, 14, 10, 7, 5, 3]
        out = {0:95.0}
        def make_fn(cim, q, s):
            q = int(q)
            buf = io.BytesIO()
            cim.save(buf, format="JPEG", quality=q, optimize=True)
            buf.seek(0)
            return Image.open(buf).convert("RGB").copy()
        for sev in [1,2,3,4,5]:
            out[sev] = pick_best_param(clean_images, make_fn, candidates, metric, targets[sev], seed + 23*sev)
        payload["maps"]["jpeg_quality"] = {str(k): float(v) for k, v in out.items()}

    elif img_perturb == "brightness":
        candidates = [0.05, 0.10, 0.15, 0.22, 0.30, 0.38, 0.45, 0.55, 0.65, 0.75, 0.90]
        out = {0:0.0}
        def make_fn(cim, delta, s):
            sign = pick_sign_from_seed(int(s))
            factor = 1.0 + sign * float(delta)
            factor = clamp(factor, 0.05, 3.0)
            return ImageEnhance.Brightness(cim).enhance(factor)
        for sev in [1,2,3,4,5]:
            out[sev] = pick_best_param(clean_images, make_fn, candidates, metric, targets[sev], seed + 29*sev, bidirectional=True)
        payload["maps"]["bright_delta"] = {str(k): float(v) for k, v in out.items()}

    elif img_perturb == "contrast":
        candidates = [0.05, 0.10, 0.15, 0.22, 0.30, 0.38, 0.45, 0.55, 0.65, 0.80, 1.00]
        out = {0:0.0}
        def make_fn(cim, delta, s):
            sign = pick_sign_from_seed(int(s) + 1337)
            factor = 1.0 + sign * float(delta)
            factor = clamp(factor, 0.05, 3.5)
            return ImageEnhance.Contrast(cim).enhance(factor)
        for sev in [1,2,3,4,5]:
            out[sev] = pick_best_param(clean_images, make_fn, candidates, metric, targets[sev], seed + 31*sev, bidirectional=True)
        payload["maps"]["contrast_delta"] = {str(k): float(v) for k, v in out.items()}

    else:
        raise ValueError(f"Unsupported img_perturb for calibration: {img_perturb}")

    return payload

def load_or_calibrate_maps(
    args,
    clean_images: List[Image.Image]
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, int], Dict[int, float], Dict[int, float]]:
    """
    Returns 5 maps:
      blur_radius_map, noise_sigma_map, jpeg_quality_map, bright_delta_map, contrast_delta_map
    Each map includes keys 0..5.
    """
    blur_radius_map = dict(DEFAULT_BLUR_RADIUS)
    noise_sigma_map = dict(DEFAULT_NOISE_SIGMA)
    jpeg_quality_map = dict(DEFAULT_JPEG_QUALITY)
    bright_delta_map = dict(DEFAULT_BRIGHT_DELTA)
    contrast_delta_map = dict(DEFAULT_CONTRAST_DELTA)

    if not args.use_calibration and not args.auto_calibrate:
        return blur_radius_map, noise_sigma_map, jpeg_quality_map, bright_delta_map, contrast_delta_map

    # calibration file path depends on selected img_perturb
    calib_root = args.calib_cache_dir if args.calib_cache_dir else args.out_dir
    ensure_dir(calib_root)

    calib_path = os.path.join(
        calib_root,
        f"calib_{args.img_perturb}_{args.calib_metric}.json"
    )

    if args.use_calibration and os.path.exists(calib_path):
        with open(calib_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        if not args.auto_calibrate:
            # requested to use calibration but file not present
            raise RuntimeError(f"--use_calibration set but file not found: {calib_path}. "
                               f"Run once with --auto_calibrate to create it.")
        payload = auto_calibrate_image_maps(
            img_perturb=args.img_perturb,
            clean_images=clean_images,
            metric=args.calib_metric,
            seed=args.seed,
            targets=None
        )
        with open(calib_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[CALIB] Saved calibration to: {calib_path}")

    maps = payload.get("maps", {})

    def parse_map(m: Dict[str, Any], cast=float) -> Dict[int, Any]:
        out = {}
        for k, v in m.items():
            out[int(k)] = cast(v)
        # ensure 0..5 exist
        for sev in range(0, 6):
            if sev not in out:
                # fill missing with default if possible
                pass
        return out

    if "blur_radius" in maps:
        m = parse_map(maps["blur_radius"], float)
        for sev in range(0, 6):
            if sev in m:
                blur_radius_map[sev] = float(m[sev])

    if "noise_sigma" in maps:
        m = parse_map(maps["noise_sigma"], float)
        for sev in range(0, 6):
            if sev in m:
                noise_sigma_map[sev] = float(m[sev])

    if "jpeg_quality" in maps:
        m = parse_map(maps["jpeg_quality"], float)
        for sev in range(0, 6):
            if sev in m:
                # stored as float but should be int
                jpeg_quality_map[sev] = int(round(float(m[sev])))

    if "bright_delta" in maps:
        m = parse_map(maps["bright_delta"], float)
        for sev in range(0, 6):
            if sev in m:
                bright_delta_map[sev] = float(m[sev])

    if "contrast_delta" in maps:
        m = parse_map(maps["contrast_delta"], float)
        for sev in range(0, 6):
            if sev in m:
                contrast_delta_map[sev] = float(m[sev])

    print("[CALIB] Loaded maps:")
    if args.img_perturb == "gaussian_blur":
        print("  blur_radius_map:", blur_radius_map)
    elif args.img_perturb == "gaussian_noise":
        print("  noise_sigma_map:", noise_sigma_map)
    elif args.img_perturb == "jpeg":
        print("  jpeg_quality_map:", jpeg_quality_map)
    elif args.img_perturb == "brightness":
        print("  bright_delta_map:", bright_delta_map)
    elif args.img_perturb == "contrast":
        print("  contrast_delta_map:", contrast_delta_map)

    return blur_radius_map, noise_sigma_map, jpeg_quality_map, bright_delta_map, contrast_delta_map

# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shared_cache_root", default=None,
                    help="Optional global cache root for decoupled image/text perturbation reuse.")
    ap.add_argument("--target_mode", default="all",
                    choices=["all", "clean", "img_only", "txt_only", "mixed"],
                    help="Which JSONL to materialize into out_dir. Caches are still reused automatically.")

    ap.add_argument("--img_perturb", default="gaussian_blur",
                    choices=["gaussian_blur","gaussian_noise","jpeg","brightness","contrast"])
    ap.add_argument("--txt_perturb", default="typo", choices=sorted(TXT_PERTURB.keys()))

    ap.add_argument("--img_severity", type=int, default=2, choices=[0,1,2,3,4,5])
    ap.add_argument("--txt_severity", type=int, default=2, choices=[0,1,2,3,4,5])

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=1000)

    # image save
    ap.add_argument("--save_format", default="jpg", choices=["jpg", "png"],
                    help="How to save perturbed images. Recommend png to avoid extra JPEG artifacts. Default: jpg")
    ap.add_argument("--skip_existing", action="store_true",
                    help="If set, skip generating an image if it already exists on disk.")

    # calibration
    ap.add_argument("--auto_calibrate", action="store_true",
                    help="If set, calibrate the selected --img_perturb so severity 1-5 matches target quality ladder.")
    ap.add_argument("--use_calibration", action="store_true",
                    help="If set, load and apply calibration_{img_perturb}.json from out_dir.")
    ap.add_argument("--calib_samples", type=int, default=64,
                    help="How many images to use for calibration (default 64).")
    ap.add_argument("--calib_metric", default="psnr", choices=["psnr","ssim"],
                    help="Calibration metric. If 'ssim' but skimage unavailable, falls back to psnr.")
    ap.add_argument(
    "--calib_cache_dir",
    default=None,
    help="Directory to store/reuse calibration files. If None, use out_dir."
)

    args = ap.parse_args()
    ensure_dir(args.out_dir)
    cache_root = args.shared_cache_root or os.path.join(args.out_dir, "_cache")
    ensure_dir(cache_root)

    # --------------------------------------------------------
    # Load a small set of clean images for calibration if needed
    # --------------------------------------------------------
    clean_images: List[Image.Image] = []
    if args.auto_calibrate or args.use_calibration:
        # sample first calib_samples images deterministically from input
        for obj in load_jsonl(args.input_jsonl):
            if len(clean_images) >= args.calib_samples:
                break
            img_path = obj["image_path"]
            try:
                clean_images.append(Image.open(img_path).convert("RGB"))
            except Exception:
                continue
        if len(clean_images) < max(8, min(args.calib_samples, 16)):
            print(f"[WARN] Only loaded {len(clean_images)} images for calibration; results may be noisy.")

    # Load or run calibration (for selected img_perturb) and obtain maps
    blur_radius_map, noise_sigma_map, jpeg_quality_map, bright_delta_map, contrast_delta_map = \
        load_or_calibrate_maps(args, clean_images)

    # output image dir (tag includes severities)
    input_rows = load_jsonl_rows(args.input_jsonl, limit=args.max_samples)
    n = len(input_rows)

    img_out_dir = shared_image_dir(cache_root, args.img_perturb, args.img_severity, args.seed)
    ensure_dir(img_out_dir)

    clean_rows, img_rows, txt_rows, mixed_rows = [], [], [], []

    text_cache_path = shared_text_cache_path(cache_root, args.txt_perturb, args.txt_severity, args.seed, args.max_samples)
    text_cache = load_text_cache(text_cache_path) if args.txt_severity != 0 else {}
    fresh_text_cache_rows: List[Dict[str, Any]] = []

    for obj in input_rows:
        qid = str(obj["id"])
        q = obj["question"]
        img_path = obj["image_path"]

        # Decouple the cache keys so identical image or text perturbations are reusable
        # across different perturbation pairs.
        seed_img = stable_int_seed(qid, args.seed, args.img_perturb, args.img_severity)
        seed_txt = stable_int_seed(qid, args.seed, args.txt_perturb, args.txt_severity)

        # ---- image perturb ----
        img_new_path = img_path
        if args.img_severity != 0:
            ext = ".png" if args.save_format == "png" else ".jpg"
            candidate_path = os.path.join(img_out_dir, f"{qid}{ext}")

            if args.skip_existing and os.path.exists(candidate_path):
                img_new_path = candidate_path
            else:
                try:
                    im = Image.open(img_path).convert("RGB")
                except Exception as e:
                    raise RuntimeError(f"Failed to open image: {img_path} ({e})")

                if args.img_perturb == "gaussian_blur":
                    im_p = img_gaussian_blur(im, args.img_severity, blur_radius_map)
                elif args.img_perturb == "gaussian_noise":
                    im_p = img_gaussian_noise(im, args.img_severity, seed_img, noise_sigma_map)
                elif args.img_perturb == "jpeg":
                    im_p = img_jpeg(im, args.img_severity, jpeg_quality_map)
                elif args.img_perturb == "brightness":
                    im_p = img_brightness(im, args.img_severity, seed_img, bright_delta_map)
                elif args.img_perturb == "contrast":
                    im_p = img_contrast(im, args.img_severity, seed_img, contrast_delta_map)
                else:
                    raise ValueError(f"Unknown img_perturb: {args.img_perturb}")

                img_new_path = save_image(
                    im_p,
                    candidate_path,
                    args.img_perturb,
                    args.img_severity,
                    args.save_format,
                    jpeg_quality_map
                )

        # ---- text perturb ----
        q_txt = q
        if args.txt_severity != 0:
            q_txt = text_cache.get(qid)
            if q_txt is None:
                q_txt = TXT_PERTURB[args.txt_perturb](q, args.txt_severity, seed_txt)
                text_cache[qid] = q_txt
                fresh_text_cache_rows.append({"id": qid, "question": q_txt})

        base = dict(obj)  # keep all fields (gt_text etc.)

        # clean
        clean_obj = dict(base)
        clean_obj.update({
            "perturb_img": {"type": "none", "severity": 0},
            "perturb_txt": {"type": "none", "severity": 0},
            "seed": args.seed
        })

        # img-only
        img_obj = dict(base)
        img_obj["image_path"] = img_new_path
        img_obj["question"] = q
        img_obj.update({
            "perturb_img": {"type": args.img_perturb, "severity": args.img_severity},
            "perturb_txt": {"type": "none", "severity": 0},
            "seed": args.seed
        })

        # txt-only
        txt_obj = dict(base)
        txt_obj["image_path"] = img_path
        txt_obj["question"] = q_txt
        txt_obj.update({
            "perturb_img": {"type": "none", "severity": 0},
            "perturb_txt": {"type": args.txt_perturb, "severity": args.txt_severity},
            "seed": args.seed
        })

        # mixed
        mixed_obj = dict(base)
        mixed_obj["image_path"] = img_new_path
        mixed_obj["question"] = q_txt
        mixed_obj.update({
            "perturb_img": {"type": args.img_perturb, "severity": args.img_severity},
            "perturb_txt": {"type": args.txt_perturb, "severity": args.txt_severity},
            "seed": args.seed
        })

        clean_rows.append(clean_obj)
        img_rows.append(img_obj)
        txt_rows.append(txt_obj)
        mixed_rows.append(mixed_obj)

    if fresh_text_cache_rows:
        merged = [{"id": qid, "question": question} for qid, question in sorted(text_cache.items())]
        write_text_cache(text_cache_path, merged)

    tag = f"{args.img_perturb}_is{args.img_severity}__{args.txt_perturb}_ts{args.txt_severity}"
    clean_dst = os.path.join(args.out_dir, f"clean_s0_{tag}.jsonl")
    img_dst = os.path.join(args.out_dir, f"img_only_{tag}.jsonl")
    txt_dst = os.path.join(args.out_dir, f"txt_only_{tag}.jsonl")
    mixed_dst = os.path.join(args.out_dir, f"mixed_{tag}.jsonl")

    clean_cache_jsonl = canonical_clean_jsonl_path(cache_root, args.seed, args.max_samples)
    if not os.path.exists(clean_cache_jsonl):
        ensure_dir(os.path.dirname(clean_cache_jsonl))
        dump_jsonl(clean_cache_jsonl, clean_rows)

    img_cache_jsonl = canonical_img_only_jsonl_path(
        cache_root, args.img_perturb, args.img_severity, args.seed, args.max_samples
    )
    if not os.path.exists(img_cache_jsonl):
        ensure_dir(os.path.dirname(img_cache_jsonl))
        dump_jsonl(img_cache_jsonl, img_rows)

    txt_cache_jsonl = canonical_txt_only_jsonl_path(
        cache_root, args.txt_perturb, args.txt_severity, args.seed, args.max_samples
    )
    if not os.path.exists(txt_cache_jsonl):
        ensure_dir(os.path.dirname(txt_cache_jsonl))
        dump_jsonl(txt_cache_jsonl, txt_rows)

    if args.target_mode in ("all", "clean"):
        maybe_link_or_copy(clean_cache_jsonl, clean_dst)
    if args.target_mode in ("all", "img_only"):
        maybe_link_or_copy(img_cache_jsonl, img_dst)
    if args.target_mode in ("all", "txt_only"):
        maybe_link_or_copy(txt_cache_jsonl, txt_dst)
    if args.target_mode in ("all", "mixed"):
        dump_jsonl(mixed_dst, mixed_rows)

    print("Done.")
    print("N =", n)
    print("Image cache dir:", img_out_dir)
    if args.txt_severity != 0:
        print("Text cache path:", text_cache_path)
    print("Cache root:", cache_root)
    print("JSONLs saved to:", args.out_dir)

if __name__ == "__main__":
    main()
