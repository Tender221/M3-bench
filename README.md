# Multimodal Robustness Evaluation Toolkit

This repository provides a Linux-first evaluation toolkit for multimodal QA models under paired image and text perturbations. It supports:

- Perturbation data generation on a severity grid
- Model inference on clean / image-only / text-only / mixed inputs
- Standard OpenQA metrics such as EM and F1
- Bootstrap-based interaction analysis
- Heatmaps and one-page summary figures
- User-defined model and dataset adapters

Windows is not an official target platform for this project. Some utility scripts may still work on Windows, but the recommended environment for reproducible use is Linux.

## License

This project is released under the Apache License 2.0. See `LICENSE` for details.

## Repository Layout

```text
root/
├─ code/
│  ├─ run_matrix.py
│  ├─ batch_run_matrix.py
│  ├─ config_examples/
│  ├─ infer/
│  │  ├─ infer_openqa_qwen25.py
│  │  ├─ infer_openqa_qwen3.py
│  │  ├─ infer_openqa_llava.py
│  │  ├─ infer_openqa_blip2.py
│  │  ├─ data/
│  │  └─ eval/
│  └─ route/
└─ dataset/
   ├─ GQA/
   └─ OK-VQA/
```

## Supported Workflow

The default pipeline is:

1. Prepare a base OpenQA-style JSONL dataset.
2. Generate perturbed samples on an image-text severity grid.
3. Run model inference on each grid cell.
4. Evaluate predictions.
5. Compute bootstrap interaction statistics.
6. Export summary CSV, text reports, heatmaps, and a summary figure.

## Data Format

The toolkit expects OpenQA-style JSONL input. Each line should contain at least:

```json
{
  "id": "sample_001",
  "question": "What is shown in the image?",
  "gt_text": ["cat"],
  "image_path": "/absolute/path/to/image.jpg"
}
```

Notes:

- `gt_text` can be either a string or a list of strings.
- `image_path` should be a valid local path.
- Additional fields are allowed and will be stored in `meta`.

This open-source package includes preprocessing scripts, but does not bundle processed benchmark JSONL files by default. See `dataset/README.md`.

## Environment

Recommended:

- Linux
- Python 3.10+
- CUDA-capable PyTorch environment if you run large VLMs

Typical Python packages include:

- `torch`
- `transformers`
- `tqdm`
- `Pillow`
- `numpy`
- `matplotlib`

Install the dependencies required by your chosen model family. For example, Qwen-VL or LLaVA may require additional packages beyond the common stack above.

## Quick Start

### 1. Run a single model on one perturbation pair

```bash
python code/run_matrix.py \
  --exp_root ./output/okvqa/Qwen2.5-VL-7B/jpeg_ocr_grid \
  --shared_data_root ./output/okvqa/shared/jpeg_ocr_grid \
  --shared_cache_root ./output/okvqa/shared/_global_cache \
  --dataset okvqa \
  --base_jsonl ./dataset/OK-VQA/okvqa_val_openqa.jsonl \
  --img_perturb jpeg \
  --txt_perturb ocr \
  --severities 0,1,2,3,4,5 \
  --model_name Qwen2.5-VL-7B \
  --infer_script ./code/infer/infer_openqa_qwen25.py \
  --model_path /path/to/Qwen2.5-VL-7B \
  --run_bootstrap \
  --run_excess
```

### 2. Run a batch of models and datasets

```bash
python code/batch_run_matrix.py \
  --only_models qwen25,qwen3 \
  --only_datasets okvqa,gqa \
  --only_pairs jpeg_ocr,gaussian_blur_typo
```

### 3. Dry run before the real job

```bash
python code/batch_run_matrix.py --dry_run
```

### 4. Minimal smoke test

Use a tiny subset before launching a full grid job:

```bash
python code/run_matrix.py \
  --exp_root ./output/smoke/Qwen2.5-VL-7B/jpeg_ocr_grid \
  --shared_data_root ./output/smoke/shared/jpeg_ocr_grid \
  --shared_cache_root ./output/smoke/shared/_global_cache \
  --dataset okvqa \
  --base_jsonl ./dataset/OK-VQA/okvqa_val_openqa.jsonl \
  --img_perturb jpeg \
  --txt_perturb ocr \
  --severities 0,1 \
  --limit 8 \
  --model_name Qwen2.5-VL-7B \
  --infer_script ./code/infer/infer_openqa_qwen25.py \
  --model_path /path/to/Qwen2.5-VL-7B
```

## Main Entry Points

### `code/run_matrix.py`

Single experiment orchestrator. It:

- prepares or reuses perturbation data
- runs inference
- runs evaluation
- exports summary files
- optionally runs bootstrap / excess / heatmap / summary figure

Useful flags:

- `--prepare_data_only`
- `--reuse_data_only`
- `--shared_data_root`: pair-specific shared input directory for one perturbation grid.
- `--shared_cache_root`: optional global cache root shared across different perturbation pairs.
- `--prefix <run_name>`: optional experiment name prefix used in prediction filenames and bootstrap-ready links. If omitted, the default is `<img_perturb>_<txt_perturb>`.
- `--run_bootstrap`
- `--run_excess`
- `--run_new_metrics`
- `--run_heatmap`
- `--run_summary_figure`

How the grid is constructed:

- `--severities 0,1,2,3` expands automatically into a full image-text grid.
- `cell(i,j)` means image severity `i` and text severity `j`.
- `(0,0)` is `clean`.
- `(i,0)` with `i>0` is `img_only`.
- `(0,j)` with `j>0` is `txt_only`.
- `(i,j)` with `i>0` and `j>0` is `mixed`.

How result names are formed:

- Shared perturbed JSONL names are generated automatically from the perturbation pair and severities, for example `mixed_jpeg_is2__ocr_ts3.jsonl`.
- Prediction files are generated automatically as `<prefix>__<mode>.jsonl`, for example `smoke_s0123_100__mixed.jsonl`.
- Result folders encode the cell position, for example `cells/2_3/` means image severity `2` and text severity `3`.
- `manifest.json` records the run-level settings such as dataset, perturbation pair, severities, scripts, and prefix.
- `reports/metrics_summary.csv` records one row per evaluated cell with its `cell`, `mode`, prediction path, and metric path.

Recommended storage model:

- `shared_data_root` stores the pair-specific per-cell JSONL entrypoints consumed by inference.
- `shared_cache_root` stores reusable perturbation caches. Reuse the same path across runs if you want cross-pair cache reuse.
- `exp_root` stores model-specific predictions, metrics, bootstrap outputs, and figures.

What each path controls in practice:

- `shared_data_root` decides where the current perturbation pair writes its per-cell JSONL entry files such as `clean`, `img_only`, `txt_only`, and `mixed`.
- `shared_cache_root` decides where reusable low-level artifacts are cached, including generated perturbed images, generated perturbed questions, and canonical intermediate JSONLs.
- `exp_root` decides where one model writes its own predictions and reports. Changing models should usually change `exp_root`, but does not require changing `shared_data_root`.
- `--prefix` only affects run naming inside result files. It does not decide cache reuse by itself.

Shared data and cache layout:

- `clean/data/clean_s0_<tag>.jsonl`: clean JSONL entrypoint.
- `cells/<i>_<j>/data/img_only_<tag>.jsonl`: image-only JSONL entrypoint.
- `cells/<i>_<j>/data/txt_only_<tag>.jsonl`: text-only JSONL entrypoint.
- `cells/<i>_<j>/data/mixed_<tag>.jsonl`: mixed JSONL entrypoint.
- `<shared_cache_root>/images/<img_perturb>/is<severity>/seed<seed>/...`: reusable image perturbation cache.
- `<shared_cache_root>/text/<txt_perturb>/ts<severity>/seed<seed>_n<samples>/questions.jsonl`: reusable text perturbation cache.
- `<shared_cache_root>/jsonl/...`: canonical reusable clean / img_only / txt_only JSONL caches.

Example layout:

```text
output/
  okvqa/
    shared/
      jpeg_ocr_grid/
        manifest.json
        clean/
          data/
            clean_s0_okvqa_n50.jsonl
        cells/
          1_0/
            data/
              img_only_jpeg_is1__ocr_ts0.jsonl
          0_2/
            data/
              txt_only_jpeg_is0__ocr_ts2.jsonl
          2_3/
            data/
              mixed_jpeg_is2__ocr_ts3.jsonl
      _global_cache/
        images/
          jpeg/
            is1/
        text/
          ocr/
            ts2/
              seed0_n50/
                questions.jsonl
        jsonl/
          okvqa/
            ...
    Qwen2.5-VL-7B/
      jpeg_ocr_grid/
        manifest.json
        clean/
          pred/
            jpeg_ocr__clean.jsonl
          metrics/
            vqa_score.json
        cells/
          2_3/
            pred/
              jpeg_ocr__mixed.jsonl
            metrics/
              vqa_score.json
        reports/
          metrics_summary.csv
          bootstrap_ready/
          excess_f1.txt
          heatmap_mixed_f1.png
```

Result directory layout:

- `clean/pred/` and `cells/<i>_<j>/pred/`: prediction JSONL files produced by the selected model.
- `clean/metrics/` and `cells/<i>_<j>/metrics/`: per-cell metric outputs.
- `reports/metrics_summary.csv`: one row per evaluated cell; this is the easiest place to confirm which cell and mode were actually run.
- `reports/bootstrap_ready/`: prediction and metric files linked or copied into a bootstrap-friendly flat structure.
- `reports/excess_f1.txt`: excess analysis summary when `--run_excess` is enabled.
- `reports/heatmap_*.png` and summary figures: generated only when `--run_heatmap` or `--run_summary_figure` is enabled.

Reuse rules:

- Same `shared_data_root`, different model: reuse the same generated inputs with `--reuse_data_only`.
- Same `shared_cache_root`, different text perturbation but same image perturbation: image cache is reused.
- Same `shared_cache_root`, different image perturbation but same text perturbation: text cache is reused.
- If `--shared_cache_root` is omitted, the cache defaults to `<shared_data_root>/_cache`, so reuse stays local to that pair-specific directory.

Important note:

- Cross-pair reuse is not inferred from matching perturbation names alone. It only happens when the runs point to the same `--shared_cache_root`.
- Cross-model reuse of already prepared cell JSONLs depends on reusing the same `--shared_data_root` and then running with `--reuse_data_only`.
- If you want a clean mental model, think of `shared_data_root` as pair-level reusable inputs and `shared_cache_root` as global reusable building blocks.

Typical examples:

Reuse existing shared inputs for a different model:

```bash
python code/run_matrix.py \
  --exp_root ./output/okvqa/BLIP2-opt-2.7B/jpeg_ocr_grid \
  --shared_data_root ./output/okvqa/shared/jpeg_ocr_grid \
  --shared_cache_root ./output/okvqa/shared/_global_cache \
  --dataset okvqa \
  --img_perturb jpeg \
  --txt_perturb ocr \
  --severities 0,1,2,3 \
  --model_name BLIP2-opt-2.7B \
  --infer_script ./code/infer/infer_openqa_blip2.py \
  --model_path /path/to/BLIP2-OPT-2.7B \
  --reuse_data_only \
  --run_bootstrap \
  --run_excess
```

Generate a different pair while reusing the same global cache:

```bash
python code/run_matrix.py \
  --exp_root ./output/okvqa/Qwen2.5-VL-7B/jpeg_typo_grid \
  --shared_data_root ./output/okvqa/shared/jpeg_typo_grid \
  --shared_cache_root ./output/okvqa/shared/_global_cache \
  --dataset okvqa \
  --base_jsonl ./dataset/OK-VQA/okvqa_val_openqa.jsonl \
  --img_perturb jpeg \
  --txt_perturb typo \
  --severities 0,1,2,3 \
  --model_name Qwen2.5-VL-7B \
  --infer_script ./code/infer/infer_openqa_qwen25.py \
  --model_path /path/to/Qwen2.5-VL-7B
```

### `code/batch_run_matrix.py`

Higher-level batch scheduler. It:

- prepares shared perturbation data once per dataset-pair
- reuses shared data for multiple models
- supports custom model and dataset registries

Recommended batch-mode cache convention:

- Use one global cache root per dataset, for example `./output/okvqa/shared/_global_cache`.
- Pass the same `--shared_cache_root` to all batch jobs if you want cache reuse across different perturbation pairs.

Example batch command with global cache reuse:

```bash
python code/batch_run_matrix.py \
  --only_models qwen25,blip2 \
  --only_datasets okvqa \
  --only_pairs jpeg_ocr,jpeg_typo,gaussian_blur_ocr \
  --shared_cache_root ./output/okvqa/shared/_global_cache
```

With the command above:

- `jpeg_ocr` and `jpeg_typo` can reuse the same cached JPEG perturbations.
- `jpeg_ocr` and `gaussian_blur_ocr` can reuse the same cached OCR text perturbations.
- each pair still keeps its own `shared_data_root`, so the per-cell JSONL entry files remain easy to inspect independently.

## Custom Dataset Integration

There are two ways to add your own dataset.

### Option A: Direct JSONL

If your dataset is already in the OpenQA JSONL format, pass:

- `--dataset your_dataset_name`
- `--jsonl_path /path/to/data.jsonl`

### Option B: Dataset config

Use a dataset config JSON. Example:

`code/config_examples/custom_dataset_loader.example.json`

Then run inference with:

```bash
python code/infer/infer_openqa_qwen25.py \
  --dataset my_dataset \
  --dataset_config ./code/config_examples/custom_dataset_loader.example.json \
  --model_path /path/to/model \
  --output_path ./tmp/preds.jsonl
```

For batch runs, you can also register datasets via:

- `code/config_examples/custom_datasets.example.json`

and pass:

```bash
python code/batch_run_matrix.py --datasets_config ./my_datasets.json
```

## Custom Model Integration

To add your own model adapter, prepare an inference script that accepts at least:

- `--dataset`
- `--jsonl_path`
- `--dataset_config`
- `--output_path`
- `--model_path`

Then register it with a model registry JSON, for example:

- `code/config_examples/custom_models.example.json`

Run:

```bash
python code/batch_run_matrix.py --models_config ./my_models.json
```

## Outputs

After a run, the main outputs are typically:

- `reports/metrics_summary.csv`
- `reports/bootstrap_<metric>.txt`
- `reports/excess_<metric>.txt`
- `reports/new_metrics/summary_<metric>.txt`
- `reports/new_metrics/summary_<metric>.json`
- `reports/new_metrics/per_cell_<metric>.csv`
- `reports/figures/*heatmap.png`
- `reports/figures/summary_<metric>.png`

The toolkit now prints these output locations at the end of each run.

## Visualization

The repository includes:

- heatmap generation from bootstrap statistics
- a one-page summary figure combining metric grid and interaction summaries

Example:

```bash
python code/infer/eval/summary_figure.py \
  --summary_csv ./output/exp/reports/metrics_summary.csv \
  --bootstrap_txt ./output/exp/reports/bootstrap_f1.txt \
  --summary_json ./output/exp/reports/new_metrics/summary_f1.json \
  --out_png ./output/exp/reports/figures/summary_f1.png
```

## Current Built-in Model Scripts

- Qwen2.5-VL
- Qwen3-VL
- LLaVA-1.5
- BLIP2

These scripts are examples of the adapter interface, not a claim that every model family is fully tested in every environment.

## Current Built-in Dataset Hooks

- OK-VQA
- GQA
- RealWorldQA loader path remains available in the shared loader code

## Limitations

- Linux is the only officially recommended platform.
- Different VLM families may need different dependency versions.
- Some older auxiliary scripts in the repository are not part of the polished main workflow.
- You should validate model paths, CUDA setup, and dataset paths in your own environment.

## Suggested Release Checklist

Before publishing, it is recommended to:

1. Verify the main workflow on a clean Linux environment.
2. Remove or archive old helper scripts that are not part of the public pipeline.
3. Confirm the Apache-2.0 license choice matches your release preference.
4. Add environment setup instructions for each supported model family.
5. Add a small demo dataset or a toy example for smoke testing.

See also:

- `RELEASE_CHECKLIST.md`
- `LICENSE_OPTIONS.md`

## Citation

If you use this toolkit in academic work, please cite the corresponding paper or project page once released.
