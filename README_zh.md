# 多模态鲁棒性评测工具箱

这是一个面向多模态问答模型的鲁棒性评测工具箱，主打图像扰动与文本扰动的组合评测。当前仓库支持：

- 在扰动强度网格上生成评测数据
- 对 clean / 仅图像扰动 / 仅文本扰动 / 混合扰动输入做推理
- 计算 OpenQA 常用指标，如 EM、F1
- 基于 bootstrap 的交互效应分析
- 生成热力图与单页汇总图
- 支持用户自定义模型接入
- 支持用户自定义数据集接入

本项目以 Linux 为主支持平台。Windows 不再作为官方重点适配目标，个别脚本可能仍可运行，但如果你希望获得稳定、可复现的使用体验，建议直接在 Linux 环境下使用。

## 许可证

本项目采用 Apache License 2.0 开源，详见根目录下的 `LICENSE` 文件。

## 仓库结构

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

## 整体流程

默认评测流程如下：

1. 准备基础 OpenQA 格式 JSONL 数据。
2. 在图像扰动和文本扰动的 severity 网格上生成扰动样本。
3. 在每个网格单元上运行模型推理。
4. 计算评测指标。
5. 进行 bootstrap 交互效应分析。
6. 输出汇总 CSV、文本报告、热力图和总览图。

## 数据格式

输入数据默认采用 OpenQA 风格的 JSONL，每一行至少包含：

```json
{
  "id": "sample_001",
  "question": "图中是什么？",
  "gt_text": ["cat"],
  "image_path": "/absolute/path/to/image.jpg"
}
```

说明：

- `gt_text` 可以是字符串，也可以是字符串列表。
- `image_path` 需要是本地有效路径。
- 其余附加字段会被保存在 `meta` 中。

这个开源整理目录默认只包含数据预处理脚本，不默认附带处理好的 benchmark JSONL 文件。可参考 `dataset/README.md` 自行准备对应数据。

## 环境建议

推荐环境：

- Linux
- Python 3.10+
- 如果运行大模型，建议使用带 CUDA 的 PyTorch 环境

常见依赖包括：

- `torch`
- `transformers`
- `tqdm`
- `Pillow`
- `numpy`
- `matplotlib`

不同模型家族还可能需要额外依赖，比如 Qwen-VL、LLaVA 等请根据各自模型要求安装。

## 快速开始

### 1. 运行单个模型 + 单个扰动对

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

### 2. 批量运行多个模型和数据集

```bash
python code/batch_run_matrix.py \
  --only_models qwen25,qwen3 \
  --only_datasets okvqa,gqa \
  --only_pairs jpeg_ocr,gaussian_blur_typo
```

### 3. 先 dry run 检查命令

```bash
python code/batch_run_matrix.py --dry_run
```

### 4. 最小 smoke test

建议在正式大规模运行前，先用一个很小的子集做冒烟测试：

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

## 主要入口脚本

### `code/run_matrix.py`

这是单个实验的主调度脚本，负责：

- 生成或复用扰动数据
- 调用模型推理
- 调用评测脚本
- 输出汇总结果
- 可选执行 bootstrap / excess / 热力图 / 总览图

常用参数：

- `--prepare_data_only`
- `--reuse_data_only`
- `--shared_data_root`：某一个扰动对对应的共享输入目录，里面放按 cell 划分的 JSONL 入口文件。
- `--shared_cache_root`：可选的全局缓存根目录。不同扰动对如果共用这个路径，就能共享图像/文本扰动缓存。
- `--prefix <run_name>`：可选的实验名前缀，会用于预测文件名和 bootstrap_ready 里的文件链接；如果不传，默认使用 `<img_perturb>_<txt_perturb>`。
- `--run_bootstrap`
- `--run_excess`
- `--run_new_metrics`
- `--run_heatmap`
- `--run_summary_figure`

网格组合规则：

- `--severities 0,1,2,3` 会自动展开成完整的图像-文本扰动网格。
- `cell(i,j)` 表示图像扰动强度为 `i`，文本扰动强度为 `j`。
- `(0,0)` 对应 `clean`。
- `(i,0)` 且 `i>0` 对应 `img_only`。
- `(0,j)` 且 `j>0` 对应 `txt_only`。
- `(i,j)` 且 `i>0` 且 `j>0` 对应 `mixed`。

结果命名规则：

- 共享扰动数据 JSONL 会根据扰动对和强度自动命名，例如 `mixed_jpeg_is2__ocr_ts3.jsonl`。
- 预测结果文件会自动命名为 `<prefix>__<mode>.jsonl`，例如 `smoke_s0123_100__mixed.jsonl`。
- 结果目录里的 `cells/2_3/` 表示图像强度 `2`、文本强度 `3`。
- `manifest.json` 会记录这次运行的全局配置，例如数据集、扰动对、强度列表、脚本路径和 prefix。
- `reports/metrics_summary.csv` 会为每个已评测的 cell 自动写一行，包含 `cell`、`mode`、预测路径和指标路径。

推荐的目录理解方式：

- `shared_data_root`：某一个扰动对的共享输入目录，主要给推理脚本消费。
- `shared_cache_root`：全局缓存目录，主要给跨不同扰动对的图像/文本扰动复用。
- `exp_root`：某一个模型对应的实验结果目录，里面放预测、指标、bootstrap、热力图和总览图。

这三个路径在实际使用里的分工可以这样理解：

- `shared_data_root` 决定当前这个扰动对把按 cell 划分的 JSONL 入口文件写到哪里，例如 `clean`、`img_only`、`txt_only`、`mixed`。
- `shared_cache_root` 决定底层可复用缓存写到哪里，包括已经生成过的扰动图像、已经生成过的扰动文本，以及标准化后的中间 JSONL。
- `exp_root` 决定某一个模型自己的预测结果和评测结果写到哪里。通常换模型时需要换 `exp_root`，但不一定需要换 `shared_data_root`。
- `--prefix` 只影响结果文件里的命名，本身并不决定缓存是否会复用。

共享数据与缓存目录结构：

- `clean/data/clean_s0_<tag>.jsonl`：clean 条件的 JSONL 入口文件。
- `cells/<i>_<j>/data/img_only_<tag>.jsonl`：当前 cell 的 image-only JSONL。
- `cells/<i>_<j>/data/txt_only_<tag>.jsonl`：当前 cell 的 text-only JSONL。
- `cells/<i>_<j>/data/mixed_<tag>.jsonl`：当前 cell 的 mixed JSONL。
- `<shared_cache_root>/images/<img_perturb>/is<severity>/seed<seed>/...`：可复用的图像扰动缓存。
- `<shared_cache_root>/text/<txt_perturb>/ts<severity>/seed<seed>_n<samples>/questions.jsonl`：可复用的文本扰动缓存。
- `<shared_cache_root>/jsonl/...`：可复用的 clean / img_only / txt_only 标准 JSONL 缓存。

一个典型目录结构大致如下：

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

结果目录结构：

- `clean/pred/` 和 `cells/<i>_<j>/pred/`：当前模型输出的预测 JSONL。
- `clean/metrics/` 和 `cells/<i>_<j>/metrics/`：每个 cell 对应的评测结果。
- `reports/metrics_summary.csv`：每个已评测 cell 一行，是最适合快速确认“到底跑了哪些组合”的汇总文件。
- `reports/bootstrap_ready/`：为了后续 bootstrap 准备的平铺文件目录。
- `reports/excess_f1.txt`：开启 `--run_excess` 时生成的 excess 分析结果。
- `reports/heatmap_*.png` 和 summary figure：只有在开启 `--run_heatmap` 或 `--run_summary_figure` 时才会生成。

复用规则：

- 同一个 `shared_data_root`，换模型：可以用 `--reuse_data_only` 直接复用已经生成好的输入数据。
- 同一个 `shared_cache_root`，如果图像扰动相同、只换文本扰动：图像缓存会复用。
- 同一个 `shared_cache_root`，如果文本扰动相同、只换图像扰动：文本缓存会复用。
- 如果不传 `--shared_cache_root`，缓存会默认落到 `<shared_data_root>/_cache`，此时复用只会局限在这个 pair 对应的目录内部。

几个容易误解但很重要的点：

- 不同扰动对之间不会因为名字“看起来一样”就自动共享缓存。只有它们显式指向同一个 `--shared_cache_root` 时，跨 pair 复用才会发生。
- 不同模型之间复用已经准备好的 cell JSONL，依赖的是同一个 `--shared_data_root`，以及运行时使用 `--reuse_data_only`。
- 如果想最好理解整个机制，可以把 `shared_data_root` 当成“pair 级别复用输入”，把 `shared_cache_root` 当成“全局底层构件缓存”。

典型示例：

复用已有共享输入，只换模型：

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

换一个新的扰动对，但仍然复用同一个全局缓存：

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

这是更高层的批处理脚本，负责：

- 每个 dataset + pair 只生成一次共享扰动数据
- 多个模型复用同一份共享数据
- 支持自定义模型注册表和数据集注册表

批量运行时的推荐缓存约定：

- 建议每个数据集配一个统一的全局缓存根，例如 `./output/okvqa/shared/_global_cache`。
- 如果你希望不同扰动对之间也复用缓存，请在所有批量任务里传同一个 `--shared_cache_root`。

例如：

```bash
python code/batch_run_matrix.py \
  --only_models qwen25,blip2 \
  --only_datasets okvqa \
  --only_pairs jpeg_ocr,jpeg_typo,gaussian_blur_ocr \
  --shared_cache_root ./output/okvqa/shared/_global_cache
```

上面这个命令下：

- `jpeg_ocr` 和 `jpeg_typo` 会复用同一份 JPEG 图像扰动缓存。
- `jpeg_ocr` 和 `gaussian_blur_ocr` 会复用同一份 OCR 文本扰动缓存。
- 但每个 pair 仍然保留自己独立的 `shared_data_root`，所以每组 cell JSONL 入口文件仍然是分开、可检查的。

## 自定义数据集接入

你可以通过两种方式接入自己的数据集。

### 方式 A：直接使用 JSONL

如果你的数据已经是 OpenQA JSONL 格式，那么直接传：

- `--dataset your_dataset_name`
- `--jsonl_path /path/to/data.jsonl`

### 方式 B：使用 dataset config

如果你的字段名不同，或者你想通过配置文件定义加载方式，可以参考：

- `code/config_examples/custom_dataset_loader.example.json`

调用示例：

```bash
python code/infer/infer_openqa_qwen25.py \
  --dataset my_dataset \
  --dataset_config ./code/config_examples/custom_dataset_loader.example.json \
  --model_path /path/to/model \
  --output_path ./tmp/preds.jsonl
```

如果你想在批处理脚本中注册自定义数据集，可以参考：

- `code/config_examples/custom_datasets.example.json`

然后运行：

```bash
python code/batch_run_matrix.py --datasets_config ./my_datasets.json
```

## 自定义模型接入

如果你要接自己的模型，需要准备一个推理脚本，至少支持以下参数：

- `--dataset`
- `--jsonl_path`
- `--dataset_config`
- `--output_path`
- `--model_path`

然后通过模型注册表接入，参考：

- `code/config_examples/custom_models.example.json`

运行示例：

```bash
python code/batch_run_matrix.py --models_config ./my_models.json
```

## 输出结果

运行完成后，常见输出包括：

- `reports/metrics_summary.csv`
- `reports/bootstrap_<metric>.txt`
- `reports/excess_<metric>.txt`
- `reports/new_metrics/summary_<metric>.txt`
- `reports/new_metrics/summary_<metric>.json`
- `reports/new_metrics/per_cell_<metric>.csv`
- `reports/figures/*heatmap.png`
- `reports/figures/summary_<metric>.png`

现在脚本在运行结束时也会明确打印这些结果文件路径。

## 可视化

仓库目前包含两类结果可视化：

- 从 bootstrap 统计结果生成热力图
- 生成一张总览图，把指标网格与交互分析结果放在同一张图里

示例：

```bash
python code/infer/eval/summary_figure.py \
  --summary_csv ./output/exp/reports/metrics_summary.csv \
  --bootstrap_txt ./output/exp/reports/bootstrap_f1.txt \
  --summary_json ./output/exp/reports/new_metrics/summary_f1.json \
  --out_png ./output/exp/reports/figures/summary_f1.png
```

## 当前内置模型脚本

- Qwen2.5-VL
- Qwen3-VL
- LLaVA-1.5
- BLIP2

这些脚本的定位更偏“适配示例”和“默认入口”，并不代表所有模型家族、所有依赖版本都已经在所有环境下充分验证。

## 当前内置数据集入口

- OK-VQA
- GQA
- `load_dataset.py` 中仍保留了 RealWorldQA 的公共加载入口

## 当前限制

- Linux 是唯一官方推荐平台。
- 不同 VLM 模型对依赖版本的要求可能不同。
- 仓库中仍有少量较早期的辅助脚本，不属于当前整理后的主流程。
- 发布前仍建议你在自己的 Linux 环境里完整跑通一次主流程。

## 建议的开源前检查项

建议在正式发布前做以下事情：

1. 在干净的 Linux 环境中完整验证主流程。
2. 清理或归档那些不准备作为正式接口公开的旧辅助脚本。
3. 确认 Apache-2.0 是你最终想采用的许可证。
4. 为不同模型家族补一份环境安装说明。
5. 准备一个小型 demo 数据或 smoke test 样例。


## 引用

如果你在论文或项目中使用了这个工具箱，请在对应论文或项目页面发布后补充引用信息。
