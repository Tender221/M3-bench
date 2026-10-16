# Compact data release

## Included

- `results/joint_metrics_v1/`: 18 existing-label profile CSVs, shared image-bootstrap draws, cohort/conditional support, all candidate comparisons, new figure and a physical file manifest. These do not contain new model responses or change historical labels. Analysis metadata, code/runtime pins and definitions are separate from original model-generation provenance.
- `code/binary/reproduce_all.py` and its replay/metric helpers: one CPU entry that derives all 12 original source tables plus the profiles; tests and pinned analysis/plotting requirements accompany it. `inference_contract.json` supplies public model revisions and retained generation metadata, with historical provenance limits stated explicitly.
- `results/binary_v1/binary_masks.npz`: all 4×1500×441 original completed binary outcomes, including separate semantic, format and task bits.
- `results/binary_v1/roots.csv`:1500 root IDs, source dataset/question/image IDs, question type and original image-group fingerprint. It contains no original image paths, image pixels, questions or reference-answer text.
- Twelve source CSVs for the current binary summaries, exact eligible pair/cell gaps, source composition, slices, budget discovery and directed model replacements.
- Current binary figure PNG/PDF exports, preserved byte-for-byte from their source bundle.
- The legacy `results/semantic_v1` sensitivity tables and `results/expanded_dev_v1` semantic adaptation summaries are descriptive and do not add semantic confidence intervals. The new `joint_metrics_v1` release separately includes common4 semantic-view profile intervals under its fixed image bootstrap.

## NPZ schema

Members: `flags.npy`, `models.npy`, `conditions.npy`, `root_ids.npy`. Flags are unsigned bytes in C order with shape `(4,1500,441)`:

```text
flags = 8 + 1*semantic_correct + 2*format_compliant + 4*task_pass
task_pass == (semantic_correct AND format_compliant)
```

The8 bit records completion. It is distinct from answer correctness. Models, conditions and root IDs must be read from the archive, not inferred from filesystem order. Conditions are clean,20 image-only,20 text-only and 400 joints. The image-family order is blur,noise,JPEG,brightness,contrast; text-family order is typo,keyboard,OCR-like,whitespace/punctuation,casing; nonzero severities are1–4.

`manifest.json` pins the compact archive and source tables by SHA-256/size. These pins establish release byte identity, not an independent judgment of scientific truth.

## Dataset rights and excluded assets

Original datasets: [GQA](https://cs.stanford.edu/people/dorarad/gqa/), [OK-VQA](https://okvqa.allenai.org/), [VQAv2](https://visualqa.org/). Obtain images/annotations through their original distributors and follow the applicable terms. The repository's existing code license does not override original dataset/model licenses.

No original images, transformed image assets, raw prediction text archives, model weights/checkpoints, sealed-test data, SSH credentials, private server logs or unpublished manuscript source/PDF are included in this compact update. Published source IDs support provenance but do not imply that excluded assets have been externally released.

The original benchmark is an observed finite library. The separate200-image adaptation evaluation uses known-benchmark development images and is not an untouched final test. Conditions are correlated within an original image; do not treat millions of conditions as millions of independent pictures.


