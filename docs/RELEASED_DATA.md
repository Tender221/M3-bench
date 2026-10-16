# Compact data release

## Included

- `results/binary_v1/binary_masks.npz`: all4×1500×441 original completed binary outcomes, including separate semantic, format and task bits.
- `results/binary_v1/roots.csv`:1500 root IDs, source dataset/question/image IDs, question type and original image-group fingerprint. It contains no original image paths, image pixels, questions or reference-answer text.
- Twelve source CSVs for the current binary summaries, exact eligible pair/cell gaps, source composition, slices, budget discovery and directed model replacements.
- Current binary figure PNG/PDF exports, preserved byte-for-byte from their source bundle.
- Descriptive semantic sensitivity CSVs and the completed200-image development summary. New semantic confidence intervals are not claimed.

## NPZ schema

Members: `flags.npy`, `models.npy`, `conditions.npy`, `root_ids.npy`. Flags are unsigned bytes in C order with shape `(4,1500,441)`:

```text
flags = 8 + 1*semantic_correct + 2*format_compliant + 4*task_pass
task_pass == (semantic_correct AND format_compliant)
```

The8 bit records completion. It is distinct from answer correctness. Models, conditions and root IDs must be read from the archive, not inferred from filesystem order. Conditions are clean,20 image-only,20 text-only and400 joints. The image-family order is blur,noise,JPEG,brightness,contrast; text-family order is typo,keyboard,OCR-like,whitespace/punctuation,casing; nonzero severities are1–4.

`manifest.json` pins the compact archive and source tables by SHA-256/size. These pins establish release byte identity, not an independent judgment of scientific truth.

## Dataset rights and excluded assets

Original datasets: [GQA](https://cs.stanford.edu/people/dorarad/gqa/), [OK-VQA](https://okvqa.allenai.org/), [VQAv2](https://visualqa.org/). Obtain images/annotations through their original distributors and follow the applicable terms. The repository's existing code license does not override original dataset/model licenses.

No original images, transformed image assets, raw prediction text archives, model weights/checkpoints, sealed-test data, SSH credentials, private server logs or unpublished manuscript source/PDF are included in this compact update. Published source IDs support provenance but do not imply that excluded assets have been externally released.

The original benchmark is an observed finite library. The separate200-image adaptation evaluation uses known-benchmark development images and is not an untouched final test. Conditions are correlated within an original image; do not treat millions of conditions as millions of independent pictures.
