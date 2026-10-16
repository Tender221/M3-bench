# Reproduce the current binary analysis

## CPU entry

PowerShell 7.2+ (with its .NET runtime) is sufficient. The release wrapper is platform-neutral and uses no GPU or network:

```powershell
pwsh -NoProfile -File code/binary/analyze_binary_masks.ps1 -Output ./output/recheck
pwsh -NoProfile -File code/binary/test_semantic_sensitivity.ps1
```

The wrapper defaults to `results/binary_v1/binary_masks.npz` and `roots.csv`, verifies their hashes against the included manifest, and requires a fresh output directory. It rederives the paper's core qualification, event/affected-question, reduced-slice and reference-preservation counts for both Task and Semantic. It also retains semantic outcomes on the unchanged primary cohorts. `root_records.csv` permits inspection at the question/image-group level.

The compact masks contain all original per-condition outcomes, not sampled rows or regenerated predictions. Exact condition/model/root identifiers are stored inside the NPZ. The C# core is byte-identical to the implementation checked against the completed original task results and independently reimplemented for the semantic sensitivity. Synthetic checks include a format-only clean exclusion, an off-diagonal semantic failure, a format-only joint failure, corresponding anchors without a clean prerequisite, and fixed-reference success loss. No new Judge validation is implied.

The packaged entry was actually checked on Windows with PowerShell 7.6.5 and .NET 10.0.11: all 48 built-in source checks, 208 task/semantic summary-field comparisons and 24 synthetic assertions passed. The wrapper uses portable .NET APIs; a new Linux execution of this packaged entry is not claimed. The separate original Judge runtime was Linux, as recorded in the scoring contract.

## Figure source

The 12 original descriptive source CSVs are in `results/binary_v1/`. Figures supplied here are the current binary publication figures, not renamed lexical-score plots. Optional figure regeneration uses the archived, previously executed plotting script:

```bash
python -m pip install numpy pandas matplotlib
python code/binary/plot_binary_metrics.py --data results/binary_v1 --output output/binary-figures
```

The output directory must not exist. Arial is preferred with DejaVu Sans fallback; different font installations can change layout/render bytes. All source fractions and denominators must remain unchanged. The public plotting script is the same source used for the released figures; the packaging step does not rerun Python or claim a new platform test.

## What is and is not reproduced

- The public CPU entry rederives the core counts from all compact condition labels. Source CSVs additionally expose exact pair/cell and fixed-grid budget analyses used in the figures.
- Scoring protocol, model revision and safe-exact routing are in `docs/SCORING.md` and `code/binary/`. Existing-label reanalysis does not rerun grading or regenerate the 2,646,000 original answers.
- Raw image perturbation assets and complete raw model response archives are outside this compact release. Full inference requires obtaining original datasets/models under their terms and respecting the exact frozen input contract. This release does not claim a new end-to-end clean-room rerun.
- The 63,000-row adaptation development experiment is a separate known-benchmark scope. Its source summary and semantic sensitivity are published separately; its training data, checkpoints, raw predictions and private execution infrastructure are not included.

## Legacy toolkit

`code/run_matrix.py`, `code/batch_run_matrix.py`, model adapters, old evaluation helpers and dataset preprocessing remain at their original paths. The exact former English/Chinese instructions are archived in `docs/legacy/README_before_binary_20261016.md` and `README_zh_before_binary_20261016.md`.

These instructions document an older configurable inference/EM/F1 workflow, including examples with severity levels beyond the current paper's four nonzero levels. They are retained for users of that toolkit. **Do not use an EM/F1 output, an old default severity grid, an old interval or an old learned-policy result as the current binary M³-Test result.** The current paper's condition order, input realizations, binary protocol, populations and model revisions are frozen separately.
