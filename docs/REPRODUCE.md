# Reproduce the binary analysis and joint reliability profile

## One CPU entry

From the repository root, using Python 3.12 and a fresh output directory:

```bash
python -m pip install -r code/binary/requirements.txt
python code/binary/reproduce_all.py --output output/reproduced --plots
```

For analysis without plotting, install `code/binary/requirements-core.txt` and omit `--plots`. The numerical code uses NumPy and standard Python only. No model, Judge, PyTorch, GPU or network request is made by the analysis. Dependencies are pinned to versions actually used; installing packages is a separate preparation step.

The entry reads `results/binary_v1/` by default, verifies the physical NPZ/root hashes and complete scoring invariants, and derives:

- `binary_v1/`: all 12 original source CSVs from labels, without reading expected CSV values during derivation. Subsequent comparison verifies them against released source tables, with exact categories/counts and floating tolerance 1e-9.
- `joint_metrics_v1/`: all 18 new profile CSVs, shared bootstrap draws, masks/support metadata, source/tail sensitivities, all 3,600 candidate comparisons and selected examples. Defaults are 2,000 image-cluster replicates and seed 17.
- `figures/`, with `--plots`: new profile figures from generated CSVs. This does not redraw or overwrite the existing 15 publication figure sets.
- Execution, source/input pins, environment, timing and failure receipts. Existing output is refused; no previous run is overwritten.

Input lookup is based on code location rather than the caller's working directory. `--data` can specify another copy of the same pinned release. The frozen protocol is shipped beside the code and passed explicitly. Protocol/path errors are checked before consuming the unified output directory.

The numerical environment is Python 3.12.12/NumPy 2.4.4; plotting uses matplotlib 3.10.9. Bootstrap uses PCG64/seed 17, a shared 2000×1065 draw matrix, and fixed qualification/support masks. Intervals are linear-quantile pointwise percentile summaries with valid draw IDs retained. Each original image carries all its questions, conditions and models; condition counts are not independent sample counts. Tail reselection can produce a percentile interval excluding its observed point; no retrospective interval repair is applied. See [definitions and interpretation](JOINT_METRICS.md).

## What was verified

The mask-to-table producer ran in a fresh directory: 12 CSVs/16,188 rows reproduced, maximum floating difference 2.22e-16. An independent implementation directly parsed the public masks and checked all 24 profile summaries, 9,600 cell rows, 3,600 candidate comparisons, 72,000 bootstrap values, 36 intervals and 54 paired contrasts: 799,092 scalar comparisons passed, maximum difference 1.11e-16. Synthetic tests distinguish equal-marginal conditional accuracy from identical-cell-map suite survival and cover empty support, exact anchors and fixed bootstrap masks.

The unified entry has a separate fresh-output acceptance check and public tests. Receipts record actual code/input hashes and commands; timing and absolute installation paths are not scientific values required to be byte-identical. Replaying summaries does not independently validate Judge decisions.

## Retained alternatives and older figures

PowerShell 7.2+ can still reproduce core Task/Semantic counts through .NET without Python:

```powershell
pwsh -NoProfile -File code/binary/analyze_binary_masks.ps1 -Output ./output/core-counts
pwsh -NoProfile -File code/binary/test_semantic_sensitivity.ps1
```

This packaged entry was checked with PowerShell 7.6.5/.NET 10.0.11 on Windows: 48 source checks, 208 summary comparisons and 24 synthetic assertions passed. A new Linux execution of that wrapper is not claimed.

The original 15 binary figure sets have a retained, separate plotting entry. Install pandas 3.0.2 as well as plotting requirements if regenerating them:

```bash
python -m pip install pandas==3.0.2
python code/binary/plot_binary_metrics.py --data results/binary_v1 --output output/original-figures
```

Each figure output directory must be fresh. Arial is preferred with DejaVu Sans fallback. Fonts can alter render bytes without changing source values.

## Reproducibility tiers

| Tier | Included contract/material | Executed by the CPU entry? |
|---|---|---|
| Labels → tables, profiles, intervals and new figures | Complete condition labels, image grouping, source IDs, code, protocol, dependencies | Yes; fresh-output replay and independent numeric checks |
| Raw answers → binary labels | Frozen safe-exact router, semantic/format prompts, parser and Judge identity in [SCORING](SCORING.md) | No; complete raw response archives and Judge weights are not bundled |
| Images/questions → model responses | Public revisions and retained generation contract in `code/binary/inference_contract.json`; legacy adapters | No; original assets/dataset access and exact input reconstruction are separate requirements |
| Training/adaptation regeneration | Published development summaries and interpretation | No; training data/checkpoints and private execution infrastructure are outside this release |

Original models used native wrappers, greedy 32-token generation and seed 0. BLIP-2's revision is a public reproduction pin; its original local config hash was not retained. The three-model historical environment snapshot is supporting evidence, not a complete time-stamped pip export. Fourth-model provenance is separately identified. Metadata availability does not imply a new end-to-end clean-room model rerun.

## Legacy toolkit

`code/run_matrix.py`, `code/batch_run_matrix.py`, adapters and preprocessing retain their historical paths. Exact former English/Chinese instructions are archived in `docs/legacy/`. They describe an older EM/F1 workflow, including severity defaults outside the current four-level grid. Those commands cannot be used unmodified as the current binary protocol. Historical scores, intervals and learned-policy claims do not replace released binary labels and fixed-population analysis.
