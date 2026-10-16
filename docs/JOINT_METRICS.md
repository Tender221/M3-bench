# Paired reliability profiles and CPU reproduction

The unified entry recomputes results from authenticated, already-scored public
labels. It runs no model, Judge, GPU job, training, or data download. The original
image/question identities and existing benchmark roles remain fixed. These
known-benchmark profiles do not establish unseen-data generalization.

## Run from any working directory

Use Python 3.12.12 and `code/binary/requirements.txt` for the tested analysis and
plotting environment. Install `code/binary/requirements-core.txt` for NumPy-only
production and omit `--plots`; matplotlib is imported only when plotting.
`environment_manifest.json` records the actual tested versions. This environment
does not describe historical GPU inference.

```bash
python -m pip install -r code/binary/requirements.txt
python code/binary/reproduce_all.py --output output/reproduced --plots
python -m unittest discover -s code/binary/tests -v
```

Outside the repository, use the script's absolute path. The default data are
resolved from that script's repository root (`results/binary_v1`), independently
of the current directory. `--data /path/to/binary_v1` overrides the location and
must contain both authenticated inputs and the twelve published reference CSVs.
`--output` must be fresh. `--bootstrap 2000 --seed 17` are the frozen defaults.
The bundled `code/binary/frozen_protocol_v2.md` is passed explicitly to both
producers and checked against SHA256
`f2eacb2dbbd8f5a145995c55da7875dfe13d9052e59d54a2a9d3dce2400a6996`.
An unreadable or changed explicit `--protocol` is rejected before output creation.

The output contains `binary_v1/` (twelve legacy CSVs), `joint_metrics_v1/`
(eighteen new CSVs, shared draws and analysis metadata), and, with `--plots`,
`figures/`. Parent and producer receipts record source/input/protocol hashes,
actual settings, environment, exposures and timing. Partial failures remain in
place and must not be overwritten. After independent legacy generation, the
entry compares all twelve tables against the published references, including
headers, row keys, exact categories/integers/nulls and floating values at absolute
tolerance `1e-9`; see `legacy_validation.json`. Reference CSVs do not feed either
scientific producer. The release acceptance additionally compared all thirty
CSV exports at `1e-12` and the full shared bootstrap matrix.

## What the profiles measure

Task success is binary semantic correctness AND format compliance. Semantic and
Semantic-on-fixed-Task views are sensitivity analyses. Completion is required;
unknown/error states cannot silently become incorrect answers. The original
answer and scoring contracts are described in [SCORING.md](SCORING.md).

| Quantity | Definition and interpretation |
|---|---|
| JSR | Joint accuracy pooled over root–cell records whose two corresponding single views pass. Its complement JFR contains the same information. |
| ECR | Eligible records divided by all 1500 × 400 root–cell records; always interpret JSR with support. |
| JSS | Fraction of strict41 roots passing all 400 joint conditions. Strict41 requires clean plus all forty singles to pass. It equals one minus affected-question coverage on the same suite. |
| TGR5 | Mean of the largest `ceil(.05 × V)` conditional cell failure rates, over the fixed cells with at least thirty eligible roots. This is a discrete tail mean, not interpolated fixed-mass CVaR. |
| H(τ) | Fraction of the same supported cells whose failure rate is strictly greater than τ; the profile is recoverable from the conditional cell map. |

Each quantity is exported on own-model cohorts and common4 cohorts. Own cohorts
are descriptive and have different populations. Common4 is the fixed intersection
of the four models' qualification masks, per exact cell or separately for
strict41; it is not a deployment-independent filter. Cell tables retain root and
original-image counts, and per-source exports show composition. The thirty-root
cutoff is a numerical support rule, not thirty independent images or a
significance threshold. Sensitivities use all predefined tail fractions
`.01/.05/.10` and support cutoffs `10/30/50`. Source-equal summaries average the
three within-source ratios, with NA for empty sources; no source-equal tail score
is introduced.

Common4 uncertainty uses 2000 draws of all 1065 original image IDs with
replacement (seed 17 / PCG64), carrying every model, question and condition
together. One matrix is shared across views and models. Original cohort and valid
cell masks stay fixed; any originally valid cell losing all support invalidates
that TGR draw. Exports retain valid draw IDs, invalid counts and same-draw model
contrasts. Intervals are pointwise descriptive 95% percentile intervals using
NumPy linear quantiles; they provide no simultaneous significance guarantee.
Own-cohort intervals are not estimated.

Full-population clean/single/joint accuracy, clean-normalized joint drop and
retention, and the minimum full-root joint-cell accuracy provide scalar context.
They are computed from the same labels, without claiming a complete
reimplementation of another benchmark. There is no weighted overall ranking.
JSR is eligible pooled accuracy, JSS complements suite coverage, and TGR/H
summarize conditional maps. The information of interest is paired qualification
and cross-condition root alignment, rather than algebraically independent scores.

## Selected observed cases and figures

`matched_candidates.csv` exports all 1200 within-model pair-of-pairs per view
(3600 total). Inclusion requires an absolute mean difference at most one
percentage point. Selection takes the first three per view, ordered by exact
equal correct count, descending survival difference, ascending mean difference,
then lexical model/pair keys. The rule is frozen; no relaxed fallback is used.
`matched_selected.csv` retains selection flags/ranks. The observed figure uses
only the Task selection and labels the 16-condition pair-suite survival, which
is distinct from 400-condition JSS. Cases are a disclosed descriptive search,
without selected-case significance intervals.

The first Task case compares BLIP-2 brightness × keyboard with contrast × OCR
on the same 88 strict41 roots (77 original images): both have 1399/1408 correct
records, while 82/88 versus 85/88 roots pass all sixteen pair conditions. The
two aligned panels show both endpoints and component counts. All cases and
sources remain available; the figure is not evidence of architectural cause.

```bash
python code/binary/plot_joint_metrics.py --data results/joint_metrics_v1 --output output/new_figures
```

The plotter reads exported CSV only and produces 180 mm SVG with editable text,
PDF with TrueType fonts and a 300 dpi PNG. `figure_provenance.json` pins the source
CSV and outputs. Historical figures are unchanged. Explicitly synthetic
equal-marginal JSR and equal-cell-map JSS examples are unit-tested and described
in `analysis_metadata.json`; they are not substituted for observed results.

## Reproducibility tiers

| Tier | Evidence and requirements |
|---|---|
| Public label → analysis | Tested locally, including default data resolution from an external clean working directory, 12 legacy tables, 18 new tables, identical shared draws and figure export. |
| Raw answers → binary labels | Public frozen scoring code and protocol are separate. Regrading requires historical raw answers, references and the fixed Judge/runtime; this CPU command does not execute it. |
| Source inputs → new model answers | Not newly executed or certified. Requires licensed source imagery, perturbations, native processors, pinned model weights, suitable GPU/runtime and subsequent binary regrading. |

`code/binary/inference_contract.json` provides an allowlisted historical identity
contract for all four model repositories and full revisions, generation settings,
precision/batches, retained config hashes and provenance qualifications. BLIP-2's
public reproduction revision does not establish its missing original-run config
hash. The retained historical environment snapshot is incomplete and does not
recover an exact Transformers source commit or full pip/conda export. Model
weights are not redistributed; original image licenses still apply. Historical
F1 commands and strength grids require adaptation to the current nonzero 1–4
binary protocol. The generation provenance does not redefine the current score.
