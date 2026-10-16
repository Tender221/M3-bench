# Code entry points

The current M³-Test binary analysis starts at `binary/analyze_binary_masks.ps1`; it uses completed included labels and needs no model/GPU. Frozen scoring protocol and safe-exact functions live beside it.

Other pre-existing files in this directory are the retained legacy configurable perturbation/inference/EM/F1 toolkit. Their paths and contents have not been changed by the binary-results refresh. Read the archived instructions in `../docs/legacy/` before using them; an old EM/F1 result or default severity grid is not the current binary paper result.
