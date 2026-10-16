# Frozen binary scoring and interpretation

The primary task score is:

```text
task_pass = semantic_correct AND format_compliant
```

Both axes are actual booleans. Inference/parsing failures are execution errors, not automatically incorrect answers. The completed release has no inference-error rows.

## Safe-exact then two independent judgments

The frozen router normalizes a restricted set of exact-reference responses. It preserves internal punctuation, word order, signs and negation. The response must match an acceptable reference after this normalization and have at most four whitespace-delimited words to use the safe-exact shortcut. All other valid prediction rows are sent to separate semantic and format judgments.

`code/binary/safe_matcher.py` contains verbatim extracted `normalize_safe`, `safe_exact`, `initial_decision` and `judge_key` functions from the executed router. `scoring_protocol.py` is the byte-identical executed frozen binary protocol, including full prompt text and strict label parsing. `source_provenance.json` records their hashes and the source/extraction contract. Earlier historical prompt hashes retained inside the frozen protocol are provenance identities; they do not replace the current binary prompts or introduce an uncertain output.

Semantic output is exactly CORRECT/INCORRECT; format output is VALID/INVALID. The semantic Judge sees the original question, all acceptable reference alternatives and the complete prediction. Matching one acceptable reference is sufficient; contradictory, refused or unselected mutually exclusive answers do not pass. Brief correct explanations can pass semantics while failing the short-answer format. A wrong short answer can pass format while failing semantics. Payload deduplication keeps different questions/reference lists separate.

## Frozen Judge identity

- Model: Qwen/Qwen3-4B-Instruct-2507.
- Revision: `cdbee75f17c01a7cc42f958dc650907174af0554`.
- Joined binary prompt SHA-256: `80b6225be13c6621fec599778b0bc103a798f7ce4effbd97084840ba65685388`.
- Linux runtime: Python3.11.17, PyTorch2.10/CUDA12.8, Transformers4.57.6, Accelerate1.14, bitsandbytes0.50.2.
- NF4 double quantization, FP16 compute, SDPA, greedy decoding, batch16, seed17, context2048, max12 generated tokens.

The Judge does **not** see the corrupted image. Its reference-agreement decision assumes the question intent and original references still apply. The release does not independently establish image answerability, human equivalence or intent preservation. No additional human audit was performed; semantic-only sensitivity uses these same judgments.

## Populations

For an axis `z`, strict41 requires `z(clean)` and all40 `z(single)` to pass. A question is joint-affected if any of400 joint conditions fails. Event density divides failure events by `400 × strict41 question count`; affected-question coverage divides affected questions by strict41 questions.

At joint `(image family a, severity i; text family b, severity j)`, corresponding-single eligibility requires `z(image-only a,i) AND z(text-only b,j)`. Clean and other strengths are not silently added. Models can have different qualified populations, so own-population fractions are not a common ranking cohort.

For replacements/updates, reference eligibility is fixed. Repairs are reference-false/target-true records; new errors are reference-true/target-false records. Do not drop damaged records by requalifying after training.

Semantic-only sensitivity has two explicitly different questions: requalify using semantics for both anchors and outcomes, or retain the original task-qualified mask and change only the outcome axis. Both are released. Neither changes historical training qualification or touches a sealed final test.
