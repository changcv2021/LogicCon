# Method and implementation

The framework implements the inference procedure in paper §5 and Appendix A.8.

| Paper component | Code |
|---|---|
| Target-aware claim parsing | `schema.Claim`, `engine.Reasoner.run`, `prompts.PARSE` |
| Claim-grounded visual verification | `schema.Verification`, `prompts.VERIFY` |
| Iterative evidence supplementation | `prompts.REFINE`, bounded per-claim loop |
| Conflict aggregation/type diagnosis | `prompts.AGGREGATE`, model call and protocol validation |
| Conflict-guided final response | `prompts.FINAL`, preservation of diagnosed fields |
| Semantic conflict-point evaluation | `judge.py`, `metrics.py` |

Claims encode the target, predicate, claimed detail and contextual constraints. The parser preserves relation direction, negation, quantifiers, count scope and nested references. It does not receive the image or ground-truth records. Every later inference module uses the same backbone object and the original image.

Each verification returns textual visual evidence and one of `supported`, `contradicted`, or `insufficient`. Refinement only runs for `insufficient`; it receives the original claim and earlier observations. `max_iterations=3` is the total number of rounds, not three additional retries. A supported/contradicted verdict requires nonempty evidence.

Aggregation is a model call. A protocol validator enforces an established contradiction, an all-supported result, and the provenance of a selected conflict pair. Type diagnosis uses the meaning of the claim and visual evidence. For example, holding a phone versus holding a remote can be an object conflict despite the shared interaction predicate.

When some evidence remains insufficient and no contradiction is established, the model gives a conservative binary response and records `uncertain=true`. The returned `evidence_status` keeps this condition explicit. Final reasoning preserves the diagnosed fields and produces a concise explanation.

## Reproduction settings

The public module prompts are in `prompts.py`. They operationalize the paper descriptions with a strict JSON interface. Direct-answer and judge prompts follow Appendix A.12/A.13. Stage-specific wording and decoding/image budgets are configurable implementation choices.

Defaults are greedy decoding, seed 20260926, 1536 output tokens per call, at most 16 parsed claims, one format/protocol retry, and three verification rounds. Qwen image budgets default to 256–1024 units of 28×28 pixels. LLaVA uses its processor's native image handling.

For `n` claims with verification counts `k_i`, the full method requires `3 + sum(k_i)` calls before retries. One-call direct inference and multi-stage inference have different costs; compare measured tokens, calls and elapsed time in addition to accuracy. Format/protocol retries are logged separately in the trace and count toward the budget.

## Model output validation

Empty claim lists, duplicate claim IDs, missing visual evidence, changed primary claim IDs, or final-answer drift are rejected. A retry asks the model to correct the JSON/protocol error. Exhausting the retry budget records a failed sample; it does not silently create a plausible label.

Resume validates input/configuration/code/environment fingerprints. Image bytes are hashed per sample. Local model configuration files are hashed; large weights are recorded by size and modification time. Use an immutable model snapshot when preserving an experiment. Remote servers must pin their served weights independently.

Tests exercise these software contracts with scripted responses. The `verification` ablation feeds verified records to a direct task response; `aggregation` stops after the explicit diagnosis; `full` adds the final constrained explanation.
