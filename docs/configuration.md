# Configuration and execution

TOML files contain `[backend]` and `[method]`. Unknown fields fail validation. `${ENV_VAR}` placeholders are expanded when loading a configuration; unresolved placeholders produce an error.

## Local Transformers

```bash
export LOGICCON_MODEL_PATH=/path/to/local/model-snapshot
logiccon-reason validate-config --config configs/reasoning/qwen2_5_vl_7b.toml
```

Available presets: `qwen2_5_vl_7b.toml`, `qwen2_5_vl_32b.toml`, and `llava_1_5.toml`. Use the corresponding checkpoint for each preset. The LLaVA example uses the HF-converted `llava-hf/llava-1.5-7b-hf` interface.

`local_files_only=true` avoids implicit downloads. `device_map=auto` distributes a large model over visible allocated GPUs; CPU/disk offload is rejected. Precision and image budgets are configured explicitly. If using a Hub model ID rather than a local path, set `revision` to a 40-character immutable commit ID and prepare that snapshot in the cache (or explicitly allow downloads in a suitable environment).

`max_new_tokens` limits each call. Increase it if complete JSON cannot fit. Increasing it also raises inference cost. Contexts are never silently truncated.

## Compatible HTTP service

```bash
export LOGICCON_SERVED_MODEL=your-served-vision-model
export LOGICCON_BASE_URL=http://your-server:8000/v1
# LOGICCON_API_KEY is optional for an unauthenticated local server.
logiccon-reason infer --config configs/reasoning/http.toml \
  --inputs inputs.jsonl --image-root /path/to/images-root --output runs/http
```

The adapter posts to `<base_url>/chat/completions` using `messages`, image data URLs, `temperature=0`, `max_tokens`, and `seed`. It expects `choices[0].message.content`. Use a service that supports these fields and multimodal content. Native provider APIs with different contracts require a separate adapter.

The configured endpoint receives the prompt and, for visual calls, image bytes. Keys are read from the named environment variable, omitted from run manifests, and never embedded in configuration files. Retryable transport failures are bounded; HTTP error bodies are not printed.

## Sharding and Slurm

Sharding uses input position modulo `num_shards`. `--limit`, if supplied, applies to the input prefix before sharding. Keep selection settings the same in every shard.

`scripts/slurm_infer.sh` runs an already-installed environment. Set its required environment variables and invoke `sbatch` with your site's verified resource requests. It has no hard-coded account, partition, GPU count, memory, CPUs or wall time. A typical command shape is:

```bash
export LOGICCON_CONFIG="$PWD/configs/reasoning/qwen2_5_vl_7b.toml"
export LOGICCON_INPUTS=/path/to/LogicCon/inputs/test.jsonl
export LOGICCON_IMAGE_ROOT=/path/to/LogicCon
export LOGICCON_OUTPUT=/path/to/runs/full
export LOGICCON_NUM_SHARDS=2
export LOGICCON_PYTHON=/path/to/venv/bin/python
export LOGICCON_MODEL_PATH=/path/to/model-snapshot
# Optional: LOGICCON_MODULE=<site-supported-module>, LOGICCON_MODE=base|...|full
# Optional: LOGICCON_RESUME=1, LOGICCON_LIMIT=<positive-prefix-limit>

# After supplying appropriate verified account/QOS/resources and creating log directories:
# sbatch <site-resource-flags> --array=0-1 \
#   --output=/path/to/logs/%x_%A_%a.out --error=/path/to/logs/%x_%A_%a.err \
#   scripts/slurm_infer.sh
```

Independent array elements have separate output directories and no artificial dependencies. Do not start model inference on an HPC login node.

## Restart and provenance

`--resume` requires the same configuration, input file, selection, code and dependency metadata. Completed samples are checked and skipped; failed samples are retried. A run directory has an exclusive process lock. Truncated final writes are saved to a `.partial-*` file before recovery.

To change an experiment, choose a new directory. Merge only completed shards with matching manifests. Export `requirements.resolved.txt` from the execution environment when recording a result.
