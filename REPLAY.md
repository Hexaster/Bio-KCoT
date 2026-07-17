# Bio-KCoT checkpoint replay

The replay path uses the available Qwen3-8B artifacts and does not retrain a model:

- SFT merged model: `model/kg_clean/qwen3-8b/merge`
- outcome-RL adapter: `model/GRPO/qwen3-8b/checkpoint-110`
- KG/process-RL adapter: `model/GRPO/qwen3-8b-kg/checkpoint-100`
- complete test set: the four CSV files under `dataset/data/test` (1,121 rows)

All replay paths are configured under `paths` in `config.json`:

```json
"replay_sft_merged": "model/kg_clean/qwen3-8b/merge",
"replay_outcome_checkpoint": "model/GRPO/qwen3-8b/checkpoint-110",
"replay_process_checkpoint": "model/GRPO/qwen3-8b-kg/checkpoint-100",
"replay_test_dir": "dataset/data/test",
"replay_output_dir": "evaluate/replay_results"
```

Relative paths are resolved from the repository root. Absolute paths may be used for artifacts stored elsewhere. The same values can be overridden without editing `config.json` by setting `REPLAY_SFT_MERGED`, `REPLAY_OUTCOME_CHECKPOINT`, `REPLAY_PROCESS_CHECKPOINT`, `REPLAY_TEST_DIR`, and `REPLAY_OUTPUT_DIR` in `.env`.

## 1. Copy the repository to the GPU server

The three replay variants need about 17 GB of model files. Optimizer states and older checkpoints are not required for inference.

## 2. Prepare the inference environment

Use a CUDA-enabled Python environment containing PyTorch, Transformers, PEFT, pandas, tqdm, and Accelerate. The local CPU machine can run the artifact audit, but inference requires the GPU environment.

## 3. Audit without loading the model

```bash
python3 evaluate/audit_checkpoints.py
```

This checks the merged-model index, every selected safetensors payload, LoRA rank, tokenizer files, and all 1,121 test rows.

## 4. Smoke test on the A100

```bash
python3 evaluate/replay_checkpoints.py --max-rows 4 --batch-size 1
```

Outputs are written to `evaluate/replay_results`. The default decoding is greedy so repeated runs are deterministic.

## 5. Full replay

```bash
python3 evaluate/replay_checkpoints.py --batch-size 2 --resume
```

If memory is insufficient, use `--batch-size 1`. The script loads each variant once and evaluates the combined 1,121-row test set.

## 6. Optional LLM-judge evaluation

Set `JUDGE_API_KEY`, `JUDGE_BASE_URL`, and `JUDGE_MODEL` in `.env`, then run:

```bash
python3 evaluate/replay_checkpoints.py --batch-size 2 --resume --judge
```

Judging is opt-in because it sends questions, reference answers, evidence paths, and model responses to the configured API and may incur cost.

## Scope

This replays the surviving artifacts. It does not prove that the checkpoints were produced with the exact paper hyperparameters. In particular, the selected RL checkpoints are intermediate training checkpoints, and the repository's historical KG reward implementation is not a literal step-wise implementation of the paper's process-reward equation.
