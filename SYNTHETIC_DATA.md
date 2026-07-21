# BioMolKGQA synthetic reproduction data

## What the original dataset is for

The manuscript calls the benchmark **BioMolKGQA**. It is an open-ended QA
dataset for training and evaluating knowledge-graph-grounded, multi-hop
biomolecular reasoning. Each row contains a question, short answer, explanatory
reasoning, a list of KG evidence triples, and the number of reasoning hops.

Naming is version-sensitive: the KDD draft bundled in this working repository
uses `BioMolKGQA` and the four open-ended tasks below, while the public arXiv
version at <https://arxiv.org/abs/2511.08024> describes an MCQ benchmark named
`PrimeKGQA`. This generator intentionally follows the repository's CSV schemas,
not the public paper's different MCQ format.

It covers four tasks:

| Task | Purpose | Source described by the manuscript |
| --- | --- | --- |
| Disease Indication | Select a treatment and explain its mechanism | PrimeKG |
| Drug Synergy / DDI | Infer joint or adverse effects through converging mechanisms | PrimeKG |
| PPI Reasoning | Infer downstream effects of protein perturbations | Reactome |
| Reactome Reasoning | Explain or predict pathway and counterfactual outcomes | Reactome |

The manuscript reports 8,279 training rows and 1,121 test rows after filtering
and entity-disjoint splitting. The repository ignores `dataset/data/`, so those
rows are not part of a normal Git clone even though the generation and training
scripts refer to them.

## What this replacement does

`dataset/generate_synthetic_biomolkgqa.py` creates a deterministic dataset with
the same four CSV filenames and compatible columns. It also creates:

- `test.csv`: a combined evaluation file;
- `SFT/sft.json`: low-hop instruction-tuning examples;
- `RL/rl.jsonl`: higher-hop records in the format consumed by the GRPO scripts;
- `manifest.json`: generation settings and an explicit synthetic-data label.

Every entity and mechanism is fictional and every question says so. The data is
useful for checking CSV loading, prompt formatting, SFT/GRPO plumbing, checkpoint
inference, and evaluation output. It **cannot** reproduce the paper's reported
accuracy or support biomedical conclusions.

## Generate the included fixture

From the repository root:

```bash
python3 dataset/generate_synthetic_biomolkgqa.py
```

The default fixture has 12 training and 4 test rows per task. To exercise a
larger run while preserving deterministic generation:

```bash
python3 dataset/generate_synthetic_biomolkgqa.py \
  --output-dir /tmp/biomolkgqa-synthetic \
  --train-per-task 1000 \
  --test-per-task 100 \
  --seed 42
```

The generator validates row counts, parses every `evidence_KG` value, checks
that `hop` equals the evidence-path length, and rejects train/test anchor overlap.

## Use it with this repository

Run local model inference on the combined synthetic test set:

```bash
python3 evaluate/local_hf_lora_predict.py \
  --base-model Qwen/Qwen3-8B \
  --input dataset/synthetic_biomolkgqa/test.csv \
  --predictions evaluate/results/synthetic_predictions.csv \
  --max-rows 4
```

Point the GRPO training script at the generated higher-hop split:

```bash
RL_DATASET=dataset/synthetic_biomolkgqa/RL/rl.jsonl \
  python3 code/unsloth_GRPO.py
```

`SFT/sft.json` follows the repository's `instruction`, `input`, `output`, and
`system` convention. Its `output` contains the required `<think>` and `<answer>`
tags.

## Reproducibility boundary

This fixture provides **software reproducibility**: others can run and debug the
data, training, and evaluation interfaces without receiving the unpublished
benchmark. Reproducing the scientific results still requires the authors to
release the original rows, exact generation prompts/models, quality-control
outputs, entity-disjoint split IDs, and dataset/license provenance.
