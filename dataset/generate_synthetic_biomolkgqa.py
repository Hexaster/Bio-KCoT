#!/usr/bin/env python3
"""Generate a deterministic, schema-compatible synthetic BioMolKGQA fixture.

The generated entities and mechanisms are fictional.  This data is intended for
pipeline smoke tests only; it is not a replacement for scientific evaluation.
"""

import argparse
import csv
import json
import random
from pathlib import Path


SYSTEM_PROMPT = """Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""

TASKS = {
    "disease-indication": {
        "filename": "disease-indication_{split}.csv",
        "hop_range": (4, 11),
        "fields": ["kg_source_id", "source"],
    },
    "drug-synergy": {
        "filename": "drug-synergy_{split}.csv",
        "hop_range": (3, 9),
        "fields": [
            "source_kg_index",
            "source_triplet_str",
            "drug_a_name",
            "drug_b_name",
        ],
    },
    "PPI_reasoning": {
        "filename": "PPI_reasoning_{split}.csv",
        "hop_range": (2, 15),
        "fields": ["target_protein_id", "interaction_count"],
    },
    "reactome_reasoning": {
        "filename": "reactome_reasoning_{split}.csv",
        "hop_range": (2, 23),
        "fields": ["Pathway_ID", "Pathway_Name"],
    },
}

CORE_FIELDS = [
    "question",
    "answer",
    "explanation",
    "evidence_KG",
    "hop",
]
SCORE_FIELDS = ["score", "score_reason"]

RELATIONS = (
    "activates",
    "enables",
    "increases",
    "stabilizes",
    "releases",
    "transmits_signal_to",
)


def chain(token, hop, start, end, first_relation, rng):
    nodes = [start]
    nodes.extend(f"SynNode-{token}-{i:02d}" for i in range(1, hop))
    nodes.append(end)
    relations = [first_relation]
    relations.extend(rng.choice(RELATIONS) for _ in range(hop - 1))
    return [f"({nodes[i]}, {relations[i]}, {nodes[i + 1]})" for i in range(hop)]


def facts_text(evidence):
    facts = []
    for item in evidence:
        head, relation, tail = item[1:-1].split(", ", 2)
        facts.append(f"{head} {relation.replace('_', ' ')} {tail}.")
    return " ".join(facts)


def explanation_text(evidence, conclusion):
    steps = [f"{i}. The synthetic graph states {triple}." for i, triple in enumerate(evidence, 1)]
    steps.append(f"Therefore, within this fictional graph, the answer is {conclusion}.")
    return " ".join(steps)


def base_record(question, answer, evidence, hop):
    return {
        "question": question,
        "answer": answer,
        "explanation": explanation_text(evidence, answer),
        "evidence_KG": json.dumps(evidence, ensure_ascii=False),
        "hop": hop,
        "score": 8,
        "score_reason": (
            "Synthetic schema fixture. The score preserves compatibility with the original "
            "quality filter and is not a biological-quality assessment."
        ),
    }


def disease_record(token, numeric_id, hop, rng):
    agent = f"SynAgent-{token}"
    driver = f"SynNode-{token}-01"
    phenotype = f"SynDiseaseState-{token}"
    decoy = f"SynDecoy-{token}"
    evidence = chain(token, hop, agent, phenotype, "inhibits", rng)
    question = (
        "This is a fictional mechanism-logic case, not medical advice. "
        f"{facts_text(evidence)} A second agent, {decoy}, does not inhibit {driver}. "
        f"Which synthetic agent should be selected to block the initiating mechanism?"
    )
    record = base_record(question, agent, evidence, hop)
    record.update({"kg_source_id": numeric_id, "source": f"({numeric_id}, indication, {numeric_id + 1})"})
    return record


def ddi_record(token, numeric_id, hop, rng):
    drug_a = f"SynDrugA-{token}"
    drug_b = f"SynDrugB-{token}"
    pair = f"{drug_a}+{drug_b}"
    outcome = f"SynCombinedOutcome-{token}"
    evidence = chain(token, hop, pair, outcome, "jointly_perturbs", rng)
    question = (
        "This is a fictional interaction-logic case, not medical advice. "
        f"{drug_a} and {drug_b} are co-administered. {facts_text(evidence)} "
        "What synthetic downstream outcome follows from their co-administration?"
    )
    record = base_record(question, outcome, evidence, hop)
    record.update(
        {
            "source_kg_index": numeric_id,
            "source_triplet_str": f"({numeric_id}, synergistic interaction, {numeric_id + 1})",
            "drug_a_name": f"{drug_a} is a fictional compound used only for software testing.",
            "drug_b_name": f"{drug_b} is a fictional compound used only for software testing.",
        }
    )
    return record


def ppi_record(token, numeric_id, hop, rng):
    protein = f"SynProtein-{token}"
    perturbation = f"{protein}-loss_of_function"
    outcome = f"SynSignalState-{token}"
    evidence = chain(token, hop, perturbation, outcome, "disrupts_interaction_with", rng)
    question = (
        "In a fictional protein network, "
        f"{facts_text(evidence)} What is the predicted downstream synthetic signal state?"
    )
    record = base_record(question, outcome, evidence, hop)
    record.update(
        {
            "target_protein_id": f"synthetic:{numeric_id}",
            "interaction_count": hop + 2,
        }
    )
    return record


def reactome_record(token, numeric_id, hop, rng):
    perturbation = f"SynPathwayPerturbation-{token}"
    outcome = f"SynPathwayOutcome-{token}"
    evidence = chain(token, hop, perturbation, outcome, "prevents_assembly_of", rng)
    question = (
        "Consider this fictional counterfactual pathway. "
        f"{facts_text(evidence)} What synthetic pathway outcome is predicted?"
    )
    record = base_record(question, outcome, evidence, hop)
    record.update(
        {
            "Pathway_ID": f"SYN-PATH-{numeric_id}",
            "Pathway_Name": f"Fictional pathway {token}",
        }
    )
    return record


BUILDERS = {
    "disease-indication": disease_record,
    "drug-synergy": ddi_record,
    "PPI_reasoning": ppi_record,
    "reactome_reasoning": reactome_record,
}


def csv_fields(task):
    extra = TASKS[task]["fields"]
    if task == "reactome_reasoning":
        return extra + CORE_FIELDS + SCORE_FIELDS
    return CORE_FIELDS + extra + SCORE_FIELDS


def generate_task(task, split, count, seed):
    spec = TASKS[task]
    split_code = "TR" if split == "train" else "TE"
    task_code = task.replace("-", "_").upper()
    minimum, maximum = spec["hop_range"]
    rows = []
    for index in range(count):
        token = f"{task_code}-{split_code}-{index:05d}"
        task_offset = list(TASKS).index(task) * 1_000_000
        numeric_id = (1 if split == "train" else 8) * 10_000_000 + task_offset + index * 2
        rng = random.Random(f"{seed}:{token}")
        hop = minimum + rng.randrange(maximum - minimum + 1)
        rows.append(BUILDERS[task](token, numeric_id, hop, rng))
    return rows


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_training_formats(output_dir, train_rows):
    sft_rows = []
    rl_rows = []
    for task in TASKS:
        ordered = sorted(train_rows[task], key=lambda row: (int(row["hop"]), row["question"]))
        boundary = max(1, round(len(ordered) * 0.6))
        if len(ordered) > 1:
            boundary = min(boundary, len(ordered) - 1)
        for row in ordered[:boundary]:
            sft_rows.append(
                {
                    "instruction": "Answer the question and think step by step.\n",
                    "input": row["question"],
                    "output": (
                        f"<think>\n{row['explanation']}\n</think>\n"
                        f"<answer>\n{row['answer']}\n</answer>"
                    ),
                    "system": SYSTEM_PROMPT,
                    "task_type": task,
                    "synthetic": True,
                }
            )
        for row in ordered[boundary:]:
            rl_rows.append(
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": row["question"] + "\nAnswer the question and think step by step.",
                        },
                    ],
                    "answer": row["answer"],
                    "evidence_graph": row["evidence_KG"],
                    "question": row["question"],
                    "task_type": task,
                    "synthetic": True,
                }
            )

    sft_path = output_dir / "SFT" / "sft.json"
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.write_text(json.dumps(sft_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rl_path = output_dir / "RL" / "rl.jsonl"
    rl_path.parent.mkdir(parents=True, exist_ok=True)
    with rl_path.open("w", encoding="utf-8") as handle:
        for row in rl_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(sft_rows), len(rl_rows)


def validate(output_dir, expected_train, expected_test):
    anchors = {
        "disease-indication": "kg_source_id",
        "drug-synergy": "source_kg_index",
        "PPI_reasoning": "target_protein_id",
        "reactome_reasoning": "Pathway_ID",
    }
    for task, spec in TASKS.items():
        split_anchors = {}
        for split, expected in (("train", expected_train), ("test", expected_test)):
            path = output_dir / split / spec["filename"].format(split=split)
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != expected:
                raise ValueError(f"{path}: expected {expected} rows, found {len(rows)}")
            for row in rows:
                evidence = json.loads(row["evidence_KG"])
                if len(evidence) != int(row["hop"]):
                    raise ValueError(f"{path}: hop does not match evidence length")
                if "fictional" not in row["question"].lower():
                    raise ValueError(f"{path}: missing synthetic-data disclosure")
            split_anchors[split] = {row[anchors[task]] for row in rows}
        if split_anchors["train"] & split_anchors["test"]:
            raise ValueError(f"{task}: train/test entity leakage detected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/synthetic_biomolkgqa"))
    parser.add_argument("--train-per-task", type=int, default=12)
    parser.add_argument("--test-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_per_task < 2 or args.test_per_task < 1:
        parser.error("--train-per-task must be >= 2 and --test-per-task must be >= 1")

    output_dir = args.output_dir.resolve()
    train_rows = {}
    test_rows = {}
    for task, spec in TASKS.items():
        fields = csv_fields(task)
        train_rows[task] = generate_task(task, "train", args.train_per_task, args.seed)
        test_rows[task] = generate_task(task, "test", args.test_per_task, args.seed)
        write_csv(
            output_dir / "train" / spec["filename"].format(split="train"),
            fields,
            train_rows[task],
        )
        write_csv(
            output_dir / "test" / spec["filename"].format(split="test"),
            fields,
            test_rows[task],
        )

    combined_fields = ["task_type"]
    combined_rows = []
    for task in TASKS:
        for field in csv_fields(task):
            if field not in combined_fields:
                combined_fields.append(field)
        combined_rows.extend({"task_type": task, **row} for row in test_rows[task])
    write_csv(output_dir / "test.csv", combined_fields, combined_rows)

    sft_count, rl_count = write_training_formats(output_dir, train_rows)
    manifest = {
        "name": "BioMolKGQA-Synthetic",
        "seed": args.seed,
        "synthetic": True,
        "intended_use": "Software and training-pipeline smoke tests only",
        "not_for": "Scientific benchmarking, medical decisions, or comparison with paper results",
        "counts": {
            "train_per_task": args.train_per_task,
            "test_per_task": args.test_per_task,
            "sft": sft_count,
            "rl": rl_count,
            "combined_test": len(combined_rows),
        },
        "tasks": list(TASKS),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dataset_card = f"""# BioMolKGQA-Synthetic

Deterministic fictional data for Bio-KCoT software smoke tests.

- Seed: `{args.seed}`
- Training rows: `{args.train_per_task * len(TASKS)}`
- Test rows: `{len(combined_rows)}`
- SFT rows: `{sft_count}`
- RL rows: `{rl_count}`

All entities and mechanisms are synthetic. Do not use this dataset for scientific
benchmarking, medical decisions, or comparison with the paper's reported scores.
The repository does not currently declare a license; downstream redistribution
terms must be chosen by the repository owner.
"""
    (output_dir / "README.md").write_text(dataset_card, encoding="utf-8")
    validate(output_dir, args.train_per_task, args.test_per_task)
    print(f"Generated and validated {output_dir}")
    print(f"Train: {args.train_per_task * len(TASKS)}, test: {len(combined_rows)}, SFT: {sft_count}, RL: {rl_count}")


if __name__ == "__main__":
    main()
