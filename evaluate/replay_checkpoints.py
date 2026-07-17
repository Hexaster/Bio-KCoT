import argparse
import csv
import subprocess
import sys
from pathlib import Path

from audit_checkpoints import TEST_FILES, audit


TASK_TYPES = {
    "disease-indication_test.csv": "Disease Indication",
    "drug-synergy_test.csv": "Drug Synergy",
    "PPI_reasoning_test.csv": "PPI Reasoning",
    "reactome_reasoning_test.csv": "Reactome Reasoning",
}


def build_combined_test(test_dir, output_path):
    rows = []
    fieldnames = ["task_type"]
    for name in TEST_FILES:
        with (test_dir / name).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
            for row in reader:
                row["task_type"] = TASK_TYPES[name]
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run(command):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True)


def main():
    parser = argparse.ArgumentParser(description="Replay the available Qwen3-8B Bio-KCoT artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["sft", "outcome", "process"],
        default=["sft", "outcome", "process"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0, help="Use a small prefix for a smoke test.")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--judge", action="store_true", help="Judge predictions using evaluate_qwen3_8b.py and env credentials.")
    parser.add_argument("--judge-model", default="")
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_dir = (args.output_dir or root / "evaluate/replay_results").resolve()
    audit(root)

    combined_test = output_dir / "biomolkgqa_test_1121.csv"
    row_count = build_combined_test(root / "dataset/data/test", combined_test)
    print(f"Combined replay test set: {combined_test} ({row_count} rows)")

    merged = root / "model/kg_clean/qwen3-8b/merge"
    variants = {
        "sft": ("", "model/kg_clean/qwen3-8b/merge"),
        "outcome": (root / "model/GRPO/qwen3-8b/checkpoint-110", "model/GRPO/qwen3-8b/checkpoint-110"),
        "process": (root / "model/GRPO/qwen3-8b-kg/checkpoint-100", "model/GRPO/qwen3-8b-kg/checkpoint-100"),
    }

    predictor = root / "evaluate/local_hf_lora_predict.py"
    evaluator = root / "evaluate/evaluate_qwen3_8b.py"
    for name in args.variants:
        adapter, model_name = variants[name]
        predictions = output_dir / f"qwen3-8b-{name}-predictions.csv"
        command = [
            sys.executable,
            predictor,
            "--base-model",
            merged,
            "--input",
            combined_test,
            "--predictions",
            predictions,
            "--model-name",
            model_name,
            "--prompt-style",
            "biokcot",
            "--batch-size",
            args.batch_size,
            "--max-new-tokens",
            args.max_new_tokens,
            "--temperature",
            args.temperature,
        ]
        if adapter:
            command.extend(("--adapter", adapter))
        if args.max_rows:
            command.extend(("--max-rows", args.max_rows))
        if args.resume:
            command.append("--resume")
        run(command)

        if args.judge:
            judged = output_dir / f"qwen3-8b-{name}-judged.csv"
            summary = output_dir / f"qwen3-8b-{name}-summary.json"
            command = [
                sys.executable,
                evaluator,
                "--judge-only",
                "--predictions",
                predictions,
                "--judged",
                judged,
                "--summary",
                summary,
            ]
            if args.judge_model:
                command.extend(("--judge-model", args.judge_model))
            if args.resume:
                command.append("--resume")
            run(command)


if __name__ == "__main__":
    main()
