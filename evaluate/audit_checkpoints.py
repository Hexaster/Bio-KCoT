import argparse
import csv
import json
import struct
from pathlib import Path


TEST_FILES = (
    "disease-indication_test.csv",
    "drug-synergy_test.csv",
    "PPI_reasoning_test.csv",
    "reactome_reasoning_test.csv",
)


def validate_safetensors(path):
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError("missing safetensors header length")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > 100_000_000 or 8 + header_length > size:
            raise ValueError(f"invalid safetensors header length: {header_length}")
        header = json.loads(handle.read(header_length))

    tensors = [value for key, value in header.items() if key != "__metadata__"]
    payload_size = size - 8 - header_length
    final_offset = max((item["data_offsets"][1] for item in tensors), default=0)
    if final_offset != payload_size:
        raise ValueError(f"truncated payload: expected {final_offset}, found {payload_size}")
    return len(tensors)


def require_files(directory, names):
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing {', '.join(missing)}")


def audit(project_root):
    merged = project_root / "model/kg_clean/qwen3-8b/merge"
    outcome = project_root / "model/GRPO/qwen3-8b/checkpoint-110"
    process = project_root / "model/GRPO/qwen3-8b-kg/checkpoint-100"
    test_dir = project_root / "dataset/data/test"

    require_files(
        merged,
        ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"),
    )
    index = json.loads((merged / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        count = validate_safetensors(merged / shard)
        print(f"OK SFT shard: {shard} ({count} tensors)")

    for label, checkpoint in (("outcome", outcome), ("process", process)):
        require_files(checkpoint, ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json"))
        adapter = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
        count = validate_safetensors(checkpoint / "adapter_model.safetensors")
        if adapter.get("r") != 32:
            raise ValueError(f"{checkpoint}: expected LoRA rank 32, found {adapter.get('r')}")
        print(f"OK {label} adapter: {checkpoint.relative_to(project_root)} ({count} tensors, rank 32)")

    total = 0
    for name in TEST_FILES:
        path = test_dir / name
        require_files(test_dir, (name,))
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = sum(1 for _ in csv.DictReader(handle))
        print(f"OK test data: {name} ({rows} rows)")
        total += rows
    if total != 1121:
        raise ValueError(f"expected 1121 test rows, found {total}")
    print("OK complete replay set: 1121 rows")


def main():
    parser = argparse.ArgumentParser(description="Validate Bio-KCoT replay artifacts without loading a model.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    audit(args.project_root.resolve())


if __name__ == "__main__":
    main()
