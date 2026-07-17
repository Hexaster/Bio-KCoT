import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_qwen3_8b import build_model_messages, parse_model_response, stable_row_id


BIOKCOT_SYSTEM_PROMPT = """Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
"""


def build_biokcot_messages(row):
    return [
        {"role": "system", "content": BIOKCOT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{row.get('question', '')}\nAnswer the question and think step by step.",
        },
    ]


def build_prompt(tokenizer, messages, enable_thinking=None):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "auto": "auto",
    }[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return tokenizer, model


def generate_batch(tokenizer, model, rows, args):
    prompts = [
        build_prompt(
            tokenizer,
            build_biokcot_messages(row) if args.prompt_style == "biokcot" else build_model_messages(row),
            enable_thinking=args.enable_thinking,
        )
        for row in rows
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.cutoff_len,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    return [
        tokenizer.decode(output_ids[input_width:], skip_special_tokens=True).strip()
        for output_ids in outputs
    ]


def main():
    parser = argparse.ArgumentParser(description="Run local HF base model + LoRA adapter predictions for BioKCoT test CSV.")
    parser.add_argument("--base-model", required=True, help="Base model ID or local path, e.g. Qwen/Qwen3-8B")
    parser.add_argument("--adapter", default="", help="Optional LoRA adapter path or model ID")
    parser.add_argument("--input", required=True, help="Input test CSV.")
    parser.add_argument("--predictions", required=True, help="Output predictions CSV.")
    parser.add_argument("--model-name", default="qwen3-8b-bionokg-lora", help="Name stored in prediction CSV.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test subset.")
    parser.add_argument("--batch-size", type=int, default=1, help="Local generation batch size.")
    parser.add_argument("--cutoff-len", type=int, default=4096, help="Prompt truncation length.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Generation tokens.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--prompt-style",
        choices=["biokcot", "evaluation"],
        default="evaluation",
        help="Use the checkpoint training prompt or the generic evaluation prompt.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--disable-thinking", action="store_true", help="Pass enable_thinking=False to Qwen3 chat template when supported.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.enable_thinking = False if args.disable_thinking else None

    input_path = Path(args.input)
    output_path = Path(args.predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, dtype=str)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    df = df.reset_index(drop=True)
    df.insert(0, "eval_row_id", [stable_row_id(i, row) for i, row in df.iterrows()])

    existing = []
    done = set()
    if args.resume and output_path.exists():
        existing_df = pd.read_csv(output_path, dtype=str)
        if "model_error" in existing_df.columns:
            ok = existing_df["model_error"].fillna("").eq("")
            existing_df = existing_df[ok].copy()
        existing = existing_df.to_dict("records")
        if "eval_row_id" in existing_df.columns:
            done = set(existing_df["eval_row_id"].astype(str))

    rows = [row.to_dict() for _, row in df.iterrows() if str(row.get("eval_row_id", "")) not in done]
    print(f"Input rows: {len(df)}", flush=True)
    print(f"Existing predictions: {len(existing)}", flush=True)
    print(f"Rows to predict: {len(rows)}", flush=True)

    tokenizer, model = load_model(args)
    results = list(existing)

    with tqdm(total=len(rows), desc="Predicting", unit="sample") as pbar:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            batch_start = time.time()
            try:
                raws = generate_batch(tokenizer, model, batch, args)
                errors = [""] * len(raws)
            except Exception as exc:
                raws = [""] * len(batch)
                errors = [str(exc)] * len(batch)

            for row, raw, error in zip(batch, raws, errors):
                merged = dict(row)
                merged["model_name"] = args.model_name
                merged["eval_model_temperature"] = args.temperature
                merged["eval_model_max_tokens"] = args.max_new_tokens
                merged["eval_model_top_p"] = args.top_p
                merged["eval_model_top_k"] = args.top_k
                merged["eval_batch_size"] = args.batch_size
                merged["eval_timeout"] = ""
                merged["model_raw"] = raw
                merged["model_error"] = error
                if raw and not error:
                    parsed = parse_model_response(raw)
                    merged["model_answer"] = parsed["answer"]
                    merged["model_reasoning"] = parsed["reasoning"]
                    merged["model_extracted_triples"] = parsed["extracted_triples"]
                    merged["model_parse_warning"] = parsed["parse_warning"]
                    merged["model_response"] = raw
                results.append(merged)

            pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
            pbar.update(len(batch))
            batch_errors = sum(bool(error) for error in errors)
            print(
                f"Saved predictions {start + 1}-{start + len(batch)} / {len(rows)} "
                f"in {time.time() - batch_start:.1f}s (ok={len(batch) - batch_errors}, errors={batch_errors})",
                flush=True,
            )
            if batch_errors:
                print(f"First error: {next(error for error in errors if error)}", flush=True)

    print(f"Predictions saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
