import argparse
import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import PROJECT_ROOT, path

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, total=None, desc=None, unit=None):
            self.total = total
            self.count = 0
            if desc:
                print(f"{desc}: 0/{total if total is not None else '?'} {unit or ''}".strip())

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.total is not None:
                print(f"Progress: {self.count}/{self.total}")

        def update(self, value):
            self.count += value
            if self.total is not None:
                print(f"Progress: {self.count}/{self.total}", flush=True)


DEFAULT_INPUT = path("paths.test_csv")
DEFAULT_PREDICTIONS = path("paths.evaluation_results") / "qwen3-8b_test_predictions.csv"
DEFAULT_JUDGED = path("paths.evaluation_results") / "qwen3-8b_test_judged.csv"
DEFAULT_SUMMARY = path("paths.evaluation_results") / "qwen3-8b_test_summary.json"


def load_env_file(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_json_response(raw):
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def to_json_text(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if pd.isna(value):
        return ""
    return str(value)


def parse_model_response(raw):
    try:
        parsed = parse_json_response(raw)
        return {
            "answer": parsed.get("answer", ""),
            "reasoning": parsed.get("reasoning", ""),
            "extracted_triples": to_json_text(parsed.get("extracted_triples", [])),
            "parse_warning": "",
        }
    except Exception as exc:
        text = raw or ""
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
        reasoning = think_match.group(1).strip() if think_match else text.strip()
        answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        after_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        answer = answer_match.group(1).strip() if answer_match else after_think
        if not answer:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            answer = lines[-1] if lines else ""
        triples = re.findall(r"\([^()\n]+,\s*[^()\n]+,\s*[^()\n]+\)", text)
        return {
            "answer": answer,
            "reasoning": reasoning,
            "extracted_triples": to_json_text(triples),
            "parse_warning": f"non_json_model_output: {exc}",
        }


def compact(value, limit=3500):
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def log(message):
    print(message, flush=True)


class ChatClient:
    def __init__(self, model, api_key=None, base_url=None, timeout=180):
        api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ZENMUX_API_KEY")
        )
        base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY or ZENMUX_API_KEY in .env.")
        if not base_url:
            raise ValueError("Set OPENAI_BASE_URL or OPENAI_API_BASE in .env.")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = None
        if OpenAI is not None:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _create_with_urllib(self, kwargs):
        url = f"{self.base_url}/chat/completions"
        payload = json.dumps(kwargs).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
        return data["choices"][0]["message"]["content"]

    async def chat(
        self,
        messages,
        temperature=0.0,
        max_tokens=1800,
        top_p=None,
        top_k=None,
        response_format=False,
        max_retries=3,
    ):
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if top_p is not None:
                    kwargs["top_p"] = top_p
                if top_k is not None:
                    kwargs["top_k"] = top_k
                if response_format:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.client is not None:
                    response = await asyncio.to_thread(self.client.chat.completions.create, **kwargs)
                    content = response.choices[0].message.content
                else:
                    content = await asyncio.to_thread(self._create_with_urllib, kwargs)
                if content:
                    return content, ""
                raise RuntimeError("empty_response")
            except Exception as exc:
                if attempt == max_retries - 1:
                    return "", str(exc)
                await asyncio.sleep(5 if attempt == 0 else 20)


def build_model_messages(row):
    task_type = compact(row.get("task_type", ""), 100)
    question = compact(row.get("question", ""), 5000)
    return [
        {
            "role": "system",
            "content": (
                "You are a biomedical multi-hop reasoning QA model. "
                "Answer the user's question using explicit biomedical reasoning. "
                "You may think step by step. End with a concise final answer."
            ),
        },
        {
            "role": "user",
            "content": f"""
Task type: {task_type}

Question:
{question}

Please provide a natural-language reasoning process and a concise final answer.
""".strip(),
        },
    ]


def build_judge_messages(row):
    task_type = compact(row.get("task_type", ""), 100)
    return [
        {
            "role": "system",
            "content": "You are a strict biomedical QA evaluator. Output JSON only.",
        },
        {
            "role": "user",
            "content": f"""
Evaluate a model answer for a biomedical multi-hop QA sample.

Score all numeric fields from 0 to 100.

- answer_correctness: whether the final answer matches the reference answer, allowing harmless synonyms.
- reasoning_triple_alignment: whether the natural-language reasoning follows the same key evidence path as the reference explanation/evidence_KG.
- reasoning_quality: coherence and multi-hop completeness.
- evidence_coverage: how much of the important reference evidence/path is covered by the model response.
- overall_score: preliminary holistic quality; the final reported eval_overall_score is recomputed as a weighted score using all four rubric dimensions.
- pass: true if answer_correctness >= 70, reasoning_triple_alignment >= 70, and the weighted eval_overall_score >= 70.

Return JSON only with this schema:
{{
  "answer_correctness": 0-100,
  "reasoning_triple_alignment": 0-100,
  "reasoning_quality": 0-100,
  "evidence_coverage": 0-100,
  "overall_score": 0-100,
  "pass": true/false,
  "major_issues": ["short issue", "..."],
  "rationale": "brief explanation"
}}

Task type: {task_type}

Question:
{compact(row.get("question", ""), 5000)}

Reference answer:
{compact(row.get("answer", ""), 1000)}

Reference explanation:
{compact(row.get("explanation", ""), 5000)}

Reference evidence_KG:
{compact(row.get("evidence_KG", ""), 5000)}

Model answer:
{compact(row.get("model_answer", ""), 1500)}

Model reasoning:
{compact(row.get("model_reasoning", ""), 5000)}

Full model response:
{compact(row.get("model_raw", ""), 8000)}
""".strip(),
        },
    ]


def stable_row_id(index, row):
    task = "" if pd.isna(row.get("task_type", "")) else str(row.get("task_type", ""))
    for key in ["kg_source_id", "source_kg_index", "target_protein_id", "pathway_id", "pair_key"]:
        value = row.get(key, "")
        if not pd.isna(value) and str(value).strip():
            return f"{task}:{key}:{str(value).strip()}"
    return f"{task}:row:{index}"


def normalize_100_score(value):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return value
    if 0 <= score <= 10:
        score *= 10
    if score.is_integer():
        return int(score)
    return round(score, 2)


EVAL_OVERALL_WEIGHTS = {
    "answer_correctness": 0.40,
    "reasoning_triple_alignment": 0.30,
    "reasoning_quality": 0.20,
    "evidence_coverage": 0.10,
}


def calculate_eval_overall_score(
    answer_correctness,
    reasoning_triple_alignment,
    reasoning_quality=None,
    evidence_coverage=None,
):
    try:
        answer_score = float(answer_correctness)
        alignment_score = float(reasoning_triple_alignment)
        reasoning_score = float(reasoning_quality)
        evidence_score = float(evidence_coverage)
    except (TypeError, ValueError):
        return ""
    score = (
        EVAL_OVERALL_WEIGHTS["answer_correctness"] * answer_score
        + EVAL_OVERALL_WEIGHTS["reasoning_triple_alignment"] * alignment_score
        + EVAL_OVERALL_WEIGHTS["reasoning_quality"] * reasoning_score
        + EVAL_OVERALL_WEIGHTS["evidence_coverage"] * evidence_score
    )
    if score.is_integer():
        return int(score)
    return round(score, 2)


def calculate_eval_pass(answer_correctness, reasoning_triple_alignment, eval_overall_score=None):
    try:
        return (
            float(answer_correctness) >= 70
            and float(reasoning_triple_alignment) >= 70
            and float(eval_overall_score) >= 70
        )
    except (TypeError, ValueError):
        return False


async def generate_predictions(args):
    load_env_file(PROJECT_ROOT / ".env")
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
            dropped = len(existing_df) - int(ok.sum())
            if dropped:
                print(f"Resume: retrying {dropped} previous model_error rows.")
            existing_df = existing_df[ok].copy()
        existing = existing_df.to_dict("records")
        if "eval_row_id" in existing_df.columns:
            done = set(existing_df["eval_row_id"].astype(str))

    rows = [row.to_dict() for _, row in df.iterrows() if str(row.get("eval_row_id", "")) not in done]
    print(f"Input rows: {len(df)}")
    print(f"Existing predictions: {len(existing)}")
    print(f"Rows to predict: {len(rows)}")

    client = ChatClient(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    results = list(existing)
    with tqdm(total=len(rows), desc="Predicting", unit="sample") as pbar:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            batch_start = time.time()
            log(f"Predict batch {start + 1}-{start + len(batch)} / {len(rows)}")
            calls = [
                client.chat(
                    build_model_messages(row),
                    temperature=args.temperature,
                    max_tokens=args.model_max_tokens,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    response_format=args.model_response_format,
                )
                for row in batch
            ]
            responses = await asyncio.gather(*calls)
            batch_errors = []
            for row, (raw, error) in zip(batch, responses):
                merged = dict(row)
                merged["model_name"] = args.model
                merged["eval_model_temperature"] = args.temperature
                merged["eval_model_max_tokens"] = args.model_max_tokens
                merged["eval_model_top_p"] = args.top_p
                merged["eval_model_top_k"] = args.top_k
                merged["eval_batch_size"] = args.batch_size
                merged["eval_timeout"] = args.timeout
                merged["model_raw"] = raw
                merged["model_error"] = error
                if error:
                    batch_errors.append(error)
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
            log(
                f"Saved predictions after batch in {time.time() - batch_start:.1f}s "
                f"(ok={len(batch) - len(batch_errors)}, errors={len(batch_errors)})"
            )
            if batch_errors:
                log(f"First prediction error: {batch_errors[0]}")

    print(f"Predictions saved to: {output_path}")


async def judge_predictions(args):
    load_env_file(PROJECT_ROOT / ".env")
    input_path = Path(args.predictions)
    output_path = Path(args.judged)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, dtype=str)
    if args.max_rows:
        df = df.head(args.max_rows).copy()

    existing = []
    done = set()
    if args.resume and output_path.exists():
        existing_df = pd.read_csv(output_path, dtype=str)
        if "eval_judge_error" in existing_df.columns:
            ok = existing_df["eval_judge_error"].fillna("").eq("")
            dropped = len(existing_df) - int(ok.sum())
            if dropped:
                print(f"Resume: retrying {dropped} previous eval_judge_error rows.")
            existing_df = existing_df[ok].copy()
        existing = existing_df.to_dict("records")
        if "eval_row_id" in existing_df.columns:
            done = set(existing_df["eval_row_id"].astype(str))

    rows = []
    for _, row in df.iterrows():
        if str(row.get("eval_row_id", "")) in done:
            continue
        if str(row.get("model_error", "")).strip() and str(row.get("model_error", "")).lower() != "nan":
            merged = row.to_dict()
            merged["eval_judge_error"] = "skipped_due_to_model_error"
            rows.append(merged)
        else:
            rows.append(row.to_dict())

    print(f"Prediction rows: {len(df)}")
    print(f"Existing judged rows: {len(existing)}")
    print(f"Rows to judge: {len(rows)}")

    judge_model = args.judge_model or os.getenv("JUDGE_MODEL") or os.getenv("OPENAI_MODEL")
    client = ChatClient(
        model=judge_model,
        api_key=args.judge_api_key or os.getenv("JUDGE_API_KEY") or args.api_key,
        base_url=args.judge_base_url or os.getenv("JUDGE_BASE_URL") or args.base_url,
        timeout=args.timeout,
    )

    results = list(existing)
    with tqdm(total=len(rows), desc="Judging", unit="sample") as pbar:
        for start in range(0, len(rows), args.judge_batch_size):
            batch = rows[start : start + args.judge_batch_size]
            batch_start = time.time()
            log(f"Judge batch {start + 1}-{start + len(batch)} / {len(rows)}")
            calls = []
            call_rows = []
            for row in batch:
                if row.get("eval_judge_error") == "skipped_due_to_model_error":
                    results.append(row)
                    continue
                calls.append(
                    client.chat(
                        build_judge_messages(row),
                        temperature=0.0,
                        max_tokens=args.judge_max_tokens,
                        response_format=args.judge_response_format,
                    )
                )
                call_rows.append(row)
            responses = await asyncio.gather(*calls) if calls else []
            batch_errors = []
            for row, (raw, error) in zip(call_rows, responses):
                merged = dict(row)
                merged["eval_judge_model"] = judge_model
                merged["eval_judge_max_tokens"] = args.judge_max_tokens
                merged["eval_judge_batch_size"] = args.judge_batch_size
                merged["eval_judge_raw"] = raw
                merged["eval_judge_error"] = error
                if error:
                    batch_errors.append(error)
                if raw and not error:
                    try:
                        parsed = parse_json_response(raw)
                        answer_correctness = normalize_100_score(parsed.get("answer_correctness"))
                        reasoning_triple_alignment = normalize_100_score(parsed.get("reasoning_triple_alignment"))
                        reasoning_quality = normalize_100_score(parsed.get("reasoning_quality"))
                        evidence_coverage = normalize_100_score(
                            parsed.get("evidence_coverage", parsed.get("extracted_triple_quality"))
                        )
                        eval_overall_score = calculate_eval_overall_score(
                            answer_correctness,
                            reasoning_triple_alignment,
                            reasoning_quality,
                            evidence_coverage,
                        )
                        merged.update(
                            {
                                "answer_correctness": answer_correctness,
                                "reasoning_triple_alignment": reasoning_triple_alignment,
                                "reasoning_quality": reasoning_quality,
                                "evidence_coverage": evidence_coverage,
                                "judge_overall_score": normalize_100_score(parsed.get("overall_score")),
                                "judge_pass": parsed.get("pass"),
                                "eval_overall_score": eval_overall_score,
                                "eval_pass": calculate_eval_pass(
                                    answer_correctness,
                                    reasoning_triple_alignment,
                                    eval_overall_score,
                                ),
                                "eval_major_issues": json.dumps(parsed.get("major_issues", []), ensure_ascii=False),
                                "eval_rationale": parsed.get("rationale", ""),
                            }
                        )
                    except Exception as exc:
                        merged["eval_judge_error"] = f"parse_error: {exc}"
                        batch_errors.append(merged["eval_judge_error"])
                results.append(merged)
            pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8-sig")
            pbar.update(len(batch))
            skipped = len(batch) - len(call_rows)
            log(
                f"Saved judged rows after batch in {time.time() - batch_start:.1f}s "
                f"(judged_ok={len(call_rows) - len(batch_errors)}, errors={len(batch_errors)}, skipped={skipped})"
            )
            if batch_errors:
                log(f"First judge error: {batch_errors[0]}")

    print(f"Judged predictions saved to: {output_path}")
    write_summary(output_path, Path(args.summary))


def write_summary(judged_path, summary_path):
    df = pd.read_csv(judged_path, dtype=str)
    numeric_cols = [
        "answer_correctness",
        "reasoning_triple_alignment",
        "reasoning_quality",
        "evidence_coverage",
        "eval_overall_score",
    ]
    if "evidence_coverage" not in df.columns and "extracted_triple_quality" in df.columns:
        df["evidence_coverage"] = df["extracted_triple_quality"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if set(EVAL_OVERALL_WEIGHTS).issubset(df.columns):
        df["eval_overall_score"] = (
            EVAL_OVERALL_WEIGHTS["answer_correctness"] * df["answer_correctness"]
            + EVAL_OVERALL_WEIGHTS["reasoning_triple_alignment"] * df["reasoning_triple_alignment"]
            + EVAL_OVERALL_WEIGHTS["reasoning_quality"] * df["reasoning_quality"]
            + EVAL_OVERALL_WEIGHTS["evidence_coverage"] * df["evidence_coverage"]
        ).round(4)
        df["eval_pass"] = (
            (df["answer_correctness"] >= 70)
            & (df["reasoning_triple_alignment"] >= 70)
            & (df["eval_overall_score"] >= 70)
        )
    if "eval_pass" in df.columns:
        pass_mask = df["eval_pass"].astype(str).str.lower().isin({"true", "1", "yes"})
    else:
        pass_mask = pd.Series([False] * len(df))

    summary = {
        "rows": int(len(df)),
        "judge_error_rows": int(df.get("eval_judge_error", pd.Series([""] * len(df))).fillna("").ne("").sum()),
        "pass_rows": int(pass_mask.sum()),
        "pass_rate": round(float(pass_mask.mean()), 4) if len(df) else 0.0,
        "eval_overall_weights": EVAL_OVERALL_WEIGHTS,
        "pass_rule": "answer_correctness >= 70 and reasoning_triple_alignment >= 70 and weighted eval_overall_score >= 70",
        "overall": {},
        "by_task": {},
    }
    for col in numeric_cols:
        if col in df.columns:
            summary["overall"][col] = {
                "mean": round(float(df[col].mean()), 4) if df[col].notna().any() else None,
                "median": round(float(df[col].median()), 4) if df[col].notna().any() else None,
            }
    if "task_type" in df.columns:
        for task, sub in df.groupby("task_type", dropna=False):
            sub_pass = sub["eval_pass"].astype(str).str.lower().isin({"true", "1", "yes"}) if "eval_pass" in sub else pd.Series([False] * len(sub))
            item = {
                "rows": int(len(sub)),
                "pass_rows": int(sub_pass.sum()),
                "pass_rate": round(float(sub_pass.mean()), 4) if len(sub) else 0.0,
            }
            for col in numeric_cols:
                if col in sub.columns:
                    item[col + "_mean"] = round(float(sub[col].mean()), 4) if sub[col].notna().any() else None
            summary["by_task"][str(task)] = item

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    load_env_file(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-8B on KG QA data and judge answer/reasoning/triple alignment.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV, usually outputs/final_datasets/splits/test.csv.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS), help="Output CSV for model predictions.")
    parser.add_argument("--judged", default=str(DEFAULT_JUDGED), help="Output CSV for judged predictions.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Output JSON summary.")
    parser.add_argument("--model", default=os.getenv("EVAL_MODEL", "Qwen3-8B"), help="Model under evaluation.")
    parser.add_argument("--judge-model", default=None, help="LLM judge model. Defaults to JUDGE_MODEL/OPENAI_MODEL.")
    parser.add_argument("--api-key", default=None, help="API key for evaluated model. Defaults to OPENAI_API_KEY/ZENMUX_API_KEY.")
    parser.add_argument("--base-url", default=None, help="Base URL for evaluated model. Defaults to OPENAI_BASE_URL/OPENAI_API_BASE.")
    parser.add_argument("--judge-api-key", default=None, help="API key for judge. Defaults to JUDGE_API_KEY or evaluated API key.")
    parser.add_argument("--judge-base-url", default=None, help="Base URL for judge. Defaults to JUDGE_BASE_URL or evaluated base URL.")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional small subset for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=2, help="Concurrent evaluated-model requests.")
    parser.add_argument("--judge-batch-size", type=int, default=2, help="Concurrent judge requests.")
    parser.add_argument("--model-max-tokens", type=int, default=int(os.getenv("EVAL_MAX_TOKENS", "4096")))
    parser.add_argument("--judge-max-tokens", type=int, default=int(os.getenv("JUDGE_MAX_TOKENS", "1800")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("EVAL_TEMPERATURE", "0.0")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("EVAL_TOP_P", "0.95")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("EVAL_TOP_K", "20")))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("REQUEST_TIMEOUT", "180")))
    parser.add_argument("--resume", action="store_true", help="Resume from existing outputs.")
    parser.add_argument("--generate-only", action="store_true", help="Only call Qwen3-8B and save predictions.")
    parser.add_argument("--judge-only", action="store_true", help="Only judge an existing predictions CSV.")
    parser.add_argument("--model-response-format", action="store_true", help="Send OpenAI JSON response_format to evaluated model.")
    parser.add_argument("--judge-response-format", action="store_true", help="Send OpenAI JSON response_format to judge model.")
    parser.add_argument("--summary-only", action="store_true", help="Only summarize an existing judged CSV.")
    return parser.parse_args()


async def main():
    args = parse_args()
    if args.summary_only:
        write_summary(Path(args.judged), Path(args.summary))
        return
    if not args.judge_only:
        await generate_predictions(args)
    if not args.generate_only:
        await judge_predictions(args)


if __name__ == "__main__":
    asyncio.run(main())
