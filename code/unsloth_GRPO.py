import os
import re
import json
import concurrent.futures
import sys
from pathlib import Path
from openai import OpenAI
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import env, get, path

# ==========================================
# 1. 配置区域
# ==========================================
JUDGE_API_KEY = env("JUDGE_API_KEY", required=True)
JUDGE_BASE_URL = env("JUDGE_BASE_URL", get("api.judge_base_url"))
JUDGE_MODEL = env("JUDGE_MODEL", get("api.judge_model"))

client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)

# ==========================================
# 2. 模型加载
# ==========================================
PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 4610  # 适当调大，因为包含思维链和知识图谱证据
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(path("paths.sft_kg_merged", env="SFT_KG_MERGED")),
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    load_in_8bit=False,
    max_lora_rank=lora_rank,
    fast_inference=True,
    gpu_memory_utilization=0.7,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# ==========================================
# 3. Reward 函数 (核心修改)
# ==========================================

def format_reward(completions, **kwargs):
    """
    格式检查：必须包含 <think>...</think><answer>...</answer>
    """
    rewards = []
    # 允许中间有空白字符
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"

    for completion in completions:
        # completion 是 list[dict] (新版 TRL) 或 str，这里做个兼容
        content = completion[0]["content"] if isinstance(completion, list) else completion

        if re.search(pattern, content, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def _call_judge_api(question, answer, prediction, evidence_graph):
    """
    判题逻辑：
    - question: 原始问题
    - answer: 地面真值 (Ground Truth)
    - evidence_graph: 知识图谱证据 (可用于辅助判断，或者检查模型是否用到了这些证据)
    """
    if prediction is None:
        return 0.0

    # 构造给裁判的 Prompt
    # 这里我把 evidence_graph 也加进去，让裁判判断模型回答是否符合事实逻辑
    prompt = f"""
    You are an objective judge. Check if the model's extraction matches the ground truth answer.

    [Question]:
    {question}

    [Ground Truth Answer]:
    {answer}

    [Model Extraction]:
    {prediction}

    Is the model's answer correct? It doesn't have to be exactly the same. Roughly similar will do.
    Return only "[[TRUE]]" or "[[FALSE]]".
    """
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        content = response.choices[0].message.content
        print(content)
        if "[[TRUE]]" in content:
            return 5.0
        return 0.0
    except Exception as e:
        print(f"Judge API Error: {e}")
        return 0.0


# ★★★ 修改重点：参数名必须与数据集列名一致 (answer, evidence_graph) ★★★
def accuracy_reward(completions, answer, evidence_graph, **kwargs):
    """
    - answer: 对应数据集里的 'answer' 列
    - evidence_graph: 对应数据集里的 'evidence_graph' 列
    """
    # 1. 获取问题 (TRL 可能会把原始 inputs 里的 prompt 解析出来，
    # 但比较麻烦，我们可以直接尝试从 kwargs 获取 prompt 里的 user content，或者简单起见只对比答案)
    # 这里为了方便，我们假设 prompt 的最后一句话包含了问题，或者我们不传问题给 judge，只传 answer

    # 尝试从 kwargs 里的 prompt 解析 question（如果 dataset 里保留了 question 列最好）
    questions = kwargs.get('question', [])
    if not questions:
        questions = ["(Question not provided)"] * len(completions)

    # 2. 提取模型回答
    predictions = []
    for completion in completions:
        content = completion[0]["content"] if isinstance(completion, list) else completion
        match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if match:
            predictions.append(match.group(1).strip())
        else:
            predictions.append(None)

    # 3. 并发判题
    rewards = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        # 将 answer 和 evidence_graph 传入
        for q, ans, pred, ev_graph in zip(questions, answer, predictions, evidence_graph):
            futures.append(executor.submit(_call_judge_api, q, ans, pred, ev_graph))

        for future in futures:
            rewards.append(future.result())

    return rewards


# ==========================================
# 4. 数据集加载 (直接读取 JSONL)
# ==========================================

# 你的转换代码生成的路径
dataset_path = str(path("paths.rl_dataset", env="RL_DATASET"))

# 直接加载 JSONL
dataset = load_dataset("json", data_files=dataset_path, split="train")

print(f"Dataset loaded. Size: {len(dataset)}")
print("Columns:", dataset.column_names)
# 预期输出: ['prompt', 'answer', 'evidence_graph', 'question']

# ==========================================
# 5. 训练参数配置
# ==========================================

vllm_sampling_params = SamplingParams(
    min_p=0.1,
    top_p=1.0,
    top_k=-1,
    seed=3407,
    stop=["<|endoftext|>", tokenizer.eos_token],
    include_stop_str_in_output=True,
)

training_args = GRPOConfig(
    vllm_sampling_params=vllm_sampling_params,
    temperature=1.0,
    learning_rate=5e-6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_generations=8,
    max_prompt_length=512,  # 输入 (Prompt) 最大长度
    max_completion_length=4096,  # 输出 (Think + Answer) 最大长度
    save_steps = 10,
    save_total_limit = 3,
    save_strategy="steps",
    report_to="none",
    output_dir=str(path("paths.grpo_nokg_output", env="GRPO_NOKG_OUTPUT")),

    # ★★★ 关键设置 ★★★
    # 设置为 False，确保 trainer 不会删掉 'answer' 和 'evidence_graph' 列
    # 这样它们才能被传递给 reward function
    remove_unused_columns=False,
    gradient_checkpointing=True
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=dataset,
)

checkpoint_path = str(path("paths.grpo_nokg_resume", env="GRPO_NOKG_RESUME"))
# 开始训练
trainer.train(resume_from_checkpoint=checkpoint_path)

# 保存
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
