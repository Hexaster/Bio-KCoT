import json
import re
import asyncio
import os
import openai
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import env, get, path


# -----------------------------------------------------------------------------
# 1. 你的 OpenAIChat 类 (保持原样，略作适配)
# -----------------------------------------------------------------------------
class OpenAIChat():
    def __init__(
            self,
            model_name='gpt-4o-mini',
            max_tokens=2500,
            temperature=0.7,  # 稍微调高一点温度，让思考过程更自然、多样化
            top_p=1,
            request_timeout=180,
            stop=None,
            response_format='json_object',  # 强制 JSON 输出
            logprobs=False,
            top_logprobs=None,
            n=1,
            api_key=None,
            api_base=None
    ):
        self.config = {
            'model_name': model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'request_timeout': request_timeout,
            'stop': stop,
            'response_format': response_format,
            'logprobs': logprobs,
            'top_logprobs': top_logprobs,
            'sample_n': n,
        }

        api_key = api_key or env("OPENAI_API_KEY") or env("JUDGE_API_KEY", required=True)
        api_base = api_base or env("OPENAI_BASE_URL", get("api.judge_base_url"))
        if "gpt" in model_name or 'embedding' in model_name or 'deepseek' in model_name:
            openai.api_key = api_key
            if api_base:
                openai.api_base = api_base
        else:
            openai.api_key = "EMPTY"
            openai.api_base = env("LOCAL_OPENAI_BASE_URL", get("api.local_base_url"))

    async def dispatch_openai_requests(self, messages_list) -> List[str]:
        async def _request_with_retry(messages, retry=3):
            for try_i in range(retry):
                try:
                    kwargs = {
                        'model': self.config['model_name'],
                        'messages': messages,
                        'max_tokens': self.config['max_tokens'],
                        'temperature': self.config['temperature'],
                        'top_p': self.config['top_p'],
                        'request_timeout': self.config['request_timeout'],
                        'stop': self.config['stop'],
                        'n': self.config['sample_n'],
                    }
                    if self.config['response_format']:
                        kwargs['response_format'] = {'type': self.config['response_format']}

                    # 注意：这是 openai < 1.0.0 的写法，如果你用的是新版 sdk，请改为 client.chat.completions.create
                    response = await openai.ChatCompletion.acreate(**kwargs)
                    return response

                except Exception as e:
                    print(f"Error: {e}, retrying...")
                    await asyncio.sleep(2)
            return None

        async_responses = [_request_with_retry(messages) for messages in messages_list]
        return await asyncio.gather(*async_responses)

    async def async_run(self, messages_list):
        # 简化版 run，直接调用 dispatch
        predictions = await self.dispatch_openai_requests(messages_list)
        # 提取 content
        results = []
        for pred in predictions:
            if pred and 'choices' in pred:
                results.append(pred['choices'][0]['message']['content'])
            else:
                results.append(None)
        return results


# -----------------------------------------------------------------------------
# 2. 实时流式清洗逻辑
# -----------------------------------------------------------------------------

INPUT_FILE = str(path("paths.train_data_dir") / "sft_rl_aggressive_split" / "sft_KG_data.json")
# 中间临时文件 (JSONL格式：每行一个json对象，方便追加)
TEMP_FILE = str(path("paths.train_data_dir") / "sft_rl_aggressive_split" / "sft_KG_data_processing.jsonl")
# 最终输出文件 (标准的 JSON Array 格式)
FINAL_FILE = str(path("paths.train_data_dir") / "sft_rl_aggressive_split" / "sft_KG_data_cleaned.json")

THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)
ANSWER_PATTERN = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)


def construct_cleaning_prompt(original_think):
    return f"""
You are an expert biological reasoning assistant. Rewrite the "Thinking Process" to be natural, hesitant, and first-person.

### STRICT CONSTRAINTS:
1. **REMOVE METADATA**: No UniProt IDs, no "knowledge graph", no "evidence mentioned", no "explanation provided".
2. **NATURAL STYLE**: Don't list all the knowledge points at once. Instead, simulate the natural reasoning process.
3. **KEEP LOGIC**: Keep the biological reasoning correct but make it sound like internal monologue.
4. **Maintain professionalism**: When explaining the biological mechanisms, one should maintain the same level of professionalism as in the original text and must not omit any details.

### OUTPUT FORMAT (JSON ONLY):
{{
    "cleaned_thought": "Your rewritten thought here..."
}}

### ONE-SHOT EXAMPLE (Learn from this):
**Original Text**:
"Okay, let's tackle this question step by step. So, the patient has seasonal allergic conjunctivitis with all the classic symptoms: itching, redness, tearing, and swollen eyelids due to allergens. The doctor needs to pick an anti-inflammatory eye drop, but there's a catch—the patient's had issues with steroids increasing their eye pressure before. So, the main challenge here is choosing a corticosteroid that's effective for the inflammation but less likely to spike intraocular pressure (IOP).\n\nFirst, I need to recall which corticosteroids are used in ophthalmology and their side effect profiles. Common ones include dexamethasone, prednisolone, fluorometholone, loteprednol, and medrysone. But I'm not exactly sure which ones are safer regarding IOP. From what I remember, some steroids are considered \"soft\" or have lower risk because they’re metabolized quickly in the eye or have different potencies.\n\nDexamethasone is a potent steroid but is known to cause significant IOP elevation in some patients. Prednisolone is also quite potent and might have similar issues. Then there's loteprednol, which I think was designed to have a lower risk of IOP spikes because of its ester-based structure that allows it to be metabolized more easily. Fluorometholone is another one that's sometimes mentioned as having less effect on IOP. Medrysone... I'm a bit fuzzy on that. Maybe it's less commonly used but has a specific indication here.\n\nThe question is asking for the most appropriate one. The explanation mentioned medrysone. But why? Let me think about the mechanism. Corticosteroids in general work by binding to glucocorticoid receptors, which then leads to the production of lipocortins. These inhibit phospholipase A2, which is involved in the arachidonic acid pathway. Blocking that reduces prostaglandins and leukotrienes, which are inflammatory mediators. So, the mechanism is standard for steroids, but the key here is the side effect profile.\n\nSo, the key point is that medrysone has a lower propensity to raise IOP compared to others. Maybe because it's less potent? Or perhaps it's metabolized in a way that doesn't affect the trabecular meshwork as much. Loteprednol, for instance, is a lipid ester prodrug that's activated in the eye but has rapid deactivation, which might reduce side effects. But the correct answer here is medrysone. I need to confirm if medrysone is indeed the one with the lower risk. Maybe it's a less potent steroid, so while it's effective for allergic conjunctivitis, it doesn't have as strong an effect on IOP. \n\nAlternatively, maybe medrysone has a different chemical structure that doesn't penetrate certain parts of the eye as much, thus avoiding the cells that regulate IOP. Another angle: some steroids are more selective in their action. For example, those with higher mineralocorticoid activity might have different effects, but I think the main issue with IOP is glucocorticoid receptor-mediated.\n\nIn summary, the clinician needs a steroid that's effective against inflammation but with minimal IOP effect. Among the options, medrysone is the answer, likely due to its lower potency or specific formulation that reduces this risk. The general mechanism is the standard corticosteroid pathway of inhibiting phospholipase A2 via lipocortin, thereby reducing inflammatory mediators.\n"

**Rewritten Natural Thought**:
""Okay, let's analyze this patient's case. We're dealing with seasonal allergic conjunctivitis—itching, redness, the works. Standard treatment would be a topical corticosteroid. However, there's a critical constraint here: the patient has a history of steroid-induced intraocular pressure (IOP) elevation. This makes the choice tricky. I need an agent that calms the inflammation but won't spike the pressure./nLet me run through the standard ophthalmic steroids. Dexamethasone? Definitely not. It's highly potent and notorious for raising IOP. Prednisolone acetate is similar; too risky for this specific patient.\nWhat about the 'safer' options? Loteprednol comes to mind first—it's an ester-based 'soft' steroid designed to deactivate quickly. That's a strong candidate./nBut wait... let me dig a bit deeper. Is there something even more specific for surface issues with minimal penetration?/nMedrysone. Right, I recall Medrysone. It's structurally related to progesterone and has significantly lower potency compared to dexamethasone. Because it doesn't penetrate the eye as well, it rarely reaches the trabecular meshwork in high enough concentrations to cause that IOP spike./nSo, while Loteprednol is safe, Medrysone is classically cited as having one of the lowest risks for IOP elevation, making it ideal for simple allergic conjunctivitis where deep penetration isn't needed./nAs for the mechanism... does being 'weaker' change how it works? No. It still follows the standard pathway: binding to the intracellular glucocorticoid receptors, translocating to the nucleus, and inducing the synthesis of lipocortins (annexins). These lipocortins specifically inhibit phospholipase A2. By blocking phospholipase A2, the entire arachidonic acid cascade is halted—preventing the release of inflammatory prostaglandins and leukotrienes.\nSo, putting it all together: Medrysone is the best fit here due to its limited intraocular penetration and safety profile."

### YOUR TASK:
Rewrite the following input text following the natural style above.

### ORIGINAL TEXT:
{original_think}
"""


async def main():
    # ---------------- 配置 API ----------------

    chat = OpenAIChat(
        model_name='deepseek-r1',
        response_format='json_object'
    )
    # -----------------------------------------

    # 1. 读取原始数据
    print(f"Loading raw data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # data = data[:10]

    total_items = len(data)

    # 2. 检查断点：读取临时文件，看处理了多少条
    processed_count = 0
    if os.path.exists(TEMP_FILE):
        with open(TEMP_FILE, 'r', encoding='utf-8') as f:
            for _ in f:
                processed_count += 1
        print(f"Found existing progress: {processed_count} items already processed.")

    # 3. 打开临时文件准备追加写入 ('a' 模式)
    # 使用 buffer=1 并不是真正的无缓冲，但在 python 中 flush 更重要
    f_out = open(TEMP_FILE, 'a', encoding='utf-8')

    # 4. 批处理循环
    BATCH_SIZE = 10  # 每次并发处理 10 条

    # 从 processed_count 开始继续处理
    for i in range(processed_count, total_items, BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, total_items)
        batch_indices = list(range(i, batch_end))
        batch_data = [data[idx] for idx in batch_indices]

        print(f"Processing items {i} to {batch_end}...")

        messages_batch = []
        valid_indices_in_batch = []  # 记录这个batch里哪些是真正需要调API的

        # 准备 Prompt
        for local_idx, item in enumerate(batch_data):
            original_output = item.get('output', '')
            think_match = THINK_PATTERN.search(original_output)

            if think_match:
                original_think = think_match.group(1).strip()
                prompt_content = construct_cleaning_prompt(original_think)
                messages = [
                    {"role": "system", "content": "You are a rewriting assistant."},
                    {"role": "user", "content": prompt_content}
                ]
                messages_batch.append(messages)
                valid_indices_in_batch.append(local_idx)
            else:
                # 如果没有 think 标签，这个 item 就不需要调 API，但需要写入文件
                pass

        # 并发调用 API
        batch_responses = []
        if messages_batch:
            batch_responses = await chat.async_run(messages_batch)

        # 处理结果并立即写入文件
        response_idx = 0

        for local_idx, item in enumerate(batch_data):
            # item 是字典引用，修改它会影响内存中的数据
            # 如果这个 item 发送了请求
            if local_idx in valid_indices_in_batch:
                res_text = batch_responses[response_idx]
                response_idx += 1

                if res_text:
                    try:
                        res_json = json.loads(res_text)
                        cleaned_think = res_json.get("cleaned_thought", "")

                        if cleaned_think:
                            # 重新组装 output
                            original_full_output = item['output']
                            answer_match = ANSWER_PATTERN.search(original_full_output)
                            answer_part = answer_match.group(0) if answer_match else ""

                            # 更新 item 数据
                            item['output'] = f"<think>\n{cleaned_think}\n</think>\n{answer_part}"
                    except:
                        print(f"Error parsing JSON for item index {i + local_idx}, keeping original.")

            # --- 关键步骤：写入一行 JSON 到临时文件 ---
            # ensure_ascii=False 保证中文正常显示
            json_line = json.dumps(item, ensure_ascii=False)
            f_out.write(json_line + "\n")

        # --- 关键步骤：刷新缓冲区，确保写入硬盘 ---
        f_out.flush()
        print(f"Saved batch {i} to {batch_end} to {TEMP_FILE}")

    f_out.close()
    print("All items processed.")

    # 5. 最后转换：将 JSONL 转回标准的 JSON Array 格式
    # 如果你的下游任务支持 jsonl，这一步可以省略
    print(f"Converting {TEMP_FILE} to final JSON format: {FINAL_FILE}...")
    final_data = []
    with open(TEMP_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                final_data.append(json.loads(line))

    with open(FINAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    print("Conversion complete. Task finished!")


if __name__ == "__main__":
    asyncio.run(main())
