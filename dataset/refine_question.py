import pandas as pd
import json
import asyncio
import openai
from typing import List
from tqdm import tqdm
import  os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import env, get, path

# ---------------------------------------------------------
# 复用你提供的 OpenAIChat 类 (此处省略，保持原样即可)
# ---------------------------------------------------------
class OpenAIChat():
    # more details on: https://platform.openai.com/docs/api-reference/chat
    def __init__(
            self,
            model_name='gpt-4o-mini',  # 推荐使用支持 json mode 的模型
            max_tokens=2500,
            temperature=0.5,
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
        # 配置 API Key 和 Base URL
        if "gpt" in model_name or 'embedding' in model_name or 'deepseek' in model_name:
            openai.api_key = api_key
            if api_base:
                openai.api_base = api_base
        else:
            openai.api_key = "EMPTY"
            openai.api_base = env("LOCAL_OPENAI_BASE_URL", get("api.local_base_url"))

    async def dispatch_openai_requests(
            self,
            messages_list,
    ) -> List[str]:
        async def _request_with_retry(messages, retry=3):
            for try_i in range(retry):
                try:
                    if "embedding" in self.config['model_name']:
                        response = await openai.Embedding.acreate(
                            model=self.config['model_name'],
                            input=messages,
                        )
                    else:
                        response = await openai.ChatCompletion.acreate(
                            model=self.config['model_name'],
                            response_format={'type': self.config['response_format']},
                            messages=messages,
                            max_tokens=self.config['max_tokens'],
                            temperature=self.config['temperature'],
                            top_p=self.config['top_p'],
                            request_timeout=self.config['request_timeout'],
                            stop=self.config['stop'],
                            # logprobs=self.config['logprobs'],
                            top_logprobs=self.config['top_logprobs'],
                            n=self.config['sample_n'],
                        )
                    return response

                except openai.error.InvalidRequestError as e:
                    print(f"Invalid Request: {e}")
                    # 不重试致命错误，直接返回 None
                    return None
                except (openai.error.RateLimitError, openai.error.APIError,
                        openai.error.Timeout, openai.error.APIConnectionError,
                        openai.error.ServiceUnavailableError) as e:
                    print(f'Retry {try_i + 1} Error: {e}, waiting...')
                    await asyncio.sleep(5 if try_i < 1 else 20)  # 简单的退避策略
                except Exception as e:
                    print(f"Unknown Error: {e}")
                    await asyncio.sleep(5)
            return None

        async_responses = [
            _request_with_retry(messages)
            for messages in messages_list
        ]

        return await asyncio.gather(*async_responses)

    async def async_run(self, messages_list, expected_type='json'):
        retry = 10
        responses = [None for _ in range(len(messages_list))]
        messages_list_cur_index = [i for i in range(len(messages_list))]

        while retry > 0 and len(messages_list_cur_index) > 0:
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]

            predictions = await self.dispatch_openai_requests(
                messages_list=messages_list_cur,
            )

            if "embedding" in self.config['model_name']:
                preds = [prediction['data'][0]['embedding'] if prediction is not None else None for prediction in
                         predictions]
            else:
                if self.config['logprobs'] == False:
                    preds = [prediction['choices'][0]['message']['content'] if prediction is not None else None for
                             prediction in predictions]
                else:
                    preds = [
                        [
                            prediction['choices'][0]['message']['content'],
                            [d['logprob'] for d in prediction['choices'][0]['logprobs']['content']]
                        ] if prediction is not None else None for prediction in predictions
                    ]

            finised_index = []
            for i, pred in enumerate(preds):
                if pred is not None:
                    responses[messages_list_cur_index[i]] = pred
                    finised_index.append(messages_list_cur_index[i])

            messages_list_cur_index = [i for i in messages_list_cur_index if i not in finised_index]
            retry -= 1

        return responses

# =========================================================
# 1. OpenAIChat API 类 (保持你原有的逻辑不变)
# =========================================================
class OpenAIChat():
    def __init__(
            self,
            model_name='gpt-4o',  # 建议使用 GPT-4o 或 DeepSeek-V3 等强逻辑模型
            max_tokens=2500,
            temperature=0.3,  # 温度设低，保证合成的严谨性
            top_p=1,
            request_timeout=180,
            stop=None,
            response_format='json_object',  # 强制 JSON
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
                    response = await openai.ChatCompletion.acreate(
                        model=self.config['model_name'],
                        response_format={'type': self.config['response_format']},
                        messages=messages,
                        max_tokens=self.config['max_tokens'],
                        temperature=self.config['temperature'],
                        top_p=self.config['top_p'],
                        request_timeout=self.config['request_timeout'],
                        stop=self.config['stop'],
                        n=self.config['sample_n'],
                    )
                    return response
                except Exception as e:
                    print(f"Error: {e}, waiting...")
                    await asyncio.sleep(5)
            return None

        async_responses = [_request_with_retry(messages) for messages in messages_list]
        return await asyncio.gather(*async_responses)

    async def async_run(self, messages_list, expected_type='json'):
        # 简化版 async run
        predictions = await self.dispatch_openai_requests(messages_list)
        preds = [p['choices'][0]['message']['content'] if p else None for p in predictions]
        return preds


# =========================================================
# 2. 核心逻辑：数据融合与 Prompt 构建
# =========================================================

# 文件路径配置
FILE_PATH_1 = str(path("paths.data_dir") / "disease-indication-drug.csv")
FILE_PATH_2 = str(path("paths.data_dir") / "disease-indication-drug2.csv")
OUTPUT_PATH = str(path("paths.data_dir") / "disease-indication-drug_synthesized.csv")
EXISTING_DATA_PATH = OUTPUT_PATH
NEW_OUTPUT_PATH = str(path("paths.data_dir") / "disease-indication-drug_synthesized_part2.csv")


def load_and_align_data():
    """
    读取两个 CSV，并基于 source (三元组) 或 kg_source_id 进行对齐合并
    """
    print("Loading datasets...")
    try:
        df1 = pd.read_csv(FILE_PATH_1)
        df2 = pd.read_csv(FILE_PATH_2)
    except FileNotFoundError as e:
        print(f"Error finding files: {e}")
        return None

    print(f"Dataset 1 shape: {df1.shape}")
    print(f"Dataset 2 shape: {df2.shape}")

    # 这里的 'source' 列应该是 "(head, relation, tail)" 这种唯一标识
    # 我们使用 inner merge 确保只处理两个文件里都有的数据
    # suffixes=('_v1', '_v2') 会自动给重名列加上后缀
    merge_key = 'source' if 'source' in df1.columns else 'kg_source_id'

    merged_df = pd.merge(df1, df2, on=merge_key, suffixes=('_v1', '_v2'), how='inner')

    print(f"Merged aligned dataset shape: {merged_df.shape}")
    return merged_df


def construct_synthesis_prompt(row):
    """
    构建让 LLM 充当'医学主编'的 Prompt
    """
    # 提取 Drug Name (假设两个版本答案一致，取其一，或者取 source 里的)
    # 尝试从 answer_v1 获取，如果为空则取 answer_v2
    drug_name = str(row.get('answer_v1', ''))
    if not drug_name or drug_name == 'nan':
        drug_name = str(row.get('answer_v2', ''))

    source_triplet = row.get('source', 'Unknown Triplet')

    prompt = f"""
You are an expert Biomedical Dataset Editor. 

Your task is to synthesize a **Golden Standard** QA pair by comparing two draft versions generated by different models.
The QA pair is about Drug Indication and Mechanism of Action.

Target Context: {source_triplet}
Target Drug: {drug_name}

### Draft Version 1:
Question: {row.get('question_v1')}
Reasoning: {row.get('explanation_v1')}
KG Evidence: {row.get('evidence_KG_v1')}

### Draft Version 2:
Question: {row.get('question_v2')}
Reasoning: {row.get('explanation_v2')}
KG Evidence: {row.get('evidence_KG_v2')}

1. **Check Clinical Accuracy**
   - Verify the correct **route of administration**, **drug class**, and **disease indication**.
   - Eliminate any hallucinated facts or incorrect associations.

2. **Enforce Minimal but Sufficient Information**
   - The QUESTION should contain **only high-level phenotypic or clinical clues**.
   - Do NOT reveal the exact molecular target, receptor, or enzyme if that is central to the reasoning.
   - The question must be **uniquely solvable**, even with minimal information.

3. **Prevent Information Leakage**
   - Do NOT name the exact target, receptor, or pathway in the question.
   - Instead, describe the **pathological defect, clinical phenotype, or system-level dysfunction**.

4. **Build a Stepwise Biological Reasoning Chain**
   - The EXPLANATION must explicitly reason step-by-step, specifically describe the mechanism of action of biological molecules.
   
5. Merge the best logic from V1 and V2.
   
### Output (JSON):
{{
  "final_question": "...",
  "final_answer": "The drug name",
  "final_explanation": "...",
  "final_kg": ["(h,r,t)", ...],
  "final_hop": final_kg's number(example. 5),
  "synthesis_rationale": "Briefly explain why you chose this combination (e.g., 'Adopted V2's correct inhaler route but kept V1's detailed enzyme description.')"
}}
"""
    return prompt


# =========================================================
# 3. 主程序
# =========================================================

async def main():
    # 1. 准备数据
    df_merged = load_and_align_data()

    if os.path.exists(EXISTING_DATA_PATH):
        df_done = pd.read_csv(EXISTING_DATA_PATH)
        # 假设用 'source' 做唯一标识，转成 string 防止类型不匹配
        done_sources = set(df_done['source'].astype(str))
        print(f"Already processed: {len(done_sources)}")

        # 排除掉已经做过的
        df_merged['source'] = df_merged['source'].astype(str)
        tasks_df = df_merged[~df_merged['source'].isin(done_sources)].copy()
        print(f"Remaining tasks to process: {len(tasks_df)}")
    else:
        print("Existing file not found, please check path!")
        return

    # 2. 准备 Tasks
    tasks_data = []
    print("Preparing prompts...")

    # 为了测试，可以先取前 5 条运行: df_merged.head(5).iterrows()
    # 正式运行请去掉 .head(5)
    for idx, row in tasks_df.iterrows():
        prompt_content = construct_synthesis_prompt(row)

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt_content}
        ]

        # 保存元数据，方便最后拼表
        meta_info = {
            "source": row.get('source'),
            "kg_source_id": row.get('kg_source_id', idx),  # 优先用原来的ID
            "original_index": idx
        }

        tasks_data.append({"messages": messages, "meta": meta_info})

    # 3. 初始化 API
    # 请确保替换这里为你的真实 key 或 base_url
    chat_client = OpenAIChat(
        model_name='deepseek-r1',  # 或 'deepseek-chat'
        api_key=env("OPENAI_API_KEY") or env("JUDGE_API_KEY", required=True),
        api_base=env("OPENAI_BASE_URL", get("api.judge_base_url")),
        max_tokens=3000
    )

    # 4. 批量运行
    batch_size = 20
    results = []

    print(f"Starting synthesis for {len(tasks_data)} items...")

    with tqdm(total=len(tasks_data), desc="Synthesizing", unit="task") as pbar:
        for i in range(0, len(tasks_data), batch_size):
            batch = tasks_data[i: i + batch_size]
            batch_messages = [t['messages'] for t in batch]

            # 异步请求
            responses = await chat_client.async_run(batch_messages, expected_type='json')

            # 解析结果
            for j, resp_str in enumerate(responses):
                meta = batch[j]['meta']
                if resp_str:
                    try:
                        # 清洗一下 markdown 格式 (```json ... ```)
                        if "```json" in resp_str:
                            resp_str = resp_str.split("```json")[1].split("```")[0]
                        elif "```" in resp_str:
                            resp_str = resp_str.split("```")[1].split("```")[0]

                        json_res = json.loads(resp_str)

                        # 构造最终的一行数据
                        record = {
                            "question": json_res.get('final_question'),
                            "answer": json_res.get('final_answer'),
                            "explanation": json_res.get('final_explanation'),
                            "evidence_KG": json_res.get('final_kg'),
                            "hop": json_res.get('final_hop'),
                            "kg_source_id": meta['kg_source_id'],
                            "source": meta['source'],
                        }
                        results.append(record)
                    except Exception as e:
                        print(f"JSON Parse Error for {meta['source']}: {e}")
                        # 可以选择把原始内容存下来debug
                        # results.append({"error": str(e), "raw": resp_str, **meta})
                else:
                    print(f"Request failed for {meta['source']}")

            # 阶段性保存 (每50条保存一次，防崩)
            if i % 50 == 0 and i > 0:
                pd.DataFrame(results).to_csv(NEW_OUTPUT_PATH, index=False)

            pbar.update(len(batch))

    # 5. 最终保存
    final_df = pd.DataFrame(results)
    final_df.to_csv(NEW_OUTPUT_PATH, index=False)
    print(f"Synthesis Complete! Processed {len(final_df)} items.")
    print(f"Saved to: {NEW_OUTPUT_PATH}")

    print("Merging two files...")
    df1 = pd.read_csv(EXISTING_DATA_PATH)
    df2 = pd.read_csv(NEW_OUTPUT_PATH)
    df_final = pd.concat([df1, df2], ignore_index=True)

    # 覆盖保存回原路径（这时候已经是 4360 + 1320 的完整版了）
    df_final.to_csv(EXISTING_DATA_PATH, index=False)
    print(f"Merge complete! Total rows: {len(df_final)}")


if __name__ == "__main__":
    asyncio.run(main())
