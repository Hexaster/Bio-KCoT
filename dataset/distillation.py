import pandas as pd
import json
import asyncio
import openai
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import env, get, path
import os


# ---------------------------------------------------------
# 1. OpenAIChat API 类 (完全保留你提供的代码)
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
            response_format=None, 
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
            'response_format': {"type": "text"},
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
                            response_format=self.config['response_format'],
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


# ---------------------------------------------------------
# 2. 核心逻辑：生成思维链 (CoT)
# ---------------------------------------------------------

INPUT_CSV_PATH = str(path("paths.data_dir") / "reactome_reasoning_dataset.csv")
OUTPUT_CSV_PATH = str(path("paths.data_dir") / "reactome_reasoning_dataset_reasoning.csv")


def construct_reasoning_prompt(row):
    """
    构建 prompt，包含问题、答案和背景，要求生成思维过程
    """
    question = row.get('question', '')
    answer = row.get('answer', '')
    explanation = row.get('explanation', '')
    evidence_kg = row.get('evidence_KG', '')

    prompt = f"""
You are an expert in biomedical reasoning and pharmacology.
I will provide you with a Question and the Correct Answer.
Your task is to give the thinking process that leads from the Question to the Answer.

### Input Data
**Question:** {question}
**Correct Answer:** {answer}

### Task
1. Generate a step-by-step reasoning chain.
2. The reasoning process should pretend not to know the correct answer.
3. You MUST wrap the actual thinking content inside `<thought>` and `</thought>` tags.
4. Do NOT output JSON. Output only the tagged reasoning.

### Output Example
<thought>
... your detailed step-by-step reasoning here ...
</thought>
"""
    return prompt



async def main():
    chat = OpenAIChat(model_name='deepseek-r1', max_tokens=4000)

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"File not found: {INPUT_CSV_PATH}")
        return

    full_df = pd.read_csv(INPUT_CSV_PATH)

    # ===== 断点续跑 =====
    start_index = 0
    if os.path.exists(OUTPUT_CSV_PATH):
        try:
            processed_df = pd.read_csv(OUTPUT_CSV_PATH)
            start_index = len(processed_df)
            print(f"Found existing output. Resuming from row {start_index}...")
        except pd.errors.EmptyDataError:
            print("Output file exists but empty. Starting from 0.")

    df_to_process = full_df.iloc[start_index:].copy()
    print(f"Rows left to process: {len(df_to_process)}")

    # df_to_process = df_to_process[:5]

    # ===== 构造 prompts =====
    messages_list = []
    for index, row in df_to_process.iterrows():
        prompt = construct_reasoning_prompt(row)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        messages_list.append(messages)

    batch_size = 10

    print("Starting generation...")

    for i in tqdm(range(0, len(messages_list), batch_size)):
        batch_messages = messages_list[i:i+batch_size]
        current_batch_df = df_to_process.iloc[i:i+batch_size].copy()

        batch_responses = await chat.async_run(batch_messages)

        import re

        batch_reasonings = []
        pattern = re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE)

        for response_text in batch_responses:
            if response_text:
                match = pattern.search(response_text)
                if match:
                    reasoning_text = match.group(1).strip()  # 提取中间内容
                else:
                    reasoning_text = response_text.strip()  # 如果没有 <thought>，直接保存原文
                batch_reasonings.append(reasoning_text)
            else:
                batch_reasonings.append("")

        current_batch_df['cot_reasoning'] = batch_reasonings

        # ===== 新建 or 追加 =====
        write_header = not os.path.exists(OUTPUT_CSV_PATH)
        current_batch_df.to_csv(
            OUTPUT_CSV_PATH,
            mode='a',
            index=False,
            header=write_header
        )

    print(f"Processing complete. All data saved to {OUTPUT_CSV_PATH}")

if __name__ == "__main__": 
    asyncio.run(main())
