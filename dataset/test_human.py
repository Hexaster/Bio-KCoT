import asyncio
import openai
import os
import pandas as pd
import re
from typing import List
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import env, get, path

# ==============================================================================
# 1. 异步请求封装类 (基于你提供的代码进行微调)
# ==============================================================================
class OpenAIChat():
    def __init__(
            self,
            model_name='gpt-5.4-mini', 
            max_tokens=2000,
            temperature=0.1,
            top_p=1,
            request_timeout=180,
            stop=None,
            response_format=None, # 修改为 None，因为我们使用自由文本 + <score> 提取
            logprobs=False,
            top_logprobs=None,
            n=1,
            api_key="",
            api_base=""
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

        openai.api_key = api_key
        if api_base:
            openai.api_base = api_base

    async def dispatch_openai_requests(self, messages_list) -> List[str]:
        async def _request_with_retry(messages, retry=3):
            for try_i in range(retry):
                try:
                    kwargs = {
                        "model": self.config['model_name'],
                        "messages": messages,
                        "max_tokens": self.config['max_tokens'],
                        "temperature": self.config['temperature'],
                        "top_p": self.config['top_p'],
                        "request_timeout": self.config['request_timeout'],
                        "n": self.config['sample_n']
                    }
                    # 只有在明确指定 response_format 时才传入
                    if self.config['response_format']:
                        kwargs['response_format'] = {'type': self.config['response_format']}

                    response = await openai.ChatCompletion.acreate(**kwargs)
                    return response

                except openai.error.InvalidRequestError as e:
                    print(f"Invalid Request: {e}")
                    return None
                except (openai.error.RateLimitError, openai.error.APIError,
                        openai.error.Timeout, openai.error.APIConnectionError,
                        openai.error.ServiceUnavailableError) as e:
                    print(f'Retry {try_i + 1} Error: {e}, waiting...')
                    await asyncio.sleep(5 if try_i < 1 else 20)
                except Exception as e:
                    print(f"Unknown Error: {e}")
                    await asyncio.sleep(5)
            return None

        async_responses = [_request_with_retry(messages) for messages in messages_list]
        return await asyncio.gather(*async_responses)

    async def async_run(self, messages_list):
        retry = 5
        responses = [None for _ in range(len(messages_list))]
        messages_list_cur_index = list(range(len(messages_list)))

        while retry > 0 and len(messages_list_cur_index) > 0:
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]
            predictions = await self.dispatch_openai_requests(messages_list=messages_list_cur)

            preds = [
                pred['choices'][0]['message']['content'] if pred is not None else None 
                for pred in predictions
            ]

            finised_index = []
            for i, pred in enumerate(preds):
                if pred is not None:
                    responses[messages_list_cur_index[i]] = pred
                    finised_index.append(messages_list_cur_index[i])

            messages_list_cur_index = [i for i in messages_list_cur_index if i not in finised_index]
            retry -= 1

        return responses

# ==============================================================================
# 2. Prompt 模板与解析逻辑
# ==============================================================================
SYSTEM_PROMPT = """You are a Senior Medical and Bioinformatics Expert evaluating the quality of a QA dataset.
You will be provided with a medical reasoning Task Category, Question, Gold Answer, Gold Explanation, and Evidence Knowledge Graph (KG).

Please evaluate the OVERALL QUALITY of the QUESTION and its GOLD STANDARD. 
You must provide ONLY ONE overall score on a scale of 1 to 10 (1 = extremely poor/flawed, 10 = perfect/rigorous).

In your text analysis, please qualitatively discuss the following three dimensions:
1. Validity: Scientific correctness and timeliness.
2. Complexity_KG: Reasoning depth and strict necessity of the provided KG.
3. Uniqueness: Unambiguity of the solution given the constraints.

CRITICAL INSTRUCTION: DO NOT assign separate or sub-scores to these three individual dimensions. They are strictly guidelines to structure your written analysis. 

Write down your analysis and reasoning freely. 
HOWEVER, you must conclude your response by providing the SINGLE final integer score enclosed EXACTLY in <score> and </score> tags. 
For example: <score>8</score>"""

def build_user_prompt(row):
    return f"""
Task Category: {row['Task_Category']}
Question: {row['Question']}
Gold Answer: {row['Gold_Answer']}
Gold Explanation: {row['Gold_Explanation']}
Evidence KG: {row['Evidence_KG']}
"""

def parse_response(text):
    if not text:
        return None, "API returned empty or failed."
    score = None
    match = re.search(r'<score>\s*(\d+)\s*</score>', text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        rationale = re.sub(r'<score>.*?</score>', '', text, flags=re.IGNORECASE).strip()
    else:
        rationale = text.strip() 
    return score, rationale

# ==============================================================================
# 3. 异步主程序调度 (分块并发 + 断点续传)
# ==============================================================================
async def main_async():
    # 你的 API 配置
    API_KEY = env("OPENAI_API_KEY") or env("JUDGE_API_KEY", required=True)
    BASE_URL = env("OPENAI_BASE_URL", get("api.judge_base_url"))
    MODEL_NAME = env("HUMAN_EVAL_MODEL", "gpt-5.4-2026-03-05")
    
    input_file = str(path("paths.data_dir") / "Expert_Review_Sample_100.xlsx")
    output_file = str(path("paths.data_dir") / "LLM_Evaluation_Results1.csv")
    
    print(f"Loading dataset {input_file}...")
    
    # 断点续传逻辑
    if os.path.exists(output_file):
        print(f"Found existing progress file: {output_file}. Resuming...")
        df = pd.read_csv(output_file)
    else:
        df = pd.read_excel(input_file)
        df['LLM_Score'] = None
        df['LLM_Rationale'] = ""

    # 初始化你封装好的聊天类
    chat_client = OpenAIChat(
        model_name=MODEL_NAME,
        api_key=API_KEY,
        api_base=BASE_URL,
        response_format=None # 确保关闭 JSON 模式，使用自由文本提取
    )

    # 找出还没有被评分的行的索引
    unprocessed_indices = df[df['LLM_Score'].isna()].index.tolist()
    print(f"Total rows to process: {len(unprocessed_indices)} / {len(df)}")

    # 分块并发：每次并发处理 CHUNK_SIZE 条数据
    CHUNK_SIZE = 10 
    
    for i in range(0, len(unprocessed_indices), CHUNK_SIZE):
        chunk_indices = unprocessed_indices[i : i + CHUNK_SIZE]
        print(f"\nProcessing batch {i//CHUNK_SIZE + 1} (Rows {chunk_indices[0]} to {chunk_indices[-1]})...")
        
        # 组装这一批次的 prompt 列表
        messages_list = []
        for idx in chunk_indices:
            row = df.loc[idx]
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(row)}
            ]
            messages_list.append(messages)

        # 发起并发请求！
        raw_responses = await chat_client.async_run(messages_list)

        # 解析并写入 DataFrame
        for idx, raw_text in zip(chunk_indices, raw_responses):
            score, rationale = parse_response(raw_text)
            df.at[idx, 'LLM_Score'] = score
            df.at[idx, 'LLM_Rationale'] = rationale

        # 这一批次跑完，立刻保存 CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Batch {i//CHUNK_SIZE + 1} saved successfully!")

    print(f"\n✅ All done! Results saved to {output_file}")

# 启动异步程序
if __name__ == "__main__":
    # 如果你在 Jupyter Notebook 中运行，使用 await main_async()
    # 如果在标准 Python 脚本中运行，使用 asyncio.run()
    asyncio.run(main_async())
