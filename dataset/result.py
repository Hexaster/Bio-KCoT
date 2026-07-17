import pandas as pd
import json
import asyncio
import openai
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import env, get, path
from tqdm import tqdm
import os


# =========================================================
# 1. 你的 OpenAIChat 类 (严格保持原样，未做修改)
# =========================================================
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
                        # 注意：如果不需要强制JSON，建议初始化时传入 response_format=None
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
                        # 只有当 response_format 不为 None 时才加入参数，防止报错
                        if self.config['response_format']:
                            kwargs['response_format'] = {'type': self.config['response_format']}

                        response = await openai.ChatCompletion.acreate(**kwargs)
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
# 2. 业务逻辑代码
# =========================================================

# --- 配置 ---
INPUT_DIR = str(path("paths.test_data_dir", env="TEST_DATA_DIR"))
FILES_TO_PROCESS = [
    'disease-indication_test.csv',
    'drug-synergy_test.csv',
    'PPI_reasoning_test.csv',
    'reactome_reasoning_test.csv'
]

# API 配置
MODEL_NAME = "o3-mini"  # 或者是 deepseek-chat

# 批处理大小 (一次发多少个请求给 async_run)
BATCH_SIZE = 5


def construct_messages(question):
    """构建输入给模型的 message 列表"""
    prompt = f"""
You are an expert in biomedical reasoning.
Please answer the following question.

Question: {question}

Answer:
"""
    return [
        {"role": "user", "content": prompt}
    ]


async def process_single_csv(file_name, chat_client):
    input_path = os.path.join(INPUT_DIR, file_name)
    output_path = os.path.join(INPUT_DIR, file_name.replace('.csv', '_o3-mini_result.csv'))

    print(f"\n🚀 开始处理: {file_name}")

    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return

    df = pd.read_csv(input_path)

    if 'question' not in df.columns:
        print(f"⚠️ {file_name} 缺少 'question' 列，跳过")
        return

    # --- 断点续传逻辑 ---
    processed_indices = set()
    results_map = {}  # index -> result string

    if os.path.exists(output_path):
        print("  🔄 检测到已有结果，正在加载...")
        df_existing = pd.read_csv(output_path)
        # 假设只要这一行有 model_response 且不为空，就算处理过了
        for i, row in df_existing.iterrows():
            if pd.notna(row.get('model_response')) and str(row.get('model_response')) != "":
                results_map[i] = row['model_response']
                processed_indices.add(i)

    # 准备任务列表 (indices)
    all_indices = list(df.index)
    pending_indices = [i for i in all_indices if i not in processed_indices]

    print(f"  📊 总条数: {len(df)}, 待处理: {len(pending_indices)}")

    if not pending_indices:
        print("  ✅ 该文件已全部处理完成。")
        return

    # 分批处理
    # range step = BATCH_SIZE
    for i in tqdm(range(0, len(pending_indices), BATCH_SIZE), desc=f"Running {file_name}"):
        batch_indices = pending_indices[i: i + BATCH_SIZE]

        # 1. 构建 batch messages
        batch_messages_list = []
        for idx in batch_indices:
            q = str(df.loc[idx, 'question'])
            batch_messages_list.append(construct_messages(q))

        # 2. 调用 API (async_run 会并发处理这个列表)
        # 注意：这里 expected_type 只是个标记，DeepSeek R1 可能不返回 JSON，所以我们不管它
        batch_responses = await chat_client.async_run(batch_messages_list, expected_type='text')

        # 3. 收集结果并保存
        new_rows = []
        for j, local_idx in enumerate(batch_indices):
            resp = batch_responses[j]
            if resp is None:
                resp = "Error/Empty"

            # 记录到内存 map (其实也可以直接写文件)
            results_map[local_idx] = resp

            # 准备写入行 (包含原始数据 + 结果)
            row_data = df.loc[local_idx].to_dict()
            row_data['model_response'] = resp
            new_rows.append(row_data)

        # 4. 实时追加写入 (Append Mode)
        # 这样即使程序崩了，跑完的 batch 也就保存了
        if new_rows:
            df_batch = pd.DataFrame(new_rows)
            # 确保列顺序一致，如果文件不存在则写入header
            file_exists = os.path.exists(output_path)

            # 如果文件已存在，我们只追加，不要 header
            # 如果文件不存在，写入 header
            # 注意：这里的追加写入方式比较简单，适合简单的断点续传。
            # 更严谨的做法是最后统一合并，但为了防止丢失数据，我们用追加。
            # 为了保证结果CSV顺序和原CSV一致，通常最后建议做一次 sort 或 merge。
            # 但这里为了简单有效，我们将结果存入单独文件，或者直接追加。

            # 为了避免追加造成的乱序难以复原，推荐方案：
            # 每次只把新跑出来的存入一个 temp 文件，最后合并？
            # 或者：直接追加到 output_path。因为我们有 results_map，最后可以重新生成一个整洁的文件。

            df_batch.to_csv(
                output_path,
                mode='a',
                header=not file_exists,
                index=False,
                encoding='utf-8-sig'
            )

    print(f"  💾 完成。结果已追加至: {output_path}")


async def main():
    # 初始化你的 OpenAIChat 类
    # 注意：response_format=None，因为我们是问答，不强制 JSON
    chat_client = OpenAIChat(
        model_name=MODEL_NAME,
        response_format=None,  # 重要：DeepSeek R1 输出文本，不要强制 JSON Object
        request_timeout=300,
        max_tokens=4000
    )

    for file_name in FILES_TO_PROCESS:
        await process_single_csv(file_name, chat_client)

    print("\n🎉 所有文件处理完毕！")


if __name__ == "__main__":
    asyncio.run(main())
