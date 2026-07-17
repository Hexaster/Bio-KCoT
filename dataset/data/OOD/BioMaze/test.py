import json
import asyncio
import openai
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import env, get, path


# ==========================================
# 1. 你提供的 OpenAIChat 类
# ==========================================
class OpenAIChat():
    # 保持你原来的类定义不变
    def __init__(
            self,
            model_name='gpt-4o-mini',  # 推荐使用支持 json mode 的模型
            max_tokens=2500,
            temperature=0.0,  # 评估任务建议将温度调为 0，保证结果稳定
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
                            top_logprobs=self.config['top_logprobs'],
                            n=self.config['sample_n'],
                        )
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

    async def async_run(self, messages_list, expected_type='json'):
        retry = 10
        responses = [None for _ in range(len(messages_list))]
        messages_list_cur_index = [i for i in range(len(messages_list))]

        while retry > 0 and len(messages_list_cur_index) > 0:
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]
            predictions = await self.dispatch_openai_requests(messages_list=messages_list_cur)

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


# ==========================================
# 2. 评估逻辑与 Prompt 设计
# ==========================================

EVALUATION_PROMPT = """你是一个专业的生物医学领域的评分专家。
请根据提供的【用户问题】和【标准答案】，对大模型的【预测回答】进行评分。

评分标准（0-100分）：
- 100分：预测回答完美涵盖了标准答案的核心信息，准确无误。
- 50分：预测回答包含部分正确信息，但不够完整或存在部分偏差。
- 0分：预测回答完全跑题（例如输出无关的文献或内容）、给出错误信息或未回答问题。

请仅输出一个 JSON 对象，结构如下：
{{
    "reasoning": "简短的评分理由（50字以内）",
    "score": <0到100之间的整数>
}}

【用户问题】：{question}
【标准答案】：{answer}
【预测回答】：{prediction}
"""


async def evaluate_dataset():
    # 配置文件路径
    input_file = str(path("paths.ood_biomaze") / "biomaze_openended_results_openbiollm-8b.json")
    output_file = str(path("paths.ood_biomaze") / "biomaze_openended_results_openbiollm-8b_evaluated2.json")

    # 1. 加载数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {len(data)} 条待评估数据。")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 2. 初始化你的 Chat 模型 (记得填入你的 API KEY)
    evaluator = OpenAIChat(
        model_name='deepseek-r1',
        temperature=0.0,  # 裁判模型建议用 0 temperature，降低随机性
    )

    # 3. 构造并发请求列表
    messages_list = []
    for item in data:
        # 清理异常长的 prediction 以防止超出 token 限制（可选）
        prediction_text = item.get('model_prediction', '')
        if len(prediction_text) > 8000:
            prediction_text = prediction_text[:8000] + "...[截断]"

        prompt = EVALUATION_PROMPT.format(
            question=item.get('question', ''),
            answer=item.get('answer', ''),
            prediction=prediction_text
        )

        messages = [
            {"role": "system", "content": "You are a strict, objective json-only evaluator."},
            {"role": "user", "content": prompt}
        ]
        messages_list.append(messages)

    print("开始并发请求大模型进行评分...")

    # 4. 执行异步请求
    responses = await evaluator.async_run(messages_list)

    # 5. 解析并写入结果
    for i, response in enumerate(responses):
        if response:
            try:
                # 解析模型返回的 JSON
                res_json = json.loads(response)
                # 记录 score 和 reasoning
                data[i]['score'] = int(res_json.get('score', 0))
                data[i]['judge_reasoning'] = res_json.get('reasoning', '')
            except json.JSONDecodeError:
                data[i]['score'] = 0
                data[i]['judge_reasoning'] = "解析 JSON 失败"
                print(f"第 {i} 条解析失败: {response}")
        else:
            data[i]['score'] = 0
            data[i]['judge_reasoning'] = "API 请求失败或返回为空"

    # 6. 保存最终结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"评估完成！结果已保存至: {output_file}")


if __name__ == "__main__":
    # Python 3.7+ 运行异步主函数的标准方式
    asyncio.run(evaluate_dataset())
