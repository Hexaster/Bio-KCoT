import asyncio
import openai
import json
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from collections import defaultdict
from typing import List

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import env, get, path

# ------------------------------------------------------------------------------
# 1. OpenAIChat 类
# ------------------------------------------------------------------------------
class OpenAIChat():
    # more details on: https://platform.openai.com/docs/api-reference/chat
    def __init__(
            self,
            model_name='gpt-4o-mini',  # 推荐使用支持 json mode 的模型
            max_tokens=5000,
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


# ------------------------------------------------------------------------------
# 2. 数据处理：构建邻接表
# ------------------------------------------------------------------------------
def parse_and_group_ppi(file_path):
    """
    读取 PPI 文件，按中心蛋白 (Hub) 分组。
    结构: { "UniprotID": [ {interaction_details}, ... ] }
    """
    grouped_data = defaultdict(list)

    print(f"Reading {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip(): continue
                parts = line.split('\t')
                if len(parts) < 6: continue

                id1 = parts[0]  # Interactor 1
                id2 = parts[3]  # Interactor 2

                # 过滤自相互作用 (Self-loops)
                if id1 == id2: continue

                context = parts[7] if len(parts) > 7 else "N/A"

                # 双向记录，因为 PPI 是无向图，任何一方都可以是 Subject
                # 记录 ID1 的邻居
                grouped_data[id1].append({
                    "partner_id": id2,
                    "context": context,
                    "raw_genes": parts[4]  # Ensembl IDs for extra info
                })
                # 记录 ID2 的邻居
                grouped_data[id2].append({
                    "partner_id": id1,
                    "context": context,
                    "raw_genes": parts[1]
                })

    except FileNotFoundError:
        print("File not found.")
        return {}

    # 过滤掉孤立点 (只有一个或没有相互作用的蛋白不适合做多跳推理)
    # 只保留至少有 2 个不同 interaction 的蛋白
    filtered_data = {k: v for k, v in grouped_data.items() if len(v) >= 2}
    print(f"Total proteins: {len(grouped_data)}, Proteins with >1 interaction: {len(filtered_data)}")
    return filtered_data


# ------------------------------------------------------------------------------
# 3. 构建 Prompt
# ------------------------------------------------------------------------------
FEW_SHOT_PROMPT = """
**CORE CONSTRAINTS:**
1. Minimal Info: Provide only the essential molecular states or interactions needed for logical reasoning. No extraneous narrative.
2. Unique Answer: Each question must have a single deterministic solution based on logic (Topology, Epistasis, AND-gate, or Dominant Negative). Avoid ambiguous outcomes.
3. Target Reasoning Patterns:
   - Epistasis ("Override"): A downstream mutation makes upstream states irrelevant.
   - Bottleneck ("AND Gate"): A complex requires multiple subunits; missing any → Fail.
   - Dominant Negative ("Poison"): One defective subunit disables the entire complex.dless of how much good subunit is present.

Output Format (JSON)
---
### Few-Shot Example 1 (Downstream Effect Reasoning)

**Input Data:**  
Target: ADCY7 (P41586)  
Interactions:  
- PAC1 (P18509) [Context: GPCR Signaling]  
- TrkA (Q16620) [Context: RTK Signaling]

**Output:**
{
  "question": "Protein ADCY7 receives activating input from PAC1 and inhibitory input from PDE4D. In a neuron where PACAP stimulation is combined with a PDE4D inhibitor, but ADCY7 carries a loss-of-function mutation that weakens its interaction with PAC1, what is the most likely net effect on intracellular cAMP levels, and why?",
  "answer": "Moderate Increase (not synergistic)",
  "explanation": "1. PACAP activates PAC1, which normally stimulates ADCY7. 2. ADCY7 produces cAMP, raising intracellular cAMP levels. 3. PDE4D degrades cAMP and normally limits the signal via negative feedback. 4. Inhibiting PDE4D removes the negative constraint, favoring cAMP accumulation. 5. However, the ADCY7 mutation weakens its interaction with PAC1, reducing upstream activation. 6. The loss of strong activation is partially compensated by reduced degradation, leading to a moderate but not synergistic increase in cAMP.",
  "evidence_KG": [
    "(PACAP, activates, PAC1)",
    "(PAC1, interacts_with, ADCY7)",
    "(ADCY7, produces, cAMP)",
    "(PDE4D, degrades, cAMP)",
    "(PDE4D_inhibitor, blocks, PDE4D)",
    "(ADCY7_mutant, weakens_interaction_with, PAC1)"
  ],
  "hop": 6
}

---
### Few-Shot Example 2 (Intervention / Blockade Reasoning)

**Input Data:**  
Target: PAC1 (P18509)  
Interactions:  
- G-protein Alpha S [Context: R-HSA-420103]

**Output:**
{
  "question": "A cell line expresses PAC1, Gs, ADCY7, and PKA. A drug (Drug-Y) blocks the PAC1–Gs interaction. Surprisingly, overexpression of a constitutively active ADCY7 mutant restores PKA activity even in the presence of Drug-Y. What does this imply about the position of ADCY7 in the signaling cascade, and why does the rescue occur?",
  "answer": "ADCY7 acts downstream of PAC1–Gs and can bypass the blockade",
  "explanation": "1. PAC1 normally activates Gs, which in turn activates ADCY7. 2. ADCY7 produces cAMP, which activates PKA. 3. Drug-Y blocks the PAC1–Gs interaction, preventing upstream signal transmission. 4. Without Gs activation, wild-type ADCY7 is not stimulated and cAMP production drops. 5. A constitutively active ADCY7 no longer requires Gs input. 6. Therefore, overexpressed active ADCY7 restores cAMP and PKA activation, bypassing the PAC1–Gs blockade and placing ADCY7 downstream of that interaction.",
  "evidence_KG": [
    "(PAC1, interacts_with, Gs)",
    "(Gs, activates, ADCY7)",
    "(ADCY7, produces, cAMP)",
    "(cAMP, activates, PKA)",
    "(Drug-Y, blocks, PAC1–Gs_interaction)",
    "(ADCY7_active_mutant, bypasses, Gs_requirement)"
  ],
  "hop": 6
}
"""

def construct_messages(target_id, interactions):
    # 将数据整理成清晰的文本
    data_str = f"Target Protein: {target_id}\nInteractions:\n"
    for i, p in enumerate(interactions[:15]):
        data_str += f"- Partner: {p['partner_id']} | Context: {p['context']}\n"

    user_content = f"""
    ### Current Task Data
    {data_str}

    Based on the 'Current Task Data', generate ONE reasoning question following the JSON format. 
    Focus on the relationship between the Target Protein and its Partners.
    """

    messages = [
        {"role": "system", "content": FEW_SHOT_PROMPT},
        {"role": "user", "content": user_content}
    ]
    return messages


# ------------------------------------------------------------------------------
# 4. 主流程
# ------------------------------------------------------------------------------
async def main():
    # --- 配置区域 ---
    INPUT_FILE = str(path("paths.data_dir") / "reactome_PPI.txt")
    OUTPUT_FILE = str(path("paths.data_dir") / "PPI_reasoning_questions.csv")
    INTERMEDIATE_FILE = str(path("paths.data_dir") / "intermediate_results.csv")
    BATCH_SIZE = 10  # 并发数
    LIMIT = 1  # 测试用，设置为 None 则跑全量
    # ----------------

    # 1. 初始化模型
    chat_client = OpenAIChat(model_name='deepseek-r1')

    # 2. 读取数据
    hub_data = parse_and_group_ppi(INPUT_FILE)
    if not hub_data: return

    # 3. 预处理任务列表 (Prepare Tasks)
    target_ids = list(hub_data.keys())
    target_ids.sort(key=lambda x: len(hub_data[x]), reverse=True)
    target_ids = target_ids[:2000]
    if LIMIT:
        target_ids = target_ids[:]

    tasks_data = []
    print(f"正在构建 {len(target_ids)} 个任务请求...")

    for tid in target_ids:
        msgs = construct_messages(tid, hub_data[tid])
        tasks_data.append({
            'messages': msgs,
            'meta': {
                'target_protein_id': tid,
                'interaction_count': len(hub_data[tid])
            }
        })

    # 4. 批量并发执行
    results = []

    # 使用 tqdm 显示进度条
    with tqdm(total=len(tasks_data), desc="Generating Reasoning Q&A", unit="task") as pbar:
        for i in range(0, len(tasks_data), BATCH_SIZE):
            # 切片获取当前批次
            batch = tasks_data[i: i + BATCH_SIZE]
            batch_messages = [t['messages'] for t in batch]

            # 调用异步接口
            responses = await chat_client.async_run(batch_messages)

            # 解析结果
            for j, resp_str in enumerate(responses):
                meta = batch[j]['meta']
                target_id = meta['target_protein_id']

                if resp_str:
                    try:
                        # 尝试解析 JSON
                        json_res = json.loads(resp_str)

                        # 将元数据合并进去 (Flatten)
                        json_res['target_protein_id'] = target_id
                        json_res['interaction_count'] = meta['interaction_count']

                        # 处理 evidence_KG (防止列表存入 CSV 变乱码，转为字符串)
                        if 'evidence_KG' in json_res and isinstance(json_res['evidence_KG'], list):
                            json_res['evidence_KG'] = json.dumps(json_res['evidence_KG'], ensure_ascii=False)

                        results.append(json_res)
                    except json.JSONDecodeError:
                        print(f"Failed to decode JSON for {target_id}")
                        # 保存错误信息以便后续debug
                        results.append({
                            "target_protein_id": target_id,
                            "error": "json_decode_error",
                            "raw_response": resp_str
                        })
                else:
                    print(f"Request failed or empty for {target_id}")
                    results.append({"target_protein_id": target_id, "error": "api_no_response"})

            # --- 中间保存 (关键步骤) ---
            # 每跑完一个 batch 就保存一次，防止程序崩溃白跑
            # 这里的写法是每次全量覆盖保存 (overwrite)，适合中小规模数据 (几万条以内)
            # 如果数据量极大，建议改用 mode='a' (append) 模式
            df_temp = pd.DataFrame(results)
            df_temp.to_csv(INTERMEDIATE_FILE, index=False, encoding='utf-8-sig')

            # 更新进度条
            pbar.update(len(batch))

    # 5. 最终保存
    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n全部完成! 结果已保存至 {OUTPUT_FILE}")
    print(f"共生成 {len(final_df)} 条数据。")


if __name__ == "__main__":
    asyncio.run(main())
