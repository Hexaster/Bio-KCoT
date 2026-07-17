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


# ---------------------------------------------------------
# 1. OpenAIChat API 类
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

# ---------------------------------------------------------
# 2. 数据处理与 Prompt 构建
# ---------------------------------------------------------

FILE_PATHS = {
    'kg': str(path("paths.primekg_dir") / "kg.csv"),
    'drug_feat': str(path("paths.primekg_dir") / "drug_features.csv"),
}

# 增加了更多药理学相关的字段，以便推理 DDI
TARGET_DRUG_COLS = [
    'description', 'half_life', 'indication', 'mechanism_of_action',
    'protein_binding', 'pharmacodynamics', 'state',
    'atc_1', 'atc_2', 'atc_3', 'atc_4',
    'category', 'group', 'pathway',
    'molecular_weight', 'tpsa', 'clogp',
    'toxicity', 'targets', 'enzymes', 'transporters'
]


def load_data():
    try:
        df_kg = pd.read_csv(FILE_PATHS['kg'])
        df_drug = pd.read_csv(FILE_PATHS['drug_feat'])
        df_drug['node_index'] = df_drug['node_index'].astype(str)
        df_drug = df_drug.set_index('node_index')
        return df_kg, df_drug
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None


def extract_features(row, target_cols):
    features = {}
    for col in target_cols:
        if col in row.index:
            val = row[col]
            if hasattr(val, 'item'): val = val.item()
            if pd.notna(val) and str(val).strip() != "":
                features[col] = val
    return features


def construct_ddi_prompt(drug_a_data, drug_b_data, relation_name, id_a, id_b):
    # 提取两个药物的特征
    info_a = extract_features(drug_a_data, TARGET_DRUG_COLS)
    info_b = extract_features(drug_b_data, TARGET_DRUG_COLS)

    source_triplet = {
        "relation": relation_name,
        "drug_A_id": id_a,
        "drug_B_id": id_b
    }

    # ---------------------------------------------------------
    # Refactored Few-Shot: Drug-Drug Synergistic Interaction
    # ---------------------------------------------------------
    few_shot_examples = """
Example:
Input Context: Drug A: Sulfamethoxazole, Drug B: Trimethoprim. Relation: Synergistic interaction.
Output JSON:
{
  "question": "What potential drug-drug interactions or risks might arise from the co-administration of NS-398 and Fluperolone? Please explain the reasoning.",
  "answer": "Functional anti-inflammatory synergy with synergistic amplification of toxicity (especially gastrointestinal and renal toxicity)",
  "explanation": "1. Drug Profile & Classification: NS-398 is a selective COX-2 inhibitor (NSAID). Fluperolone is a moderately potent Corticosteroid (Glucocorticoid). 2. Molecular Mechanism of Action: NS-398 specifically inhibits the Cyclooxygenase-2 (COX-2) isoenzyme, blocking the conversion of Arachidonic Acid to Prostaglandin H2 (PGH2) and reducing inflammatory prostaglandins (PGE2). Fluperolone binds to glucocorticoid receptors, upregulating Lipocortin-1 (Annexin A1), which inhibits Phospholipase A2 (PLA2), thereby cutting off the supply of Arachidonic Acid at the source. 3. Interaction Logic (Pathway Reasoning): Both drugs target the same 'Arachidonic Acid Cascade' but at different levels. Fluperolone reduces the substrate (Arachidonic Acid), while NS-398 blocks the downstream enzyme (COX-2). While this provides powerful dual anti-inflammatory efficacy, it also critically depletes constitutive prostaglandins (PGE2, PGI2) that are essential for maintaining gastric mucosal integrity and renal blood flow. 4. Clinical Consequence: The combination removes both the substrate and the enzymatic pathway for producing cytoprotective prostaglandins. This results in a 'Synergistic Toxicity,' dramatically increasing the risk of gastric ulceration, bleeding, and renal ischemia compared to using either drug alone.",
  "evidence_KG": [
    "(NS-398, inhibits, COX-2)",
    "(Fluperolone, targets, Glucocorticoid Receptor)",
    "(Glucocorticoid Receptor, regulates, Phospholipase A2)",
    "(Arachidonic Acid pathway, involves, PLA2 and COX-2)",
    "(Prostaglandins, maintain, Gastric Mucosa)"
  ],
  "hop": 5
}
"""

    prompt = f"""
You are an expert pharmacology reasoning engine.

Your task is to generate a dataset entry consisting of a **Concise Clinical Question** and a **Deeply Reasoned Mechanistic Answer**.

- **GOAL**:
  1. **The Question**: Keep it simple. "What are the potential interactions between [Drug A] and [Drug B]?"
  2. **The Answer**: State the primary outcome (Synergy, Antagonism, Toxicity).
  3. **The Explanation**: MUST be a detailed, multi-step biological deduction. Do not skip the "Mechanism" step.

- **REQUIRED REASONING STRUCTURE (The 4 Steps)**:
  * **Step 1: Drug Profile**: Identify the class and category of each drug.
  * **Step 2: Molecular Mechanism (CRITICAL)**: Explain *HOW* each drug works. (e.g., "Inhibits Enzyme X," "Blocks Receptor Y," "Reduces synthesis of Z"). You must describe the biochemical pathway.
  * **Step 3: Interaction Logic**: Analyze how these two mechanisms interact within the body. Do they hit the same pathway? Do they overlap in toxicity targets (e.g., Kidney, Stomach)?
  * **Step 4: Clinical Consequence**: Synthesize the mechanism and logic into a final clinical risk or benefit summary.

- **INPUT DATA USAGE**:
  • Use `mechanism_of_action`, `targets`, `enzymes` from the input to fill in Step 2.

### One-Shot Example (Follow this structure):
{few_shot_examples}

### Current Task Input:
Source Triple: {source_triplet}

Drug A Features: 
{json.dumps(info_a, indent=2)}

Drug B Features: 
{json.dumps(info_b, indent=2)}

### Output (JSON only):
"""
    return prompt


# ---------------------------------------------------------
# 3. 主逻辑
# ---------------------------------------------------------

async def main():
    # 1. 载入数据 (只读 KG 和 Drug)
    df_kg, df_drug = load_data()
    if df_kg is None: return

    # 2. 筛选 Synergistic Interaction
    target_relation = 'synergistic interaction'
    ddi_df = df_kg[df_kg['display_relation'] == target_relation].copy()

    print(f"Total raw {target_relation} relations: {len(ddi_df)}")

    # 3. 确保 x 和 y 都是 Drug
    ddi_df = ddi_df[
        (ddi_df['x_type'].astype(str).str.contains('drug', case=False)) &
        (ddi_df['y_type'].astype(str).str.contains('drug', case=False))
        ]

    # 4. 采样策略：按 Drug A (x_index) 分组，每种药取 1-2 个协同互作对象
    ddi_df = ddi_df.groupby('x_index', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 1), random_state=42)
    )

    print(f"Filtered (Up to 2 interactions per Drug): {len(ddi_df)} tasks remaining")

    # 准备 Prompt 列表
    tasks_data = []

    for idx, row in ddi_df.iterrows():
        id_a = str(row['x_index'])
        id_b = str(row['y_index'])

        # 检查两个药物是否有特征数据
        if id_a not in df_drug.index or id_b not in df_drug.index:
            continue

        drug_a_data = df_drug.loc[id_a]
        drug_b_data = df_drug.loc[id_b]

        # 获取药物名称作为 meta info
        name_a = drug_a_data.get('description', f"Drug_{id_a}")
        name_b = drug_b_data.get('description', f"Drug_{id_b}")

        # 构建 Prompt
        prompt_content = construct_ddi_prompt(
            drug_a_data,
            drug_b_data,
            row['display_relation'],
            id_a,
            id_b
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt_content}
        ]

        source_str = f"({id_a}, {row['display_relation']}, {id_b})"

        meta_info = {
            "source_kg_index": idx,
            "source_triplet_str": source_str,
            "drug_a_name": name_a,
            "drug_b_name": name_b
        }
        tasks_data.append({"messages": messages, "meta": meta_info})

    print(f"Valid tasks prepared: {len(tasks_data)}")

    # 防止 token 爆炸，演示时只跑前 50 个
    tasks_data = tasks_data[:]

    # 5. 初始化 API 类
    chat_client = OpenAIChat(
        model_name='deepseek-r1'
    )

    # 6. 批量处理
    batch_size = 10
    results = []

    with tqdm(total=len(tasks_data), desc="Generating DDI Q&A", unit="task") as pbar:
        for i in range(0, len(tasks_data), batch_size):
            batch = tasks_data[i: i + batch_size]
            batch_messages = [t['messages'] for t in batch]

            responses = await chat_client.async_run(batch_messages, expected_type='json')

            for j, resp_str in enumerate(responses):
                meta = batch[j]['meta']
                if resp_str:
                    try:
                        json_res = json.loads(resp_str)
                        # 合并元数据
                        json_res.update(meta)
                        results.append(json_res)
                    except json.JSONDecodeError:
                        print(f"Failed to decode JSON for {meta['drug_a_name']} + {meta['drug_b_name']}")
                        results.append({"error": "json_decode_error", "raw": resp_str, **meta})
                else:
                    pass

            # 中间保存
            if i % 10 == 0:
                pd.DataFrame(results).to_csv(path("paths.data_dir") / "intermediate_ddi_results.csv", index=False)

            pbar.update(len(batch))

    # 7. 最终保存
    final_df = pd.DataFrame(results)
    final_df.to_csv(path("paths.data_dir") / "drug-synergy-drug.csv", index=False)
    print("All done! Saved to drug-synergy-drug.csv")


if __name__ == "__main__":
    asyncio.run(main())
