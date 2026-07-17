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
    'disease_feat': str(path("paths.primekg_dir") / "disease_features.csv"),
}

TARGET_DRUG_COLS = [
    'description', 'half_life', 'indication', 'mechanism_of_action',
    'protein_binding', 'pharmacodynamics', 'state',
    'atc_1', 'atc_2', 'atc_3', 'atc_4',
    'category', 'group', 'pathway',
    'molecular_weight', 'tpsa', 'clogp'
]

TARGET_DISEASE_COLS = [
    'mondo_id', 'mondo_name',
    'group_id_bert', 'group_name_bert',
    'mondo_definition', 'umls_description',
    'orphanet_definition', 'orphanet_prevalence',
    'orphanet_epidemiology', 'orphanet_clinical_description',
    'orphanet_management_and_treatment',
    'mayo_symptoms', 'mayo_causes',
    'mayo_risk_factors', 'mayo_complications',
    'mayo_prevention', 'mayo_see_doc'
]

def load_data():
    try:
        df_kg = pd.read_csv(FILE_PATHS['kg'])

        df_drug = pd.read_csv(FILE_PATHS['drug_feat'])
        df_drug['node_index'] = df_drug['node_index'].astype(str)
        df_drug = df_drug.set_index('node_index')

        df_disease = pd.read_csv(FILE_PATHS['disease_feat'])
        df_disease['node_index'] = df_disease['node_index'].astype(str)
        df_disease = df_disease.set_index('node_index')

        return df_kg, df_drug, df_disease
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None, None

def extract_features(row, target_cols):
    """
    辅助函数：从Series中提取指定列，过滤掉空值
    """
    features = {}
    for col in target_cols:
        if col in row.index:
            val = row[col]
            if hasattr(val, 'item'):
                val = val.item()
            if pd.notna(val) and str(val).strip() != "":
                features[col] = val
    return features


def construct_prompt(drug_data, disease_data, relation_name, drug_id, disease_id):
    # 提取完整特征
    d_info = extract_features(drug_data, TARGET_DRUG_COLS)
    ds_info = extract_features(disease_data, TARGET_DISEASE_COLS)

    if hasattr(drug_id, 'item'): drug_id = drug_id.item()
    if hasattr(disease_id, 'item'): disease_id = disease_id.item()

    source_triplet = {
        "relation": relation_name,
        "drug_node": drug_id,
        "disease_node": disease_id
    }

    # ---------------------------------------------------------
    # Few-Shot 示例
    # ---------------------------------------------------------
    few_shot_examples = """
Example:
Input Context: Drug: Flurandrenolide, Disease: Atopic Dermatitis.
Output JSON:
{
  "question": "A patient presents with a chronic inflammatory skin condition characterized by an increased ability to form reagin and a hereditary low threshold for pruritus. Which mid-potency topical corticosteroid is most appropriate for treating this condition, and through what molecular mechanism does it reduce inflammation and itching?",
  "answer": "Flurandrenolide",
  "explanation": "1. Disease Identification: The symptoms describe Atopic Dermatitis (Eczema). 2. Drug Selection: Flurandrenolide is a potent topical corticosteroid used for this condition. 3. Mechanistic Reasoning: The drug binds to cytosolic glucocorticoid receptors, inducing the synthesis of Lipocortin-1 (Annexin-1). Lipocortin-1 directly inhibits Phospholipase A2 (PLA2). By blocking PLA2, the drug prevents the release of Arachidonic Acid from cell membranes. This acts as a bottleneck, cutting off the supply for both the COX pathway (inflammation) and the LOX pathway (Leukotrienes). Since Leukotrienes are responsible for lowering the itch threshold, inhibiting their production effectively alleviates the pruritus.",
  "evidence_KG": [
    "(Atopic Dermatitis, treated_by, Flurandrenolide)",
    "(Flurandrenolide, induces_synthesis_of, Lipocortin-1)",
    "(Lipocortin-1, inhibits, Phospholipase A2)",
    "(Phospholipase A2, releases, Arachidonic Acid)",
    "(Arachidonic Acid, precursor_of, Leukotrienes)"
  ],
  "hop": 5
}
"""

    prompt = f"""
You are an expert biology reasoning question writer.

Your task is to generate a reasoning-style question whose core is:
→ “Given a specific disease symptom or pathological process, which drug is most appropriate, and what is its molecular mechanism of action?”

- OVERALL GOAL
• The QUESTION must describe disease symptoms / biological dysfunctions.
• The ANSWER must be a specific drug.
• The explanation must require MECHANISTIC REASONING.
• Add 1–2 distinguishing biological conditions to ensure the answer is UNIQUE.
• Keep the question concise — avoid unnecessary complexity that lowers reasoning quality.
• Do NOT explicitly state the drug’s molecular target, receptor name, enzyme, or pathway in the question itself. These must be inferred in the explanation.

### One-Shot Examples (Follow this logic depth):
{few_shot_examples}

### Current Task Input:
Source Triple: {source_triplet}

Drug Features (Context): 
{json.dumps(d_info, indent=2)}

Disease Features (Context): 
{json.dumps(ds_info, indent=2)}

### Output (JSON only):
"""
    return prompt


# ---------------------------------------------------------
# 3. 主逻辑
# ---------------------------------------------------------

async def main():
    # 1. 载入数据
    df_kg, df_drug, df_disease = load_data()
    if df_kg is None: return

    # 1. 初步筛选关系
    indication_df = df_kg[df_kg['display_relation'] == 'indication'].copy()
    print(f"Total raw indication relations: {len(indication_df)}")

    # 2. 定义辅助函数：识别每一行的 drug ID
    def get_drug_id(row):
        if 'drug' in str(row['x_type']).lower():
            return row['x_index']
        elif 'drug' in str(row['y_type']).lower():
            return row['y_index']
        return None

    # 3. 创建临时列用于分组
    indication_df['temp_drug_id'] = indication_df.apply(get_drug_id, axis=1)

    # 4. 去除无法识别 drug 的行
    indication_df = indication_df.dropna(subset=['temp_drug_id'])

    # 5. 核心逻辑：按 drug ID 分组，每组随机取 2 个 (sample(n=3))
    indication_df = indication_df.groupby('temp_drug_id', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), 2), random_state=42)
    )

    print(f"Filtered (Up to 3 Diseases per Drug): {len(indication_df)} tasks remaining")

    # 准备 Prompt 列表
    tasks_data = []  # 存储 (prompt, original_row_info)

    for idx, row in indication_df.iterrows():
        x_type = str(row['x_type']).lower()

        # 匹配 Drug 和 Disease
        if 'drug' in x_type:
            drug_id = row['x_index']
            disease_id = row['y_index']
        else:
            # 否则反过来：Y 是药，X 是病
            drug_id = row['y_index']
            disease_id = row['x_index']

        drug_idx_str = str(drug_id)
        disease_idx_str = str(disease_id)

        if drug_idx_str not in df_drug.index or disease_idx_str not in df_disease.index:
            continue

        drug_data = df_drug.loc[drug_idx_str]
        disease_data = df_disease.loc[disease_idx_str]

        prompt_content = construct_prompt(
            drug_data,
            disease_data,
            row['display_relation'],
            drug_id,
            disease_id
        )
        # 构造 OpenAI 消息格式
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt_content}
        ]

        source_str = f"({drug_id}, {row['display_relation']}, {disease_id})"

        # 保存元数据以便后续合并结果
        meta_info = {
            "source_kg_index": idx,
            "source_triplet_str": source_str,
            "drug_name": drug_data.get('description', 'Unknown'),
            "disease_name": disease_data.get('mondo_name', 'Unknown')
        }
        tasks_data.append({"messages": messages, "meta": meta_info})

    print(f"Valid tasks prepared: {len(tasks_data)}")
    tasks_data = tasks_data[:]

    # 3. 初始化 API 类
    chat_client = OpenAIChat(
        model_name='gpt-5.4'
        # response_format='json_object'
    )

    # 4. 批量处理 (Batch Processing)
    batch_size = 10  # 每次并发处理 10 个
    results = []
    with tqdm(total=len(tasks_data), desc="Generating Q&A", unit="task") as pbar:
        for i in range(0, len(tasks_data), batch_size):
            batch = tasks_data[i: i + batch_size]
            batch_messages = [t['messages'] for t in batch]

            # print(f"Processing batch {i} to {min(i + batch_size, len(tasks_data))}...")

            # 调用异步接口
            responses = await chat_client.async_run(batch_messages, expected_type='json')

            # 解析结果
            for j, resp_str in enumerate(responses):
                meta = batch[j]['meta']
                if resp_str:
                    try:
                        # 尝试解析 JSON
                        json_res = json.loads(resp_str)
                        # 将元数据合并进去
                        json_res['kg_source_id'] = meta['source_kg_index']
                        json_res['source'] = meta['source_triplet_str']
                        results.append(json_res)
                    except json.JSONDecodeError:
                        print(f"Failed to decode JSON for {meta['drug_name']}")
                        results.append({"error": "json_decode_error", "raw": resp_str, **meta})
                else:
                    print(f"Request failed for {meta['drug_name']}")

            # 可选：每批保存一次，防止程序中断丢失数据
            pd.DataFrame(results).to_csv(path("paths.data_dir") / "intermediate_results_claude.csv", index=False)
            pbar.update(len(batch))

    # 5. 最终保存
    final_df = pd.DataFrame(results)
    final_df.to_csv(path("paths.data_dir") / "disease-indication-drug-claude.csv", index=False)
    print("All done! Saved to disease-indication-drug-claude.csv")


# 运行异步主程序
if __name__ == "__main__":
    asyncio.run(main())
