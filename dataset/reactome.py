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


# ---------------------------------------------------------
# 1. OpenAIChat API 类
# ---------------------------------------------------------
class OpenAIChat():
    def __init__(
            self,
            model_name='gpt-4o-mini',
            max_tokens=2500,
            temperature=0.5,
            top_p=1,
            request_timeout=180,
            stop=None,
            response_format='json_object',
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
                    print(f'Retry {try_i + 1} Error: {e}, waiting...')
                    await asyncio.sleep(5)
            return None

        async_responses = [_request_with_retry(messages) for messages in messages_list]
        return await asyncio.gather(*async_responses)

    async def async_run(self, messages_list, expected_type='json'):
        # 简化版实现，直接调用 dispatch
        predictions = await self.dispatch_openai_requests(messages_list)
        return [
            prediction['choices'][0]['message']['content'] if prediction is not None else None
            for prediction in predictions
        ]


# ---------------------------------------------------------
# 2. 数据处理与 Prompt 构建函数
# ---------------------------------------------------------

def format_pathway_context(pathway_name, reactions_df):
    """
    将一个 Pathway 下的所有 Reaction 行转换为一段可读的 Context 文本。
    """
    context = f"Pathway Name: {pathway_name}\n"

    # 获取 Pathway 级别的 Summary (通常第一行有，或者合并)
    summaries = reactions_df['Summary (Text)'].dropna().unique()
    if len(summaries) > 0:
        context += f"Pathway Summary: {summaries[0]}\n"

    context += "\nDetailed Reactions & Logic:\n"

    for idx, row in reactions_df.iterrows():
        context += f"--- Reaction: {row['Reaction_Name']} ---\n"
        context += f"  - Inputs: {row['Inputs']}\n"
        context += f"  - Outputs: {row['Outputs']}\n"
        context += f"  - Catalysts: {row['Catalysts']}\n"

        if pd.notna(row['Preceding_Events (Order)']) and row['Preceding_Events (Order)']:
            context += f"  - Preceding Events (Depends on): {row['Preceding_Events (Order)']}\n"

        if pd.notna(row['Regulators (Control)']) and row['Regulators (Control)']:
            context += f"  - Regulators: {row['Regulators (Control)']}\n"

        if pd.notna(row['Complex_Components (Detail)']) and row['Complex_Components (Detail)']:
            context += f"  - Detailed Components Breakdown: {row['Complex_Components (Detail)']}\n"

        context += "\n"

    return context


def build_prompt(pathway_context):
    """
    构建 Prompt，包含 System Prompt (Few-shot) 和 User Prompt (Context)
    """
    system_prompt = """
You are an expert biologist and logic reasoning dataset generator. 
Your task is to generate 2 high-quality reasoning Q&A cases based on the provided Biological Pathway Context.

For each pathway, you must generate exactly two types of cases:
1. **Causal Explanation**: Explain why a disruption (e.g., deficiency, mutation) leads to a specific outcome based on pathway logic.
2. **Counterfactual Intervention**: Predict the outcome of a complex scenario involving both overexpression and knockout/inhibition, requiring multi-hop reasoning.

**Strict Output Format**:
Return a JSON object with a key "cases" containing a list of 2 objects.

**Few-Shot Examples (Follow this style exactly)**:

Case 1 Example (Causal Explanation):
{
  "question": "In HIV-infected cells, researchers observed that although viral DNA successfully enters the nucleus, the efficiency of 2-LTR circular DNA formation is significantly reduced. Further detection revealed a marked decrease in the expression level of XRCC4. Explain why the decline in XRCC4 leads to the hindrance of 2-LTR circle formation.",
  "answer": "The downregulation of XRCC4 prevents the formation of the ligation complex, a critical intermediate step required before circularization can occur.",
  "explanation": "1. Pathway Identification: The formation of 2-LTR circles relies on the host's Non-Homologous End Joining (NHEJ) machinery processing linear viral DNA. 2. Complex Assembly: XRCC4 forms a mandatory complex with DNA Ligase IV (LIG4). This complex is responsible for the ligation step. 3. Sequential Dependency: The pathway follows a strict sequence: first, Ku proteins bind viral DNA; second, the 'Viral DNA:Ku Complex' must recruit the 'XRCC4:LIG4' complex. 4. Causal Reasoning: A decrease in XRCC4 expression leads to a shortage of the XRCC4:LIG4 complex. Without this complex, the intermediate 'Viral DNA:Ku Complex' cannot proceed to the ligation stage. 5. Conclusion: The pathway is blocked at the recruitment step, preventing the final circularization reaction.",
  "evidence_KG": [
    "(XRCC4, forms_complex_with, DNA Ligase IV)",
    "(Viral DNA:Ku Complex, input_of, Association of XRCC4:DNA ligase IV complex)",
    "(Association of XRCC4:DNA ligase IV complex, precedes, 2-LTR formation due to circularization)",
    "(2-LTR formation due to circularization, output, 2-LTR circle)"
  ],
  "hop": 4
}

Case 2 Example (Counterfactual Intervention):
{
  "question": "Hypothesize a scenario in an HIV-infected cell line where the enzymes XRCC4 and DNA Ligase IV (LIG4) are significantly overexpressed, but the Ku protein (Ku70/Ku80 heterodimer) is completely knocked out. Predict how the formation of 2-LTR circles will change and explain the reasoning based on the pathway logic.",
  "answer": "The formation of 2-LTR circles will be completely blocked/inhibited.",
  "explanation": "1. Pathway Logic: The 2-LTR circle formation is a multi-step sequential process. 2. Step 1 Requirement: The absolute first step requires Ku proteins (Ku70/Ku80) to bind to the ends of linear viral DNA to form the 'Viral DNA:Ku Complex'. 3. Step 2 Dependency: The subsequent reaction, which involves XRCC4 and LIG4, specifically requires the 'Viral DNA:Ku Complex' as its substrate (input), not just naked viral DNA. 4. Counterfactual Reasoning: Even though XRCC4 and LIG4 are abundant (overexpressed), the absence of Ku (Knockout) prevents Step 1 from happening. Consequently, the substrate for Step 2 is never created. 5. Conclusion: The pathway is bottlenecked at the very beginning. The downstream enzymes have nothing to act upon, resulting in zero formation of the final 2-LTR product.",
  "evidence_KG": [
    "(Ku70:Ku80, input_of, Association of Ku heterodimer with viral DNA)",
    "(Association of Ku heterodimer with viral DNA, produces, Viral DNA:Ku Complex)",
    "(Viral DNA:Ku Complex, input_of, Association of XRCC4:DNA ligase IV complex)",
    "(XRCC4:DNA Ligase IV, input_of, Association of XRCC4:DNA ligase IV complex)",
    "(Association of XRCC4:DNA ligase IV complex, precedes, 2-LTR formation)"
  ],
  "hop": 5
}
"""

    user_message = f"Here is the Pathway Context:\n{pathway_context}\n\nPlease generate the 2 cases now."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]


# ---------------------------------------------------------
# 3. 主程序逻辑
# ---------------------------------------------------------

async def main():
    # 1. 配置文件路径
    input_csv = str(path("paths.data_dir") / "reactome_full_knowledge.csv")
    output_csv = str(path("paths.data_dir") / "reactome_reasoning_dataset.csv")

    # 2. 初始化 OpenAI Chat
    chat = OpenAIChat(
        model_name='deepseek-r1',
        temperature=0.7,
    )

    # 3. 读取数据
    print(f"Reading CSV from {input_csv}...")
    df = pd.read_csv(input_csv)
    # 填充空值为字符串，防止报错
    df = df.fillna("")

    # 4. 按 Pathway_ID 分组
    grouped = df.groupby(['Pathway_ID', 'Pathway_Name'])
    print(f"Found {len(grouped)} unique pathways.")

    messages_batch = []
    metadata_batch = []  # 存储对应的 Pathway ID，方便后续处理

    # 5. 构建 Prompt 列表
    for (pid, pname), group in grouped:
        context = format_pathway_context(pname, group)
        messages = build_prompt(context)

        messages_batch.append(messages)
        metadata_batch.append({"Pathway_ID": pid, "Pathway_Name": pname})

    # 6. 执行并发请求
    batch_size = 10  # 每次并发 10 个 Pathway
    all_results = []

    print("Starting generation...")
    for i in tqdm(range(0, len(messages_batch), batch_size)):
        batch_msgs = messages_batch[i: i + batch_size]
        batch_meta = metadata_batch[i: i + batch_size]

        # 异步运行
        responses = await chat.async_run(batch_msgs)

        for meta, resp in zip(batch_meta, responses):
            if resp:
                try:
                    # 解析 JSON
                    data = json.loads(resp)
                    cases = data.get("cases", [])

                    # 将生成的 Case 添加到结果中，并附带元数据
                    for case in cases:
                        entry = {
                            "Pathway_ID": meta["Pathway_ID"],
                            "Pathway_Name": meta["Pathway_Name"],
                            **case  # 展开 generated case (question, answer, explanation...)
                        }
                        all_results.append(entry)
                except json.JSONDecodeError:
                    print(f"❌ JSON Decode Error for {meta['Pathway_ID']}")
                    continue
            else:
                print(f"❌ Failed to generate for {meta['Pathway_ID']}")

    # 7. 保存结果
    print(f"Saving {len(all_results)} generated cases to {output_csv}...")
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
