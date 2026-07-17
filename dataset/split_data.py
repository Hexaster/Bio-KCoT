import pandas as pd
import os
import re
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

# --- 1. 配置路径 ---
base_dir = str(path("paths.data_dir") / "question_org")
output_dir = str(path("paths.data_dir") / "question_org" / "split")

files = {
    "indication": os.path.join(base_dir, "disease-indication-drug_synthesized_scored.csv"),
    "synergy": os.path.join(base_dir, "drug-synergy-drug_scored.csv"),
    "ppi": os.path.join(base_dir, "PPI_reasoning_questions_scored.csv"),
    "reactome": os.path.join(base_dir, "reactome_reasoning_dataset_scored.csv"),
}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# --- 2. 辅助函数 ---

def extract_indication_ids(row_str):
    match = re.search(r'\((\d+),\s*indication,\s*(\d+)\)', str(row_str))
    if match:
        return int(match.group(1)), int(match.group(2))  # Drug, Disease
    return None, None


def extract_synergy_ids(row_str):
    match = re.search(r'\((\d+),.*,\s*(\d+)\)', str(row_str))
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def split_dataframe_by_group(df, group_col, test_ratio=1 / 11, seed=42):
    unique_groups = df[group_col].unique()
    np.random.seed(seed)
    np.random.shuffle(unique_groups)
    n_test = max(1, int(len(unique_groups) * test_ratio))
    test_groups = set(unique_groups[:n_test])
    return df[~df[group_col].isin(test_groups)], df[df[group_col].isin(test_groups)]


def print_stats(name, total, train, test):
    discarded = total - train - test
    print(f"\n[{name} 统计结果]")
    print(f"  Total (原始):    {total}")
    print(f"  Train (保留):    {train}")
    print(f"  Test  (保留):    {test}")
    print(f"  Discarded (丢弃): {discarded} (占比: {discarded / total:.2%})")
    return discarded


# --- 3. 主逻辑 ---

def main():
    stats_summary = []

    print(">>> 第一步：读取数据并构建索引...")

    # --- 预读取 DDI ---
    df_syn = pd.read_csv(files["synergy"])
    col_syn = 'source_triplet_str' if 'source_triplet_str' in df_syn.columns else 'source_kg_index'
    syn_parsed = df_syn[col_syn].apply(extract_synergy_ids)
    df_syn['drug_a'] = [x[0] for x in syn_parsed]
    df_syn['drug_b'] = [x[1] for x in syn_parsed]
    df_syn = df_syn.dropna(subset=['drug_a', 'drug_b'])

    # --- 预读取 Indication ---
    df_ind = pd.read_csv(files["indication"])
    col_ind = 'source' if 'source' in df_ind.columns else 'evidence_KG'
    ind_parsed = df_ind[col_ind].apply(extract_indication_ids)
    df_ind['drug_id'] = [x[0] for x in ind_parsed]
    df_ind['disease_id'] = [x[1] for x in ind_parsed]
    df_ind = df_ind.dropna(subset=['drug_id', 'disease_id'])

    # 统计权重
    drug_counts_in_ind = df_ind['drug_id'].value_counts().to_dict()

    print("\n>>> 第二步：优先划分 DDI (Synergy)...")

    total_syn = len(df_syn)
    all_syn_drugs = list(set(df_syn['drug_a']).union(set(df_syn['drug_b'])))

    # 权重计算
    weights = []
    for d in all_syn_drugs:
        count = drug_counts_in_ind.get(d, 0)
        weights.append(1.0 / (count + 1.0))
    weights = np.array(weights)
    weights = weights / weights.sum()

    # -----------------------------------------------------------
    # [关键修改] target_node_ratio 从 0.30 改为 0.20
    # 这意味着只有 20% 的药物会被选入测试集，剩下的 80% 都在训练集
    # -----------------------------------------------------------
    target_node_ratio = 0.3
    n_test_nodes = int(len(all_syn_drugs) * target_node_ratio)

    print(f"  - DDI 节点划分比例设定为: {target_node_ratio:.0%} (Test Nodes)")

    np.random.seed(42)
    test_drugs_indices = np.random.choice(len(all_syn_drugs), size=n_test_nodes, replace=False, p=weights)
    ddi_test_drug_set = set([all_syn_drugs[i] for i in test_drugs_indices])
    ddi_train_drug_set = set(all_syn_drugs) - ddi_test_drug_set

    # 划分 DDI
    is_syn_train = df_syn.apply(lambda r: r['drug_a'] in ddi_train_drug_set and r['drug_b'] in ddi_train_drug_set,
                                axis=1)
    is_syn_test = df_syn.apply(lambda r: r['drug_a'] in ddi_test_drug_set and r['drug_b'] in ddi_test_drug_set, axis=1)

    syn_train = df_syn[is_syn_train]
    syn_test = df_syn[is_syn_test]

    # 保存 & 统计
    syn_train.drop(columns=['drug_a', 'drug_b']).to_csv(os.path.join(output_dir, "drug-synergy_train.csv"), index=False)
    syn_test.drop(columns=['drug_a', 'drug_b']).to_csv(os.path.join(output_dir, "drug-synergy_test.csv"), index=False)

    print_stats("Drug Synergy", total_syn, len(syn_train), len(syn_test))
    stats_summary.append(("Drug Synergy", total_syn, len(syn_train), len(syn_test)))

    print("\n>>> 第三步：Indication 划分 (带随机丢弃)...")

    total_ind = len(df_ind)

    # 1. 区分
    disease_groups = df_ind.groupby('disease_id')['drug_id'].apply(set)

    safe_diseases = []
    forced_test_diseases = []

    for disease_id, drugs in disease_groups.items():
        if not drugs.isdisjoint(ddi_test_drug_set):
            forced_test_diseases.append(disease_id)
        else:
            safe_diseases.append(disease_id)

    # 2. 构建 Train 集
    ind_train = df_ind[df_ind['disease_id'].isin(safe_diseases)]

    # 3. 构建 Test 集 (保持 Train 的 10%)
    target_test_count = int(len(ind_train) * 0.1)
    potential_test_df = df_ind[df_ind['disease_id'].isin(forced_test_diseases)]

    if len(potential_test_df) > target_test_count:
        # 下采样
        np.random.shuffle(forced_test_diseases)
        final_test_diseases = []
        current_count = 0
        for d_id in forced_test_diseases:
            rows_for_this_disease = len(df_ind[df_ind['disease_id'] == d_id])
            if current_count < target_test_count:
                final_test_diseases.append(d_id)
                current_count += rows_for_this_disease
            else:
                break
        ind_test = df_ind[df_ind['disease_id'].isin(final_test_diseases)]
    else:
        ind_test = potential_test_df

    # 双重检查
    leaked = ind_train[ind_train['drug_id'].isin(ddi_test_drug_set)]
    if not leaked.empty:
        print(f"  [严重错误] Train 中存在泄露数据！")

    # 保存 & 统计
    ind_train.drop(columns=['drug_id', 'disease_id']).to_csv(os.path.join(output_dir, "disease-indication_train.csv"),
                                                             index=False)
    ind_test.drop(columns=['drug_id', 'disease_id']).to_csv(os.path.join(output_dir, "disease-indication_test.csv"),
                                                            index=False)

    print_stats("Disease Indication", total_ind, len(ind_train), len(ind_test))
    stats_summary.append(("Disease Indication", total_ind, len(ind_train), len(ind_test)))

    print("\n>>> 第四步：PPI 和 Reactome 正常划分...")

    # PPI
    df_ppi = pd.read_csv(files["ppi"])
    total_ppi = len(df_ppi)
    ppi_train, ppi_test = split_dataframe_by_group(df_ppi, 'target_protein_id')

    ppi_train.to_csv(os.path.join(output_dir, "PPI_reasoning_train.csv"), index=False)
    ppi_test.to_csv(os.path.join(output_dir, "PPI_reasoning_test.csv"), index=False)

    print_stats("PPI Reasoning", total_ppi, len(ppi_train), len(ppi_test))
    stats_summary.append(("PPI Reasoning", total_ppi, len(ppi_train), len(ppi_test)))

    # Reactome
    df_react = pd.read_csv(files["reactome"])
    total_react = len(df_react)
    react_train, react_test = split_dataframe_by_group(df_react, 'Pathway_ID')

    react_train.to_csv(os.path.join(output_dir, "reactome_reasoning_train.csv"), index=False)
    react_test.to_csv(os.path.join(output_dir, "reactome_reasoning_test.csv"), index=False)

    print_stats("Reactome Reasoning", total_react, len(react_train), len(react_test))
    stats_summary.append(("Reactome Reasoning", total_react, len(react_train), len(react_test)))

    print("\n" + "=" * 60)
    print("FINAL SUMMARY REPORT (Adjusted DDI Ratio)")
    print("=" * 60)
    print(f"{'Dataset':<20} | {'Total':<8} | {'Train':<8} | {'Test':<8} | {'Discarded':<10}")
    print("-" * 60)
    for name, tot, tr, te in stats_summary:
        disc = tot - tr - te
        print(f"{name:<20} | {tot:<8} | {tr:<8} | {te:<8} | {disc:<10}")
    print("=" * 60)
    print(f"\n所有文件已保存至: {output_dir}")


if __name__ == "__main__":
    main()
