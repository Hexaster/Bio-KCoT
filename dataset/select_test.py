import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

def create_expert_review_dataset():
    # 1. 基础路径配置
    base_dir = str(path("paths.test_data_dir"))
    
    # 2. 定义各分类的文件名与对应需要抽取的样本量
    # 注意：请根据实际文件名调整 key
    sampling_plan = {
        "drug-synergy_test.csv": {"task_name": "Drug Synergy", "n_samples": 30},
        "disease-indication_test.csv": {"task_name": "Disease Indication", "n_samples": 26},
        "PPI_reasoning_test.csv": {"task_name": "PPI Reasoning", "n_samples": 15},
        "reactome_reasoning_test.csv": {"task_name": "Reactome Reasoning", "n_samples": 29}
    }
    
    # 我们只挑选专家评审真正需要的列
    columns_to_keep = ['question', 'answer', 'explanation', 'evidence_KG']
    
    sampled_dfs = []
    
    for filename, config in sampling_plan.items():
        filepath = os.path.join(base_dir, filename)
        
        try:
            # 读取 CSV
            df = pd.read_csv(filepath)
            
            # 随机抽样，random_state=42 保证每次运行抽出的题目一样，具有可复现性
            df_sampled = df.sample(n=config["n_samples"], random_state=42)
            
            # 仅保留需要的列
            df_sampled = df_sampled[columns_to_keep]
            
            # 插入任务类型标记，方便专家知道当前审的是什么题
            df_sampled.insert(0, 'Task_Category', config["task_name"])
            
            sampled_dfs.append(df_sampled)
            print(f"✅ 成功从 {filename} 抽取 {config['n_samples']} 条数据.")
            
        except FileNotFoundError:
            print(f"❌ 找不到文件: {filepath}，请检查路径。")
        except Exception as e:
            print(f"⚠️ 处理 {filename} 时发生错误: {e}")

    # 3. 合并所有抽样数据
    if not sampled_dfs:
        print("未提取到任何数据，程序退出。")
        return
        
    final_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # 4. 重命名列，使其更符合人类阅读习惯
    final_df.rename(columns={
        'question': 'Question',
        'answer': 'Gold_Answer',
        'explanation': 'Gold_Explanation',
        'evidence_KG': 'Evidence_KG'
    }, inplace=True)
    
    # 5. 生成专家打分列 (初始为空)
    final_df['Score_1_Validity (1-10)'] = ""
    final_df['Score_2_Reasoning_KG (1-10)'] = ""
    final_df['Score_3_Uniqueness (1-10)'] = ""
    final_df['Expert_Comments'] = ""
    
    # 6. 保存为 Excel 文件
    output_file = str(path("paths.data_dir") / "Expert_Review_Sample_100.xlsx")
    
    # 使用 ExcelWriter 并设置格式以优化阅读体验
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        final_df.to_excel(writer, index=False, sheet_name='Review Data')
        
        # 自动调整列宽，方便专家直接阅读长文本
        worksheet = writer.sheets['Review Data']
        worksheet.column_dimensions['B'].width = 60  # Question 列调宽
        worksheet.column_dimensions['D'].width = 50  # Explanation 列调宽
        worksheet.column_dimensions['E'].width = 50  # Evidence_KG 列调宽

    print(f"\n🎉 抽样完成！总计 {len(final_df)} 条数据已保存至: {output_file}")
    print("建议将此 Excel 直接发给专家进行评审，避免 CSV 格式导致的文本错位。")

if __name__ == "__main__":
    create_expert_review_dataset()
