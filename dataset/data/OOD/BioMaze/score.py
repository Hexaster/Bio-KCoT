import json
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path


def analyze_evaluation_results(file_path):
    # 1. 加载数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 转换为 pandas DataFrame 方便统计
    df = pd.DataFrame(data)

    # 清理数据：确保 score 是数值类型，如果 API 失败导致没分数，填 0
    df['score'] = pd.to_numeric(df.get('score', 0), errors='coerce').fillna(0)

    total_samples = len(df)

    # 2. 基础指标计算
    avg_score = df['score'].mean()

    # 定义阈值
    # 严格准确率：100分 (或根据你的 prompt 设为 >= 90)
    strict_acc = (df['score'] == 100).sum() / total_samples * 100

    # 宽松准确率：>= 50分
    lenient_acc = (df['score'] >= 50).sum() / total_samples * 100

    # 完全错误率：0分
    zero_score_rate = (df['score'] == 0).sum() / total_samples * 100

    print("=" * 40)
    print("📊 整体评估结果")
    print("=" * 40)
    print(f"测试总数: {total_samples}")
    print(f"平均分 (0-100): {avg_score:.2f}")
    print(f"严格准确率 (满分): {strict_acc:.2f}%")
    print(f"宽松准确率 (>=50分): {lenient_acc:.2f}%")
    print(f"完全跑题/错误率 (0分): {zero_score_rate:.2f}%")

    # 3. 按标签（biocategory）分组统计分析
    # 如果你的数据中包含 biocategory 字段，可以看看模型在不同生物学科的表现
    if 'biocategory' in df.columns:
        print("\n" + "=" * 40)
        print("🧬 按生物类别 (biocategory) 统计平均分")
        print("=" * 40)

        # 按分类计算平均分和样本数
        category_stats = df.groupby('biocategory').agg(
            样本数=('score', 'count'),
            平均分=('score', 'mean'),
            满分率=('score', lambda x: (x == 100).sum() / len(x) * 100)
        ).round(2)

        # 按平均分降序排列
        category_stats = category_stats.sort_values(by='平均分', ascending=False)
        print(category_stats.to_string())

    # 4. 按问题类型（Inquiry Type）分组统计
    if 'Inquiry Type' in df.columns:
        print("\n" + "=" * 40)
        print("❓ 按问题类型 (Inquiry Type) 统计平均分")
        print("=" * 40)

        type_stats = df.groupby('Inquiry Type').agg(
            样本数=('score', 'count'),
            平均分=('score', 'mean')
        ).round(2)
        print(type_stats.to_string())


if __name__ == "__main__":
    # 替换为你上一步跑出来的结果文件路径
    result_file = str(path("paths.ood_biomaze") / "biomaze_openended_results_openbiollm-8b_evaluated2.json")
    analyze_evaluation_results(result_file)
