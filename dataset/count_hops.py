import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

# --- 配置路径 ---
# 根据您的要求，指向这两个目录
dirs_to_analyze = {
    "TRAIN Set": str(path("paths.train_data_dir")),
    "TEST Set": str(path("paths.test_data_dir")),
}


def analyze_hops_distribution():
    # 用于最后打印全局对比
    global_stats = {}

    for dataset_name, dir_path in dirs_to_analyze.items():
        print(f"\n{'=' * 60}")
        print(f"正在分析: {dataset_name}")
        print(f"目录: {dir_path}")
        print(f"{'=' * 60}")

        if not os.path.exists(dir_path):
            print(f"[错误] 目录不存在: {dir_path}")
            continue

        # 获取目录下所有 csv 文件
        files = sorted([f for f in os.listdir(dir_path) if f.endswith(".csv")])

        if not files:
            print("[警告] 该目录下没有 CSV 文件。")
            continue

        # 该数据集的总体统计
        dataset_total_hops = {}
        dataset_total_rows = 0

        for file in files:
            file_path = os.path.join(dir_path, file)
            try:
                df = pd.read_csv(file_path)

                # 检查是否存在 hop 列
                if 'hop' not in df.columns:
                    print(f"⚠️  跳过文件 {file}: 未找到 'hop' 列")
                    continue

                # 确保 hop 列是数值型或统一格式
                # 有些 hop 可能是 "2" 或 2.0，统一转为整数
                df['hop'] = pd.to_numeric(df['hop'], errors='coerce').fillna(0).astype(int)

                # 统计当前文件
                counts = df['hop'].value_counts().sort_index()
                total_in_file = len(df)
                dataset_total_rows += total_in_file

                print(f"\n📄 文件: {file}")
                print(f"   总行数: {total_in_file}")
                print(f"   Hop 分布:")

                for hop_val, count in counts.items():
                    if hop_val == 0: continue  # 跳过转换失败的异常值
                    print(f"     - {hop_val}-hop: {count:<5} (占比: {count / total_in_file:.2%})")

                    # 累加到该数据集总量
                    dataset_total_hops[hop_val] = dataset_total_hops.get(hop_val, 0) + count

            except Exception as e:
                print(f"❌ 读取文件 {file} 出错: {e}")

        # 保存该数据集的汇总信息
        global_stats[dataset_name] = dataset_total_hops

        # 打印该数据集的总汇总
        print(f"\n{'-' * 30}")
        print(f"📊 {dataset_name} 总计汇总 (Total: {dataset_total_rows})")
        print(f"{'-' * 30}")
        if dataset_total_rows > 0:
            for hop_val in sorted(dataset_total_hops.keys()):
                count = dataset_total_hops[hop_val]
                print(f"  {hop_val}-hop: {count:<6} (占比: {count / dataset_total_rows:.2%})")
        else:
            print("  无有效数据。")

    # --- 最终对比表 ---
    print(f"\n\n{'=' * 60}")
    print("FINAL COMPARISON: Train vs Test")
    print(f"{'=' * 60}")

    # 获取所有出现的 hop 值
    all_hops = set()
    for stats in global_stats.values():
        all_hops.update(stats.keys())

    sorted_hops = sorted(list(all_hops))

    # 打印表头
    header = f"{'Hop Count':<10} | {'TRAIN Count':<12} | {'TEST Count':<12}"
    print(header)
    print("-" * len(header))

    for hop in sorted_hops:
        train_c = global_stats.get("TRAIN Set", {}).get(hop, 0)
        test_c = global_stats.get("TEST Set", {}).get(hop, 0)
        print(f"{str(hop) + '-hop':<10} | {train_c:<12} | {test_c:<12}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    analyze_hops_distribution()
