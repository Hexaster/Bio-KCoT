import pandas as pd
import os
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

# --- 1. 配置路径 ---
input_dir = str(path("paths.train_data_dir"))
output_dir = str(path("paths.train_data_dir") / "sft_rl_aggressive_split")

# 目标比例
RL_RATIO = 0.4  # 30% 给 RL
SEED = 42  # 固定随机种子

# --- 关键参数：激进程度 ---
# POWER = 1: 线性 (你之前的版本)
# POWER = 3: 强力
# POWER = 5: 极度激进 (RL组几乎全是最高跳数，SFT全是低跳数)
WEIGHT_POWER = 4.5

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def split_aggressive():
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    print(f"开始执行激进加权划分 (Target: SFT 70% / RL 30%)")
    print(f"策略: 使用 Hop 的 {WEIGHT_POWER} 次方作为权重，强制拉开差距。")
    print("=" * 95)
    print(f"{'File':<30} | {'Set':<5} | {'Count':<6} | {'Avg Hop':<8} | {'Median':<8} | {'Max Hop':<8}")
    print("-" * 95)

    for filename in files:
        file_path = os.path.join(input_dir, filename)

        try:
            df = pd.read_csv(file_path)

            # 1. 清洗 Hop 列
            if 'hop' in df.columns:
                df['hop'] = pd.to_numeric(df['hop'], errors='coerce').fillna(1)
            else:
                df['hop'] = 1

            # 2. 计算激进权重 (Aggressive Weights)
            # 给 hop 加一个微小的底数防止 0 的情况，虽然一般最小是 1
            # 核心：权重 = hop ^ 4.5
            weights = np.power(df['hop'], WEIGHT_POWER)

            # 3. 加权采样 RL
            # 这样高 Hop 的数据有极大概率被选中
            df_rl = df.sample(frac=RL_RATIO, weights=weights, random_state=SEED)

            # 4. 剩下的归 SFT
            df_sft = df.drop(df_rl.index)

            # 5. 保存
            base_name = os.path.splitext(filename)[0]
            # 为了防止文件名太长，还是保持 _sft.csv 和 _rl.csv
            sft_path = os.path.join(output_dir, f"{base_name}_sft.csv")
            rl_path = os.path.join(output_dir, f"{base_name}_rl.csv")

            df_sft.to_csv(sft_path, index=False)
            df_rl.to_csv(rl_path, index=False)

            # 6. 统计对比
            # SFT
            s_mean = df_sft['hop'].mean()
            s_med = df_sft['hop'].median()
            s_max = df_sft['hop'].max()

            # RL
            r_mean = df_rl['hop'].mean()
            r_med = df_rl['hop'].median()
            r_max = df_rl['hop'].max()

            print(f"{base_name[:30]:<30} | SFT   | {len(df_sft):<6} | {s_mean:<8.2f} | {s_med:<8.1f} | {s_max:<8}")
            print(f"{'':<30} | RL    | {len(df_rl):<6} | {r_mean:<8.2f} | {r_med:<8.1f} | {r_max:<8}")
            print("-" * 95)

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    print(f"所有文件已保存至: {output_dir}")


if __name__ == "__main__":
    split_aggressive()
