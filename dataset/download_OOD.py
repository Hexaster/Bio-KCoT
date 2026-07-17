import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

# 替换为您本地文件的实际路径
file_path = str(path("paths.ood_biomaze") / "Openended-00000-of-00001.parquet")

# 读取本地数据
df = pd.read_parquet(file_path)

# 随机抽取 200 行 (random_state 保证可复现)
df_sampled = df.sample(n=200, random_state=42)

# 保存为新的 JSON 文件供模型测试使用
output_path = str(path("paths.ood_biomaze") / "biomaze_sampled_200.json")
df_sampled.to_json(output_path, orient="records", force_ascii=False, indent=4)

print(f"✅ 成功从本地文件随机抽取 200 条数据，并保存至 {output_path}")
