import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import path

# 1. 定义文件路径
score_dir = path("paths.data_dir") / "score"
file_paths = [str(score_dir / filename) for filename in [
    "disease-indication-drug_synthesized_scored.csv",
    "drug-synergy-drug_scored.csv",
    "PPI_reasoning_questions_scored.csv",
    "reactome_reasoning_dataset_scored.csv",
]]

# 2. 定义子图标题
titles = [
    "Mechanism of drug treatment",
    "Mechanism of drug-drug interactions",
    "Mechanism of protein-protein interactions",
    "Mechanism of reactome"
]


def plot_score_distribution():
    # 全局字体族设置
    plt.rcParams.update({'font.family': 'sans-serif'})

    # --- 调整画布大小 ---
    # 保持上次修改：(16, 14) 比较适中
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 14))
    axes = axes.flatten()

    for i, file_path in enumerate(file_paths):
        ax = axes[i]
        print(f"正在处理: {os.path.basename(file_path)}")

        try:
            # 读取数据
            df = pd.read_csv(file_path)

            if 'score' not in df.columns:
                ax.text(0.5, 0.5, 'No "score" column', ha='center', va='center', fontsize=20)
                ax.set_title(titles[i], fontsize=22, fontweight='bold')
                continue

            df['score'] = pd.to_numeric(df['score'], errors='coerce')

            # --- 核心修改：移除 1 分 ---
            # range(10, 1, -1) 意味着包含 10, ..., 2 (不包含 1)
            target_order = range(10, 1, -1)
            score_counts = df['score'].value_counts().reindex(target_order, fill_value=0)

            # 绘制柱状图
            bars = ax.bar(
                [str(x) for x in score_counts.index],
                score_counts.values,
                color='#4c72b0',
                edgecolor='black',
                linewidth=2.0,
                alpha=0.85
            )

            # --- 样式设置 (大字体+粗体) ---

            # 1. 标题
            ax.set_title(titles[i], fontsize=22, fontweight='bold', pad=20)

            # 2. 坐标轴标签
            ax.set_xlabel('Score', fontsize=20, fontweight='bold', labelpad=12)
            ax.set_ylabel('Count', fontsize=20, fontweight='bold', labelpad=12)

            # 3. 坐标轴刻度 (数值)
            ax.tick_params(axis='both', which='major', labelsize=18, width=2.5, length=8)

            # 强制设置刻度标签为粗体
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontweight('bold')

            # 4. 设置实线边框
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2.0)
                spine.set_edgecolor('black')

            # 5. 网格线
            ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=1.5)

            # 6. 柱子上方数值
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{int(height)}',
                        ha='center',
                        va='bottom',
                        fontsize=16,
                        fontweight='bold',
                        color='black'
                    )

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            ax.text(0.5, 0.5, "Data Load Error", ha='center', va='center', fontsize=20, color='red')

    # 紧凑布局
    plt.tight_layout(pad=3.0)

    # --- 保存文件 (PDF 和 PNG) ---
    base_filename = path("paths.pictures") / "score_distribution_large_font"

    # 1. 保存 PDF
    pdf_filename = f'{base_filename}.pdf'
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')

    # 2. 保存 PNG (增加 dpi 以保证大图清晰)
    png_filename = f'{base_filename}.png'
    plt.savefig(png_filename, format='png', bbox_inches='tight', dpi=300)

    print(f"\n绘图完成，已导出:")
    print(f"- PDF: {pdf_filename}")
    print(f"- PNG: {png_filename}")
    # plt.show()


if __name__ == "__main__":
    plot_score_distribution()
