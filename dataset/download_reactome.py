import csv
import sys
from pathlib import Path
from neo4j import GraphDatabase

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "config.json").exists())))
from biokcot_config import env, get, path

# 配置信息
URI = env("NEO4J_URI", get("neo4j.uri"))
AUTH = (env("NEO4J_USERNAME", get("neo4j.username")), env("NEO4J_PASSWORD", required=True))
OUTPUT_FILE = str(path("paths.data_dir") / "reactome_full_knowledge.csv")


def export_rich_data(tx):
    # 这个查询有点大，因为它要抓取 5 个维度的信息
    query = """
    MATCH (p:Pathway)-[:hasEvent]->(r:ReactionLikeEvent)
    WHERE p.speciesName = 'Homo sapiens' 
      AND r.speciesName = 'Homo sapiens'
      AND 'Reaction' IN labels(r)

    // --- 1. 基础物质流 (Inputs, Outputs, Catalysts) ---
    OPTIONAL MATCH (r)-[:input]->(i:PhysicalEntity)
    OPTIONAL MATCH (r)-[:output]->(o:PhysicalEntity)
    OPTIONAL MATCH (r)-[:catalystActivity]->(:CatalystActivity)-[:physicalEntity]->(cat:PhysicalEntity)

    // --- 2. 时序逻辑 (Preceding Events) ---
    // 查找谁是这一步的前置步骤
    OPTIONAL MATCH (prev:ReactionLikeEvent)-[:precedingEvent]->(r)

    // --- 3. 调节因子 (Regulation) ---
    // 查找谁在调节这个反应，并获取调节类型 (Positive/Negative)
    OPTIONAL MATCH (reg:Regulation)-[:regulatedBy]->(r)
    OPTIONAL MATCH (reg)-[:regulator]->(regulator:PhysicalEntity)

    // --- 4. 复合物拆解 (Complex Breakdown) ---
    // 如果输入或输出是复合物，查找它们的组成成分 (只向下拆解一层，防止数据爆炸)
    OPTIONAL MATCH (r)-[:input|output]->(comp:Complex)-[:hasComponent]->(sub:PhysicalEntity)

    // --- 5. 文本摘要 (Summation) ---
    // 优先取反应自己的摘要，如果没有，取通路的摘要作为背景
    OPTIONAL MATCH (r)-[:summation]->(r_sum:Summation)
    OPTIONAL MATCH (p)-[:summation]->(p_sum:Summation)

    RETURN 
        p.stId AS Pathway_ID,
        p.displayName AS Pathway_Name,

        r.stId AS Reaction_ID,
        r.displayName AS Reaction_Name,

        // 文本：优先用反应的，没有就用通路的
        COALESCE(r_sum.text, p_sum.text, '') AS Summation,

        // 聚合基础物质
        collect(DISTINCT i.displayName) AS Inputs,
        collect(DISTINCT o.displayName) AS Outputs,
        collect(DISTINCT cat.displayName) AS Catalysts,

        // 聚合逻辑信息
        collect(DISTINCT prev.displayName) AS Preceding_Events,

        // 聚合调节信息 (格式: "RegulatorName (PositiveRegulation)")
        collect(DISTINCT regulator.displayName + ' (' + head(labels(reg)) + ')') AS Regulators,

        // 聚合复合物成分
        collect(DISTINCT sub.displayName) AS Complex_Components
    ORDER BY p.displayName, r.displayName
    """

    # 因为查询较重，设置较大的超时时间并非坏事，但 execute_read 默认会自动处理
    print("⏳ 正在 Neo4j 中执行全量逻辑提取，数据量较大，请耐心等待...")
    result = tx.run(query)

    # 转换为列表返回，避免事务关闭后丢失
    return [record for record in result]


def main():
    driver = GraphDatabase.driver(URI, auth=AUTH)

    try:
        print(f"正在连接 Neo4j ({URI})...")
        with driver.session() as session:
            records = session.execute_read(export_rich_data)

            print(f"✅ 查询完成！共获取 {len(records)} 条反应数据。")
            print(f"💾 正在写入 CSV: {OUTPUT_FILE} ...")

            with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # 写入表头 (这一行包含了你要的所有丰富信息)
                header = [
                    'Pathway_ID', 'Pathway_Name',
                    'Reaction_ID', 'Reaction_Name',
                    'Summary (Text)',  # 文本描述
                    'Inputs', 'Outputs', 'Catalysts',
                    'Preceding_Events (Order)',  # 时序
                    'Regulators (Control)',  # 调节
                    'Complex_Components (Detail)'  # 成分
                ]
                writer.writerow(header)

                count = 0
                for row in records:
                    # 辅助函数：清洗文本 (去掉换行符，把列表拼成字符串)
                    def clean_list(lst):
                        if not lst: return ""
                        # 过滤掉 None，用分号连接
                        return "; ".join([str(x) for x in lst if x])

                    def clean_text(txt):
                        if not txt: return ""
                        # 把换行符替换成空格，防止破坏 CSV 格式
                        return txt.replace('\n', ' ').replace('\r', ' ')

                    writer.writerow([
                        row['Pathway_ID'],
                        row['Pathway_Name'],
                        row['Reaction_ID'],
                        row['Reaction_Name'],
                        clean_text(row['Summation']),  # 清洗摘要
                        clean_list(row['Inputs']),
                        clean_list(row['Outputs']),
                        clean_list(row['Catalysts']),
                        clean_list(row['Preceding_Events']),
                        clean_list(row['Regulators']),  # 这一列会显示 "ProteinX (NegativeRegulation)"
                        clean_list(row['Complex_Components'])
                    ])
                    count += 1

        print(f"🎉 成功！所有 {count} 条数据已保存。现在你有了一份“知识密度”极高的 CSV！")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
