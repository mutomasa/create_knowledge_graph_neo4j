import os
import sys
from pathlib import Path
import tempfile
from typing import List, Tuple, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network

# Streamlit 実行時に相対インポートが失敗しないように src を sys.path に追加
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from create_knowlege_graph_app.build_graph import (
    build_ner_pipeline,
    run_ner,
    main_verb_with_sudachi,
    ensure_indexes,
    upsert_entity,
    upsert_relation,
    SENT_SPLIT_RE,
)


def split_sentences(text: str) -> List[str]:
    return [s for s in SENT_SPLIT_RE.split(text.strip()) if s]


def color_for(etype: str) -> str:
    return {
        "PER": "#ff7f7f",
        "ORG": "#7fbfff",
        "LOC": "#7fff7f",
        "MISC": "#d1b3ff",
    }.get(etype or "", "#cccccc")


def render_pyvis_graph(
    extracted_rows: List[Dict[str, Any]],
    height_px: int = 900,
    width_px: int = 2000,
    use_physics: bool = True,
):
    # ノードメタ（タイプ・出現回数）集計
    node_meta: Dict[str, Dict[str, Any]] = {}
    for row in extracted_rows:
        for role in ("head", "tail"):
            name = row[role]
            etype = row[f"{role}_type"]
            if name not in node_meta:
                node_meta[name] = {"type": etype, "count": 0}
            node_meta[name]["count"] += 1
            if not node_meta[name].get("type") and etype:
                node_meta[name]["type"] = etype

    # 指定pxでキャンバス/コンポーネント幅を統一
    g = Network(height=f"{height_px}px", width=f"{width_px}px", directed=True, bgcolor="#111", font_color="#ddd")
    g.toggle_physics(use_physics)

    # 先にノード追加
    for name, meta in sorted(node_meta.items(), key=lambda x: (-x[1]["count"], x[0])):
        etype = meta.get("type") or ""
        count = int(meta.get("count", 1))
        size = min(12 + count * 2, 40)
        title = f"name: {name} | type: {etype} | count: {count}"
        g.add_node(name, label=name, title=title, color=color_for(etype), value=size)

    # エッジ追加
    for row in extracted_rows:
        a = row["head"]
        b = row["tail"]
        rel = row["rel"]
        sent = row.get("sentence", "")
        etitle = f"{rel} | {sent}" if sent else rel
        g.add_edge(a, b, label=rel, title=etitle)

    # オプション JSON
    options_json = """
{
  "nodes": {"shape": "dot", "font": {"size": 18}},
  "edges": {
    "arrows": {"to": {"enabled": true, "scaleFactor": 1.0}},
    "smooth": {"enabled": true, "type": "dynamic"},
    "font": {"size": 14, "align": "horizontal"}
  },
  "layout": {"improvedLayout": true}
}
    """
    try:
        g.set_options(options_json)
    except Exception:
        pass

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        # 依存をインライン展開して埋め込みの失敗を回避
        try:
            g.write_html(f.name, notebook=False, cdn_resources="in_line")  # pyvis>=0.3.2
        except TypeError:
            g.write_html(f.name, notebook=False)
        html = open(f.name, "r", encoding="utf-8").read()
    st.components.v1.html(html, width=width_px, height=height_px + 80, scrolling=True)
    # サイズの目安を表示
    st.caption(f"nodes: {len(node_meta)} | edges: {len(extracted_rows)}")


def main():
    load_dotenv()
    st.set_page_config(page_title="Japanese NER → Neo4j KG", layout="wide")
    st.title("日本語 NER / 関係抽出 → Neo4j 知識グラフ")

    with st.sidebar:
        st.header("Neo4j 接続")
        uri = st.text_input("NEO4J_URI", os.getenv("NEO4J_URI", "bolt://localhost:7687"))
        user = st.text_input("NEO4J_USER", os.getenv("NEO4J_USER", "neo4j"))
        password = st.text_input("NEO4J_PASSWORD", os.getenv("NEO4J_PASSWORD", "neo4j"), type="password")
        st.divider()
        st.header("NER モデル")
        model = st.selectbox(
            "Hugging Face モデル（失敗時はSudachiフォールバック）",
            [
                "ku-nlp/deberta-v2-base-japanese-ner",
                "izumi-lab/bert-base-japanese-ner",
                "taishi-i/bert-base-japanese-finetuned-ner",
            ],
            index=0,
        )
        st.divider()
        st.header("グラフ設定")
        graph_height = st.slider("高さ(px)", min_value=600, max_value=2200, value=1200, step=50)
        graph_width = st.slider("横幅(px)", min_value=800, max_value=3600, value=2200, step=100)
        graph_physics = st.checkbox("物理レイアウト(physics)", value=True)
        graph_hier = st.checkbox("横方向レイアウト(LR)", value=True)

    st.subheader("入力テキスト")
    default_text = ""
    sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "data", "sample.txt")
    sample_path = os.path.abspath(sample_path)
    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            default_text = f.read()
    except Exception:
        default_text = ""

    uploaded = st.file_uploader("テキストファイルをアップロード（任意）", type=["txt"])
    if uploaded is not None:
        text = uploaded.read().decode("utf-8")
    else:
        text = st.text_area("テキスト", value=default_text, height=220)

    col1, col2 = st.columns([1, 1])
    with col1:
        do_extract = st.button("抽出してNeo4jに投入")
    with col2:
        do_only_preview = st.button("抽出のみ（プレビュー）")

    if not (do_extract or do_only_preview):
        st.stop()

    with st.spinner("NERモデルを準備中..."):
        ner = build_ner_pipeline(model)

    sentences = split_sentences(text)
    st.write(f"文数: {len(sentences)}")

    pairs: List[Tuple[str, str, str]] = []
    extracted_rows = []
    for s in sentences:
        entities = run_ner(ner, s)
        if len(entities) < 2:
            continue
        rel = main_verb_with_sudachi(s)
        for i in range(len(entities) - 1):
            a = entities[i]
            b = entities[i + 1]
            pairs.append((a.text, b.text, rel))
            extracted_rows.append({
                "sentence": s,
                "head": a.text,
                "head_type": a.label,
                "rel": rel,
                "tail": b.text,
                "tail_type": b.label,
            })

    if not pairs:
        st.warning("関係を抽出できませんでした。テキストを増やすか別のサンプルをお試しください。")
        st.stop()

    st.subheader("抽出結果（先頭30件）")
    st.dataframe(extracted_rows[:30])

    st.subheader("グラフ可視化（pyvis）")
    render_pyvis_graph(
        extracted_rows,
        height_px=graph_height,
        width_px=graph_width,
        use_physics=graph_physics,
    )

    if do_extract:
        try:
            with st.spinner("Neo4j に投入中..."):
                driver = GraphDatabase.driver(uri, auth=(user, password))
                with driver.session() as session:
                    ensure_indexes(session)
                    seen_nodes = set()
                    for row in extracted_rows:
                        head = row["head"]
                        tail = row["tail"]
                        if head not in seen_nodes:
                            upsert_entity(session, head, row["head_type"])  # type: ignore[arg-type]
                            seen_nodes.add(head)
                        if tail not in seen_nodes:
                            upsert_entity(session, tail, row["tail_type"])  # type: ignore[arg-type]
                            seen_nodes.add(tail)
                    for row in extracted_rows:
                        upsert_relation(session, row["head"], row["tail"], row["rel"])  # type: ignore[arg-type]
                driver.close()
            st.success("Neo4j への投入が完了しました。")
            st.code(
                """
// 例: 直近に作成されたエンティティと関係
MATCH (e:Entity)-[r:RELATED]->(f:Entity)
RETURN e.name, r.rel, f.name
LIMIT 25;
""",
                language="cypher",
            )
        except Exception as e:
            st.error(f"Neo4j への投入でエラーが発生しました: {e}")


if __name__ == "__main__":
    main()


