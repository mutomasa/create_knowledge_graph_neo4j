import os
import re
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from transformers import pipeline
from sudachipy import dictionary, tokenizer as sudachi_tokenizer
from tqdm import tqdm


SENT_SPLIT_RE = re.compile(r"[。！？]\s*")


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int


def load_texts(input_path: str) -> List[str]:
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    # 日本語の簡易文分割
    sentences = [s for s in SENT_SPLIT_RE.split(text) if s]
    return sentences


def build_ner_pipeline(model_name: str = "ku-nlp/deberta-v2-base-japanese-ner"):
    candidates = [
        model_name,
        "izumi-lab/bert-base-japanese-ner",
        "taishi-i/bert-base-japanese-finetuned-ner",
    ]
    for m in candidates:
        try:
            return pipeline(
                "token-classification",
                model=m,
                aggregation_strategy="simple",
            )
        except Exception:
            continue

    # HFモデルが使えない場合のフォールバック（Sudachiベースの簡易NER）
    tokenizer_obj = dictionary.Dictionary().create()

    def _fallback(sentence: str):
        tokens = tokenizer_obj.tokenize(sentence, mode=sudachi_tokenizer.Tokenizer.SplitMode.C)
        results = []
        i = 0
        current_text = ""
        current_label = None
        start_idx = 0

        # 文字オフセット追跡用のポインタ
        ptr = 0
        for t in tokens:
            surf = t.surface()
            pos = t.part_of_speech()
            # 文中の surf の開始位置を ptr 以降で見つける
            found = sentence.find(surf, ptr)
            if found < 0:
                # 見つからない場合はスキップ
                continue
            this_start = found
            this_end = found + len(surf)

            is_ne = bool(pos and len(pos) >= 2 and pos[0] == "名詞" and pos[1] == "固有名詞")
            # ラベル推定
            label = None
            if is_ne:
                details = pos[2:] if len(pos) > 2 else []
                detail_str = "|".join([d for d in details if d])
                if "人名" in detail_str:
                    label = "PER"
                elif "組織" in detail_str:
                    label = "ORG"
                elif "地名" in detail_str:
                    label = "LOC"
                else:
                    label = "MISC"

            if is_ne:
                if current_label is None:
                    current_text = surf
                    current_label = label
                    start_idx = this_start
                else:
                    # 連結（隣接する固有名詞）
                    current_text += surf
                ptr = this_end
            else:
                if current_label is not None:
                    results.append({
                        "word": current_text,
                        "entity_group": current_label,
                        "start": start_idx,
                        "end": ptr,
                    })
                    current_text = ""
                    current_label = None
                ptr = this_end

        if current_label is not None:
            results.append({
                "word": current_text,
                "entity_group": current_label,
                "start": start_idx,
                "end": ptr,
            })

        return results

    class _FallbackNER:
        def __call__(self, sentence: str):
            return _fallback(sentence)

    return _FallbackNER()


def run_ner(ner, sentence: str) -> List[Entity]:
    results = ner(sentence)
    entities: List[Entity] = []
    for r in results:
        # r: {entity_group, score, word, start, end}
        entities.append(Entity(text=r["word"], label=r["entity_group"], start=int(r["start"]), end=int(r["end"])) )
    # 位置順に
    entities.sort(key=lambda e: e.start)
    return entities


def main_verb_with_sudachi(sentence: str) -> str:
    # Sudachi で品詞を見て最初の動詞(基本形)を返す
    tokenizer_obj = dictionary.Dictionary().create()
    tokens = tokenizer_obj.tokenize(sentence, mode=sudachi_tokenizer.Tokenizer.SplitMode.C)
    for m in tokens:
        pos = m.part_of_speech()
        if pos and pos[0] == "動詞":
            base = m.dictionary_form()
            if base:
                return base
            return m.surface()
    return "related_to"


def ensure_indexes(session):
    session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")


def upsert_entity(session, name: str, etype: str) -> None:
    session.run(
        "MERGE (e:Entity {name: $name}) SET e.type = coalesce(e.type, $type)",
        name=name, type=etype
    )


def upsert_relation(session, a: str, b: str, rel: str) -> None:
    session.run(
        """
        MATCH (ea:Entity {name: $a})
        MATCH (eb:Entity {name: $b})
        MERGE (ea)-[r:RELATED {rel: $rel}]->(eb)
        """,
        a=a, b=b, rel=rel
    )


def ingest(sentences: List[str], uri: str, user: str, password: str, model_name: str) -> None:
    ner = build_ner_pipeline(model_name)
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        ensure_indexes(session)
        for s in tqdm(sentences, desc="ingest"):
            entities = run_ner(ner, s)
            if len(entities) < 2:
                continue
            # ノード作成
            for ent in entities:
                upsert_entity(session, ent.text, ent.label)
            # 関係作成（隣接対 + 代表動詞）
            rel = main_verb_with_sudachi(s)
            for i in range(len(entities) - 1):
                a = entities[i].text
                b = entities[i + 1].text
                upsert_relation(session, a, b, rel)
    driver.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/sample.txt", help="日本語テキストのパス")
    p.add_argument("--model", type=str, default="ku-nlp/deberta-v2-base-japanese-ner", help="Hugging Face NER モデル名")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    sentences = load_texts(args.input)
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")
    ingest(sentences, uri, user, password, args.model)


if __name__ == "__main__":
    main()


