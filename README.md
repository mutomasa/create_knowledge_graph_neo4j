### 概要
日本語の無料テキストから NER（固有表現抽出）と簡易関係抽出を行い、Neo4j に Cypher で知識グラフを構築します。NER は Hugging Face の日本語対応モデル、関係抽出は Sudachi による動詞抽出に基づく簡易ヒューリスティクスです。

### セットアップ（uv）
- 依存関係の同期
```
uv sync
```
- Neo4j 設定（環境変数）
```
cp .env.example .env
# .env を開いて NEO4J_USER / NEO4J_PASSWORD を設定
```

### 使い方
- サンプルテキストで実行
```
uv run python -m create_knowlege_graph_app.build_graph --input data/sample.txt
```
- 独自テキストで実行
```
uv run python -m create_knowlege_graph_app.build_graph --input /path/to/your_japanese_text.txt
```

### Streamlit アプリ（可視化付き）
- 起動
```
uv run streamlit run src/create_knowlege_graph_app/streamlit_app.py --server.port 8506
```
- 機能
  - **アップロード/テキスト入力**: 日本語テキストを入力
  - **抽出のみ（プレビュー）**: 画面上で抽出結果とグラフを可視化
  - **抽出してNeo4jに投入**: Neo4j にノード/リレーションを作成
  - **pyvis**: 画面上にインタラクティブなグラフを描画


### 仕組み
- NER: `ku-nlp/deberta-v2-base-japanese-ner` を使用（Transformers pipeline で集約）
- 関係抽出: 文内のエンティティ対に対し、Sudachi で見つかった主要動詞（原形）を `rel` として付与。動詞が見つからない場合は `related_to`。
- Neo4j: ノード `:Entity{name, type}` とリレーション `[:RELATED{rel}]` を作成し、簡易な索引を作成。

### Neo4j 接続
- 既定: `bolt://localhost:7687`
- 認証: `.env` の `NEO4J_USER`, `NEO4J_PASSWORD`

### 注意
- 関係抽出は簡易ヒューリスティクスであり、厳密な依存構造に基づく関係抽出ではありません。高精度が必要な場合は、日本語依存解析（GiNZA など）や日本語REモデルへの差し替えをご検討ください。


