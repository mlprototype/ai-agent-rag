# 🚀 Agentic RAG FastAPI デモンストレーション・ガイド

本ガイドは、面接や技術デモの際に、本リポジトリの最大の特徴である**「クエリ特性に応じた回答経路の動的制御（Control Plane）」**および**「RAGインジェスチョン・パイプライン」**を、FastAPIサーバーを通じてスムーズかつ効果的に実演するための手順書です。

---

## 🛠️ 1. 事前準備（環境構築）

デモを始める前に、以下の手順で環境をセットアップします。

### 1-1. パッケージのインストール
本プロジェクトではパッケージ管理に `uv` を使用しています。以下のコマンドで依存関係を同期します。
```bash
uv sync
```

### 1-2. 環境変数（.env）の設定
`.env.example` をコピーして `.env` を作成し、必要なAPIキーを設定します。
```bash
cp .env.example .env
```
> [!IMPORTANT]
> `.env` 内の `OPENAI_API_KEY` を必ず有効なキーに設定してください。

### 1-3. データベースの起動とシードデータの投入
ベクトル検索・全文検索用の PostgreSQL (pgvector) を Docker で起動し、初期データを投入します。
```bash
# データベースの起動
docker compose up -d

# サンプルドキュメントのシード（RAGの初期データ投入）
uv run python -m infrastructure.retrieval.vector_store
```

---

## ⚡ 2. FastAPIサーバーの起動

FastAPI アプリケーションサーバーをローカルで起動します。起動時にローカルにキャッシュされたプロンプトの読み込み確認（Prewarm）が実行されます。
```bash
uv run uvicorn api.main:app --reload
```
サーバーが起動すると、 `http://localhost:8000` で API が利用可能になります。
ブラウザで [http://localhost:8000/docs](http://localhost:8000/docs) にアクセスすると、Swagger UI からインタラクティブに API を試すことも可能です。

---

## 🎬 3. クイックデモ（全自動スクリプトの実行）

面接の場などで、個別のcurlコマンドを叩く時間がない場合は、**デモ自動実行スクリプト**を実行してください。
各種ルーティング（挨拶、SQLite集計、比較、定義、新規ファイルの取り込みとそれに基づく回答）を順番に実行し、結果を綺麗に色分けして出力します。

FastAPIサーバーを起動した状態で、**別のターミナル**で以下を実行します。
```bash
uv run python scripts/demo_requests.py
```

### 💡 スクリプト実行でデモされるシナリオ
1. **挨拶・雑談 (Direct Answer)**: Heuristic Router が即座に検知し、LLM APIなしでダイレクト応答。
2. **構造化データの集計 (Structured Query)**: 売上データなどの集計意図を解釈し、SQLite上でSQLを発行して正確な数値を回答。
3. **比較クエリ (Compare Fast-Path)**: 2つの比較対象を抽出し、並列検索を行った上で共通点・相違点を構造化して回答。
4. **定義・用語説明 (Definition)**: pgvectorなどの技術用語の解説をRAGから検索して回答。
5. **インジェスチョン（ファイル取り込み）**: 新規のMarkdownファイルをAPI経由でアップロード・パース・ベクトルDB登録。
6. **追加ナレッジ検索**: 取り込んだ新規ドキュメント（カレーライスの作り方）について質問し、RAGが正しく回答できるかを実演。

---

## ⌨️ 4. 手動デモ用：主要クエリ（curlコマンド一覧）

デモの対話性を見せたい場合や、特定の機能を深掘りして見せたい場合は、以下の `curl` コマンドを使用してください。

### 経路①: 挨拶・雑談（Direct Answer 経路）
- **クエリ**: `こんにちは！自己紹介をしてください。`
- **アピールポイント**: 20文字以下の挨拶などの定型文は、Heuristic Routerによって **LLM Router呼び出し（コストと遅延）をスキップ** し、高速に応答します。
```bash
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"session_id":"demo-001","question":"こんにちは！自己紹介をしてください。"}' \
  http://localhost:8000/ask | python -m json.tool
```

### 経路②: 構造化データの集計（Structured Query 経路）
- **クエリ**: `2025年Q1の売上合計を教えてください。`
- **アピールポイント**: RAGの検索チャンクから数値を集計すると、数値の欠落や二重カウントが起きます。本システムでは、売上や在庫に対する問い合わせ意図をパースし、**SQLiteの読み取り専用（SELECT）クエリとして決定論的に集計**します。安全対策として、多重実行（`;`）や破壊的操作（`INSERT`, `UPDATE` 等）はバリデーターで厳格にブロックされます。
```bash
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"session_id":"demo-003","question":"2025年Q1の売上合計を教えてください。"}' \
  http://localhost:8000/ask | python -m json.tool
```

### 経路③: 比較クエリ（Compare Fast-Path 経路）
- **クエリ**: `FastAPIとDjangoの違いは何ですか？`
- **アピールポイント**: AとBの比較クエリに対し、通常のRAGでは情報が混ざったり片方しか検索できなかったりします。本システムでは `FastAPI` と `Django` を比較対象として抽出し、**2つのクエリで並列検索（`asyncio.gather`）**をかけ、両者のカバレッジを確認した上で、比較特化のプロンプトを用いて回答を合成します。
```bash
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"session_id":"demo-004","question":"FastAPIとDjangoの違いは何ですか？"}' \
  http://localhost:8000/ask | python -m json.tool
```

### 経路④: 定義・用語説明（Definition 経路）
- **クエリ**: `pgvectorとは何ですか？`
- **アピールポイント**: データベースに登録されているドキュメントから、用語の定義を検索・要要約します。検索結果には **ハイブリッド検索スコア**（Vector + Keyword）、確信度（Confidence）およびドキュメントソースのメタデータが正確に付与されます。
```bash
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"session_id":"demo-005","question":"pgvectorとは何ですか？"}' \
  http://localhost:8000/ask | python -m json.tool
```

### 経路⑤: ドキュメントのアップロード＆追加質問（Ingestion 経路）
- **動作**: `sample_ingest.md`（カレーライスの作り方が書かれたテストファイル）をAPI経由でアップロードし、直後に追加質問を行います。
- **アピールポイント**: 静的なデータだけでなく、**動的なファイル取り込みパイプライン（Semantic Chunkingによる意味的な分割 -> ベクトル化 -> DB保存）がFastAPI経由でエンドツーエンドで稼働していること**を実演できます。
```bash
# 1. ファイルをアップロードしてインジェスト
curl -X POST -F "file=@./sample_ingest.md" \
  http://localhost:8000/ingest/file

# 2. 追加されたナレッジに対して質問
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"session_id":"demo-006","question":"カレーライスを美味しく作る秘訣は何ですか？"}' \
  http://localhost:8000/ask | python -m json.tool
```

---

## 🎯 5. 面接でアピールするための技術的ハイライト

デモの実行中、面接官に対して以下のようなシステム設計のこだわりを説明すると、高い評価を得られます。

| 機能 / 経路 | 解決する課題 | 実装アプローチ | アピールポイント |
| :--- | :--- | :--- | :--- |
| **Heuristic Router** | 明確な意図の判定にもLLM Routerが走り、APIコストとレイテンシが増える | キーワードによる先行分岐（挨拶・業務集計・比較・明確な定義クエリ） | LLM Routerを呼び出す前の段階で高速に分類し、不要な判定API呼び出しを削減。 |
| **Structured Query** | RAGが最も苦手とする「データのカウント」や「合計・平均」といった集計計算のハルシネーション | 質問意図からメトリクスとフィルターをパースし、SQLite上のSQLクエリへ変換 | **数値の確実性**。LLMの推論に頼らず、データベース上のSELECT集計で確実な数値を返す。SQLインジェクションや破壊的操作はValidatorで厳格に遮断。 |
| **Compare Fast-Path** | 「AとBの比較」において、検索結果が片方に偏ったり、混同して誤った比較表を生成する問題 | 正規表現による比較対象A・Bの抽出と、**`asyncio.gather` による非同期並列検索** | **カバレッジの担保**。AとBそれぞれの検索結果が閾値以上存在すること（カバレッジ）を確認し、専用テンプレートで対比構造の明確な回答を出力。 |
| **Strict RAG Policy** | ナレッジベースにない情報について、LLMが一般知識で「もっともらしい嘘」を答えてしまう | プロンプトでの厳格な「一般知識回答の禁止」および「Low-Confidence Guard」 | **企業向けの実用性**。根拠ドキュメントの確信度が低い場合は、推測で答えず「見つかりませんでした」と返答させることで、ハルシネーションを極小化。 |
| **Budget & Fallback** | 外部APIの遅延や障害時、あるいは複雑なクエリの再試行ループにより、全体のタイムアウトやUX悪化が起きる | 15,000msの初期予算を設定し、各工程（Rerank, Critic, Rewrite）の残り時間に応じた**動的スキップ（段階的縮退）** | **システムの堅牢性（可観測性・制御性）**。残り予算を監視し、時間が足りない場合はCriticやRerankを自動スキップして最低限の回答を制限時間内に死守する。 |
