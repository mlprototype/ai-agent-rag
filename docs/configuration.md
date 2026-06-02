<!--
ファイルの責務:
- 本システム（Agentic RAG with Control Plane）における環境変数と設定項目の一覧をドキュメント化する。
- 開発・検証環境向けの設定から、本番運用時のチューニングポイントまで、各種しきい値やタイムアウトのデフォルト値とその意味を一覧で示す。
- 注意点: デフォルト値はローカル検証用であるため、本番適用時にはシステム負荷やレイテンシ、APIコストを踏まえた調整が必要である。
-->

# 設定項目一覧

本システムでは、各種機能の切り替え、閾値、タイムアウト、および外部サービス接続を環境変数で制御しています。

表のデフォルト値は原則としてコード既定値です。`.env.example` には主要キーのみが記載され、一部はサンプル用に上書きされています（例: `ROUTER_BUDGET_MS=500`）。また `HYBRID_*` / `TOP_K_*` / `STAGE_TIMEOUT_MS_REWRITE|HYBRID|COMPRESS` は各モジュールが環境変数を直接参照します。

## 環境変数一覧

| 環境変数 | デフォルト | 説明 |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | - | OpenAI API キー |
| `LANGSMITH_API_KEY` | - | LangSmith Hub / tracing 用 API キー |
| `LANGSMITH_WORKSPACE_ID` | - | org/private prompt sync で workspace header が必要な場合に使用 |
| `LANGCHAIN_TRACING_V2` | `true` | LangSmith トレーシング有効化 |
| `LANGCHAIN_ENDPOINT` | `https://api.smith.langchain.com` | LangSmith endpoint |
| `LANGCHAIN_PROJECT` | `ai-agent-rag` | LangSmith project 名 |
| `COHERE_API_KEY` | - | Cohere API キー (ENABLE_RERANK=true時に必須) |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` / `POSTGRES_HOST` / `POSTGRES_PORT` | `admin` / `password` / `rag_db` / `localhost` / `5432` | PostgreSQL / pgvector 接続設定 |
| **Search Pipeline** | | |
| `HYBRID_ALPHA` | `0.6` | Hybrid Search における vector score の重み |
| `RETRIEVE_K_VECTOR` | `30` | Vector Search の初期取得件数 |
| `RETRIEVE_K_KEYWORD` | `30` | Keyword Search の初期取得件数 |
| `TOP_K_MIN` | `3` | 高信頼時の最小採用件数 |
| `TOP_K_MAX` | `8` | 低信頼時の最大採用件数 |
| `STAGE_TIMEOUT_MS_REWRITE` | `3000` | RetrievalService 内 Query Rewrite のタイムアウト |
| `STAGE_TIMEOUT_MS_HYBRID` | `5000` | Hybrid Search 全体のタイムアウト |
| `STAGE_TIMEOUT_MS_COMPRESS` | `5000` | Extractive Compression のタイムアウト |
| **Agentic Control Plane** | | |
| `ENABLE_AGENTIC` | `true` | Agentic RAG (Loop/Critic) の有効化 |
| `ENABLE_RERANK` | `false` | Reranker (Cohere) の有効化 |
| `ROUTER_HEURISTIC_ENABLED` | `true` | ルールベース Router の有効化 |
| `ROUTER_HEURISTIC_COMPARE_ENABLED` | `true` | Compare ヒューリスティックの有効化 |
| `ROUTER_HEURISTIC_CONFIDENCE_THRESHOLD_PCT` | `85` | Heuristic 判定を採用する信頼度しきい値 |
| `ROUTER_BUDGET_MS` | `1500` | Router に割り当てる予算（`.env.example` では `500`） |
| `ROUTER_UNCERTAIN_CONFIDENCE_CAP_PCT` | `80` | router uncertain 時の confidence 上限 |
| `ANSWER_CRITIC` | `true` | Answer Critic による回答検証の有効化 |
| `ANSWER_CRITIC_RETRY` | `false` | Answer Critic FAIL時に再検索を試行するか |
| `MAX_RETRY` | `3` | Agentic Loop の最大リトライ回数 |
| `MAX_SUB_QUERIES` | `4` | Decomposition 時の最大 Sub-query 数 |
| `MAX_MERGED_CHUNKS` | `20` | マージ・Rerank 後の最終チャンク上限数 |
| **Prompt Ops** | | |
| `PROMPT_NAMESPACE` | `my-rag` | LangSmith Hub 同期時に使う namespace |
| `PROMPT_LOAD_TIMEOUT_MS` | `1200` | prompt loader 向けの予約済み設定値（現状は `Settings` に保持） |
| `PROMPT_FAILURE_TTL_MS` | `30000` | prompt failure TTL 向けの予約済み設定値（現状は `Settings` に保持） |
| `PREWARM_FAIL_FAST` | `true` | 起動時の prompt prewarm 失敗でプロセスを落とすか |
| **Budget & Critic** | | |
| `COMPLEX_BUDGET_MS_LOW` | `4000` | 複雑度 Low クエリの初期予算 (ms) |
| `COMPLEX_BUDGET_MS_MEDIUM` | `7000` | 複雑度 Medium クエリの初期予算 (ms) |
| `COMPLEX_BUDGET_MS_HIGH` | `9000` | 複雑度 High クエリの初期予算 (ms) |
| `BUDGET_TOTAL_RETRIEVAL_COMPLEX_MS` | `15000` | `retrieval_complex` 用の最大予算 (ms) |
| `BUDGET_RESERVED_GENERATE_MS` | `3000` | Generate ノード用に予約する予算 (ms) |
| `BUDGET_RESERVED_COMMIT_MS` | `500` | Commit ノード用に予約する予算 (ms) |
| `BUDGET_MIN_FOR_RERANK_MS` | `1000` | Rerank 実行に必要な最低 usable 予算 (ms) |
| `BUDGET_MIN_FOR_CRITIC_MS` | `1500` | Critic 実行に必要な最低 usable 予算 (ms) |
| `BUDGET_MIN_FOR_REWRITE_MS` | `3000` | Rewrite 実行に必要な最低 usable 予算 (ms) |
| `RETRIEVAL_DEGRADE_THRESHOLD_MS` | `2000` | 残り予算がこれを下回ると Rerank/Rewrite 等をスキップ |
| `FORCE_GENERATE_THRESHOLD_MS` | `1500` | 残り予算がこれを下回ると強制的に Generate へ遷移 |
| `RETRIEVAL_CRITIC_SKIP_CONFIDENCE_PCT` | `85` | 確信度が 85% 超の場合に Retrieval Critic をスキップ |
| `ANSWER_CRITIC_SKIP_CONFIDENCE_PCT` | `80` | 確信度が 80% 超の場合に Answer Critic をスキップ |
| **Timeouts (ms)** | | |
| `STAGE_TIMEOUT_MS_ROUTER` | `1800` | Router ノードのタイムアウト |
| `STAGE_TIMEOUT_MS_RETRIEVAL_CRITIC` | `2500` | Retrieval Critic のタイムアウト |
| `STAGE_TIMEOUT_MS_ANSWER_CRITIC` | `2500` | Answer Critic のタイムアウト |
| `STAGE_TIMEOUT_MS_DECOMPOSE` | `1800` | Decompose ノードのタイムアウト |
| `STAGE_TIMEOUT_MS_REWRITE_SUBQUERY` | `1800` | Rewrite ノードのタイムアウト |
| `STAGE_TIMEOUT_MS_RERANK` | `2500` | Reranker のタイムアウト |
