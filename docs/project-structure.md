<!--
ファイルの責務:
- 本システム（Agentic RAG with Control Plane）のディレクトリ構造およびファイル単位の構成を詳細に記述する。
- 開発者向けに、どのディレクトリ・ファイルが何の責務（Domain, Application, API, Infrastructure, Adapters等）を担っているかを示す。
- 注意点: 新規ファイル追加やリファクタリング時に、本構造に従って適切なレイヤーへファイルを配置すること。
-->

# プロジェクトのディレクトリ構成

本システムの全体ファイル・ディレクトリ構成は以下の通りです。

```
ai-agent-rag/
├── main.py                                  # CLI エントリーポイント
├── pyproject.toml                           # プロジェクト定義 (uv)
├── docker-compose.yml                       # pgvector コンテナ定義
├── .env.example                             # 環境変数テンプレート
├── sample_ingest.md                         # 取り込み確認用サンプル文書
├── data/                                    # --- Persistent Data ---
│   └── structured_query.db                  #   SQLite 実データベース（売上・在庫等）
│
│
├── config/                                  # --- Configuration Layer ---
│   └── settings.py                          #   環境変数・レイテンシ予算・しきい値設定
│
├── api/                                     # --- API / Interface Layer ---
│   ├── main.py                              #   FastAPI アプリケーション (v3.0.0, prompt prewarm付き)
│   └── routers/
│       └── ingest.py                        #   POST /ingest/file, /ingest/directory
│
├── application/                             # --- Application Layer ---
│   ├── agents/
│   │   ├── graph.py                         #   LangGraph ステートグラフ定義 + Control Plane
│   │   └── state.py                         #   AgentState定義 (メタデータ・予算情報保持)
│   ├── dto/
│   │   └── chat_models.py                   #   ChatRequest / ChatResponse / Source
│   ├── interfaces/
│   │   └── conversation_memory.py           #   ConversationMemory ABC (DIP)
│   └── services/
│       └── chat_service.py                  #   ユースケース + Citation抽出 + Confidence/Warning整形
│
├── domain/                                  # --- Domain Layer ---
│   ├── models/
│   │   └── retrieval_models.py              #   RetrievedChunk / RewriteResult / CompressionResult
│   └── services/
│       ├── router.py                        #   経路ルーティング (Heuristic + LLM Facade)
│       ├── heuristic_router.py              #   Heuristic分類ルール
│       ├── retrieval_budget.py              #   レイテンシ予算管理 + 段階的縮退判定
│       ├── retrieval_service.py             #   5ステージ検索パイプライン
│       ├── query_decomposer.py              #   Decompose / Rewrite による sub-query 計画
│       ├── result_merger.py                 #   Max Pooling Merge (検索結果統合)
│       ├── retrieval_critic.py              #   Retrieval Critic (検索結果の品質評価)
│       ├── answer_critic.py                 #   Answer Critic (最終回答の検証)
│       ├── compare_intent.py                #   Compare: 正規表現による A/B 抽出
│       ├── compare_retrieval.py             #   Compare: subquery builder + 並列検索
│       ├── compare_merge.py                 #   Compare: コンテキスト統合 + coverage 判定
│       ├── compare_quality_gate.py          #   Compare 品質評価ユーティリティ（現状 graph 未接続）
│       ├── coverage_checker.py              #   回答の網羅性チェック (entity/axis)
│       ├── confidence.py                    #   Confidence 算出 + Dynamic TopK
│       ├── prompt_loader.py                 #   Prompt Ops (local snapshot 読込 / prewarm log)
│       ├── prompt_registry.py               #   Prompt 定義レジストリ
│       ├── prompt_formats.py                #   YAML ↔ ChatPromptTemplate 変換
│       ├── prompt_sync.py                   #   LangSmith Hub → local transactional sync
│       ├── query_rewriter.py                #   LLM クエリ書き換え
│       ├── hybrid_search.py                 #   Vector + Keyword 統合検索
│       ├── compressor.py                    #   Extractive Compression
│       ├── expression_evaluator.py          #   決定論的な数式評価ユーティリティ
│       ├── structured_query.py              #   Structured Query オーケストレーション
│       ├── structured_query_validator.py    #   SQL インジェクション・セマンティック検証
│       ├── structured_query_sql_builder.py  #   SQLite 用 SQL 生成ロジック
│       ├── sqlite_structured_query.py       #   SQLite 実実行アダプタ (Read-only)
│       ├── structured_query_formatter.py    #   実行結果の自然言語整形
│       ├── structured_query_types.py        #   構造化クエリ用型定義
│       ├── structured_query_datasets.py     #   デモ用テーブル定義・初期データ
│       └── ingestion_service.py             #   取り込みオーケストレーション
│
├── adapters/                                # --- Adapters Layer ---
│   └── tools/
│       ├── retrieval_tool.py                #   LangChain @tool ラッパー（検索）
│       └── calculator.py                    #   LangChain @tool ラッパー（計算）
│
├── infrastructure/                          # --- Infrastructure Layer ---
│   ├── ingestion/
│   │   └── unstructured_loader.py           #   unstructured パーサー (MD/HTML/TXT)
│   ├── memory/
│   │   └── in_memory_memory.py              #   MemorySaver ラッパー (InMemory実装)
│   └── retrieval/
│       ├── reranker.py                      #   Cohere Reranker / Passthrough (Feature Flag)
│       ├── vector_store.py                  #   pgvector 接続・シード・非同期対応
│       ├── keyword_search.py                #   PostgreSQL FTS (tsvector / ts_rank)
│       ├── embedding.py                     #   OpenAI Embeddings (text-embedding-3-small)
│       └── chunking.py                      #   SemanticChunker (コサイン類似度パーセンタイル)
│
├── prompts/                                 # --- Prompt Ops ---
│   ├── router/v1.yaml                       #   Router用プロンプト
│   ├── decompose/v1.yaml                    #   Decomposer用プロンプト
│   ├── rewrite/v1.yaml                      #   Rewrite用プロンプト
│   ├── retrieval_critic/v1.yaml             #   Retrieval Critic用プロンプト
│   ├── answer_critic/v1.yaml                #   Answer Critic用プロンプト
│   ├── generate/v1.yaml                     #   Generate用プロンプト
│   └── compare_generate.yaml                #   Compare 専用 Generate プロンプト
│
├── tools/                                   # --- 運用ツール ---
│   └── sync_prompts_from_hub.py             #   LangSmith Hub → local 同期
│
├── scripts/                                 # --- ベンチマーク ---
│   ├── benchmark_router.py                  #   Router ベンチマーク (Before/After)
│   └── benchmark_compare.py                 #   Compare Pipeline ベンチマーク
│
├── tests/                                   # --- テスト ---
│   ├── test_compare_intent.py               #   Compare 意図抽出テスト (15件)
│   ├── test_compare_pipeline.py             #   Compare パイプライン E2E テスト
│   ├── test_heuristic_router.py             #   Heuristic Router テスト
│   ├── test_router_service.py               #   Router Facade テスト
│   ├── test_critic_fallbacks.py             #   Critic フォールバックテスト
│   ├── test_prompt_loader.py                #   Prompt Loader テスト
│   ├── test_prompt_sync.py                  #   Prompt Sync テスト
│   ├── test_retrieval_complex_budget.py     #   Budget / Fallback テスト
│   ├── test_retrieval_complex_fallback.py   #   retrieval_complex 縮退経路テスト
│   ├── test_evaluation.py                   #   集計・統計計算ロジックのテスト
│   ├── test_reporter.py                     #   比較・レポート生成ロジックのテスト
│   ├── test_structured_query_validator.py   #   SQ バリデータ・安全制御テスト
│   ├── test_structured_query_sql_builder.py #   SQ SQL 生成ロジックテスト
│   ├── test_structured_query_sqlite_execution.py # SQ SQLite 実行テスト
│   ├── test_structured_query_tool.py        #   SQ ツール結合テスト
│   └── test_structured_query_router.py      #   SQ ルーティングテスト
│
├── evaluation/                              # --- 評価パイプライン ---
│   ├── evaluate.py                          #   評価実行・データ収集メイン
│   ├── aggregator.py                        #   統計（p50/p95）集計ロジック
│   ├── schema.py                            #   評価レコード・レポートの Pydantic モデル
│   ├── reporter.py                          #   比較・レポート生成 (CLI)
│   ├── dataset.json                         #   評価用データセット
│   ├── results/                             #   JSON 実行結果保存先
│   ├── reports/                             #   Markdown/HTML レポート出力先
│   └── templates/                           #   Jinja2 レポートテンプレート (MD/HTML)
│
└── docs/                                    # --- 設計・運用ドキュメント ---
    ├── EVALUATION.md                        #   評価フロー運用ガイド
    ├── phase2-production-rag.md              #   Phase 2 設計書
    ├── phase2-5-design.md                    #   Phase 2.5 設計書
    ├── phase3-agentic-retrieval-v2.md        #   Phase 3 設計書
    ├── sprint3_retrieval_complex_optimization_design.md
    ├── walkthrough2.5.md
    └── implementation_2.5plan.md
```
