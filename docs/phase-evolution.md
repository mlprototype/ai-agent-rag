<!--
ファイルの責務:
- 本システム（Agentic RAG with Control Plane）の開発フェーズと実装進捗（Phase 1 〜 Phase 3）をドキュメント化する。
- 各フェーズ（Agentic RAG Core, Production RAG, Production RAG 強化, Control Plane Enhancements）の主要機能と詳細、Sprintの構成を詳細に記述する。
- 注意点: 本ファイルはプロジェクトの進化プロセスを記録するロードマップであり、各機能の設計思想を理解する手がかりとなる。
-->

# 実装フェーズ詳細

自律的なルーティングと段階的な検索処理を実装しており、各Phaseを経て機能が強化されています。

| Phase | 主要機能 | キーワード |
| :--- | :--- | :--- |
| Phase 1 | Agentic RAG Core | LangGraph / 状態遷移 |
| Phase 2 | Production RAG | Hybrid Search / Confidence |
| Phase 2.5 | Production RAG 強化 | 5ステージ検索パイプライン |
| Phase 3 | Control Plane | Heuristic Routing / Compare Fast-Path / Budget |

## Phase 1: Agentic RAG Core（✅ 実装完了）

| 機能 | 概要 |
| :--- | :--- |
| **動的ルーティング（Agent Routing）** | Heuristic + LLM Router が `direct_answer` / `calculator` / `structured_query_tool` / `agentic_retrieval` / `fallback_retrieval` を判定 |
| **Clean Architecture** | 業務ロジック、アダプター、インターフェースを疎結合にレイヤー分離 |
| **リアルタイム生成（StreamingResponse）** | FastAPI `StreamingResponse` による `text/event-stream` 配信。通常は `generate` ノードの出力を逐次返却 |
| **耐障害性（Stage Timeout & Retry）** | 外部API遅延に対する `asyncio.wait_for` タイムアウト（5秒）とフォールバックエラー処理 |
| **Prompt Ops（Prompt Versioning）** | `prompts/` 配下のローカル snapshot を runtime 正本として利用し、FastAPI 起動時に prewarm。LangSmith Hub は同期元として扱う |
| **自動評価パイプライン（Evaluation）** | Recall@3 と Answer Similarity（LLM-as-a-Judge）による検索精度の自動計測 |

## Phase 2: Production RAG（✅ 実装完了）

| 機能 | 概要 |
| :--- | :--- |
| **Conversation Memory** | `MemorySaver` によるセッション単位の会話履歴保持。`session_id` をキーとしたマルチターン対話を実現 |
| **Citation 付き回答** | `answer + query_type + route + sources[] + confidence + warning` の構造化レスポンス。引用元ドキュメントの `doc_id / chunk_id / score / snippet / rerank_score` を追跡 |
| **Document Ingestion Pipeline** | `unstructured` ベースのマルチフォーマット（MD/HTML/TXT）取り込みパイプライン。API経由でのファイル/ディレクトリ一括取り込みに対応 |
| **Semantic Chunking** | `SemanticChunker`（コサイン類似度パーセンタイル方式）による意味的文脈を保持したチャンク分割 |

## Phase 2.5: Production RAG 強化（✅ 実装完了）

検索からコンテキスト生成までの処理を、高精度化・コスト削減のための**「5ステージ検索パイプライン」**として再構成しました。

| 実行順 (Stage) | 主要機能 | 概要 |
| :--- | :--- | :--- |
| **Stage 1** | **Query Rewrite** | LLM（GPT-4o-mini）による検索向けクエリ書き換え。元のクエリと併用してヒット率を向上 |
| **Stage 2** | **Hybrid Search** | Vector（pgvector） + Keyword（PostgreSQL FTS）の並列検索とスコアの加重平均統合 |
| *(オプション)* | **Optional Reranker** | `ENABLE_RERANK=true` 時のみ Cohere Reranker による再ランクを実行（既定は Passthrough） |
| **Stage 3** | **Confidence & Dynamic TopK** | スコア分布から確信度を算出し、それに基づき取得ドキュメント数を動的制御（3〜8件） |
| **Stage 4** | **Extractive Compression** | 文レベルの関連性判定によるテキスト抽出圧縮。不要トークンを削減しハルシネーションを防止 |
| **Stage 5** | **構造化 JSON 出力** | 最終回答生成に必要な「コンテキスト・引用元・確信度」を構造化して後続へ引き渡し |

## Phase 3: Control Plane Enhancements（✅ 実装完了）

Phase 3 では、Agentic RAG の制御面を強化し、ルーティング、比較処理、複雑検索に対して、レイテンシ・品質・縮退制御を改善しました。

### Sprint 1: Heuristic Routing

| 機能 | 概要 |
| :--- | :--- |
| **Heuristic Router** | ルールベースの事前分類で `direct` / `calc` / `definition` / `compare` を高確信で即座に判定 |
| **LLM Router Skip** | Heuristic hit 時は LLM Router をスキップし、router timeout と不要な fallback を削減 |
| **Observability** | routing decision の structured observability を追加 |

### Sprint 2: Compare Fast-Path

| 機能 | 概要 |
| :--- | :--- |
| **Compare 分離** | `query_type=compare` を専用パイプラインに分離し、比較対象ごとの独立並列検索を実現 |
| **Compare パイプライン** | intent 抽出 → target 別 retrieval → merge → 専用 generate の4段構成。coverage 不足時は通常 retrieval にフォールバック |
| **Compare Metadata** | compare fast-path 成功時は `quality_gate_status` / `quality_gate_confidence` を state に記録（現状はプレースホルダ値） |
| **フォールバック** | 抽出失敗・coverage 不足時は `agentic_retrieval` へ自動フォールバック |

### Sprint 3: Retrieval Complex Budget / Fallback Control

| 機能 | 概要 |
| :--- | :--- |
| **Budget-aware 制御** | `retrieval_complex` に `generate` / `commit` 予約付き budget 管理を導入 |
| **段階的縮退** | `fallback_level` による5段階縮退（full_path → optimization_skip → critic_skip → single_retrieval_fallback → minimal_answer） |
| **Partial Retrieval** | 並列検索の部分成功を許容し、全失敗と部分失敗を区別して品質判定 |
| **Warning / Observability** | `warning_codes`（内部）と user-facing `warning`（外部）を分離。`retrieval_quality_level` で品質段階を可視化 |
| **Dynamic Skip** | rerank / critic / rewrite を残予算に応じて段階的にスキップ |

### Sprint 4: Structured Query Evolution

| 機能 | 概要 |
| :--- | :--- |
| **SQLite Backend** | Python 辞書ベースから SQLite 実データベース実行へ移行。`data/structured_query.db` による実運用に近い集計を実現 |
| **責務分離 (Pipeline)** | Parse → Validate → SQL Builder → Execute (SQLite) → Format の 5 段階パイプラインに整理 |
| **Read-only 安全制御** | SQL インジェクション対策（プレースホルダ）に加え、実行レベルでの破壊的キーワード（INSERT/UPDATE/DELETE等）の厳格なブロック |
| **Standardized Response** | 構造化クエリ専用の `source_name`（例: `SQLite (sales)`）をレスポンスに付与。RAG との識別性を向上 |

これにより、不要なAPIコストとレイテンシを削減し、**企業利用に耐える高精度かつ自律的な Production RAG** を実現しています。
