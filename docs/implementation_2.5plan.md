# Phase 2.5 実装計画: Production RAG 強化

## 概要
Phase 2 の Production RAG 基盤に、検索精度と安定性を向上させる5つのコンポーネントを追加する。
Rerank は導入せず、パイプライン全体を `Query Rewrite → Hybrid Search → Confidence/Dynamic TopK → Extractive Compression → LLM` として構成する。

---

## Proposed Changes

### データモデル (domain/models)

#### [NEW] [retrieval_models.py](file:///Users/apple/develop/ai-agent-rag/domain/models/retrieval_models.py)
パイプライン内で受け渡しするデータクラスを定義:
- `RetrievedChunk`: `doc_id`, `chunk_id`, `content`, `metadata`, `vector_score`, `bm25_score`, `hybrid_score`
- `RewriteResult`: `original_query`, `rewrite_query`, `combined_queries`
- `CompressionResult`: `compressed_text`, `source_spans`

---

### Query Rewriter (domain/services)

#### [NEW] [query_rewriter.py](file:///Users/apple/develop/ai-agent-rag/domain/services/query_rewriter.py)
- LLM（GPT-4o-mini）を使い、ユーザークエリを検索向けに短く書き換える
- `original_query` と `rewrite_query` を併用して検索に渡す
- タイムアウト（env: `STAGE_TIMEOUT_MS_REWRITE`, デフォルト 3000ms）付き
- 失敗時は `original_query` のみで続行（フォールバック）

---

### Hybrid Search (infrastructure + domain)

#### [NEW] [keyword_search.py](file:///Users/apple/develop/ai-agent-rag/infrastructure/retrieval/keyword_search.py)
- PostgreSQL FTS を SQLAlchemy 経由で直接実行
- `to_tsvector('simple', content) @@ plainto_tsquery('simple', query)` + `ts_rank` でスコア取得
- `langchain_postgres` の `PGVector` が管理するテーブル `langchain_pg_embedding` を直接参照

#### [NEW] [hybrid_search.py](file:///Users/apple/develop/ai-agent-rag/domain/services/hybrid_search.py)
- Vector Search と Keyword Search を並列実行（`asyncio.gather`）
- スコア正規化（min-max per result set）後に加重平均
  - `hybrid_score = alpha * norm_vector + (1 - alpha) * norm_bm25`
- env: `HYBRID_ALPHA=0.6`, `RETRIEVE_K_VECTOR=30`, `RETRIEVE_K_KEYWORD=30`
- Keyword Search 失敗時は Vector のみで続行（フォールバック）

---

### Confidence + Dynamic TopK (domain/services)

#### [NEW] [confidence.py](file:///Users/apple/develop/ai-agent-rag/domain/services/confidence.py)
- Hybrid Search 結果のスコア分布から Confidence を算出
  - `confidence = clamp(0.2 + 0.6 * top1 + 0.2 * margin, 0, 1)`
- Confidence に基づく Dynamic TopK:
  - `top1 >= 0.85 and margin >= 0.05` → `top_k = 3`
  - `top1 >= 0.70` → `top_k = 5`
  - else → `top_k = 8`
- env: `TOP_K_MIN=3`, `TOP_K_MAX=8`

---

### Extractive Compression (domain/services)

#### [NEW] [compressor.py](file:///Users/apple/develop/ai-agent-rag/domain/services/compressor.py)
- 各チャンクを文レベルに分割し、LLM（GPT-4o-mini）でクエリとの関連度を判定
- 関連度の高い文のみを抽出して `compressed_text` を構成
- `source_spans`（`doc_id`, `chunk_id`, `sentence_idx`）で Citation 追跡を保持
- タイムアウト（env: `STAGE_TIMEOUT_MS_COMPRESS`, デフォルト 5000ms）付き
- 失敗時は圧縮スキップ（元テキストをそのまま使用）

---

### パイプライン統合 (既存ファイル変更)

#### [MODIFY] [retrieval_service.py](file:///Users/apple/develop/ai-agent-rag/domain/services/retrieval_service.py)
- 既存の [search_knowledge_base()](file:///Users/apple/develop/ai-agent-rag/domain/services/retrieval_service.py#17-66) をパイプライン化:
  1. Query Rewrite → 2. Hybrid Search → 3. Confidence + Dynamic TopK → 4. Extractive Compression
- フォールバック順序: Compression skip → Rewrite skip → Keyword skip (vector only)
- 全体タイムアウトは既存の 5.0秒を維持

#### [MODIFY] [chat_models.py](file:///Users/apple/develop/ai-agent-rag/application/dto/chat_models.py)
- [Source](file:///Users/apple/develop/ai-agent-rag/application/dto/chat_models.py#4-12) に `hybrid_score: float` フィールドを追加

#### [MODIFY] [chat_service.py](file:///Users/apple/develop/ai-agent-rag/application/services/chat_service.py)
- [_extract_citations()](file:///Users/apple/develop/ai-agent-rag/application/services/chat_service.py#14-60) で `hybrid_score` を [Source](file:///Users/apple/develop/ai-agent-rag/application/dto/chat_models.py#4-12) に反映
- JSON出力に含まれる `confidence` をそのまま採用（RetrievalService 側で算出済み）

#### [MODIFY] [.env.example](file:///Users/apple/develop/ai-agent-rag/.env.example)
- Phase 2.5 用の設定項目を追加

---

### README 更新

#### [MODIFY] [README.md](file:///Users/apple/develop/ai-agent-rag/README.md)
- Phase 2.5 の機能テーブル追加
- パイプラインフロー図の更新
- 各コンポーネントの詳細セクション追加
- ディレクトリ構成・技術スタック・設定項目の更新

---

## Verification Plan

### 構文検証
- `python -c "from domain.services.retrieval_service import RetrievalService"` でインポートチェーン全体が解決されることを確認

### 手動検証
以下の手順でユーザーに動作確認を依頼:
1. `docker compose up -d` で pgvector 起動
2. `uv run python main.py` で CLI 起動
3. 質問を入力し、`answer + sources（hybrid_score付き） + confidence` が返却されることを確認
4. レイテンシがP95 < 3s の範囲内であることを確認
