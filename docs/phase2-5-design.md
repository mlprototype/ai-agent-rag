# Phase2.5 詳細設計: Production RAG 強化 (Rerank なし)

作成日: 2026-03-16

## 0. 目的
Phase2 で整備した Production RAG の基盤に、検索精度と安定性を上げるための改善を追加する。Rerank は導入しない前提で設計する。

確定パイプライン:
```
User Query
↓
Query Rewrite
↓
Hybrid Search
↓
Confidence Estimation
↓
Dynamic TopK
↓
Extractive Compression
↓
LLM
```

---

## 1. 追加コンポーネント
- Query Rewriter
- Hybrid Search (Vector + Keyword)
- Confidence Estimator
- Dynamic TopK
- Extractive Compressor

---

## 2. パイプライン詳細

### 2.1 Query Rewrite
目的: 検索向けの短いクエリを生成し、検索精度を向上させる。

仕様:
- 1 回の rewrite を生成
- `original_query` と `rewrite_query` を併用して検索
- rewrite 失敗時は `original_query` のみ

出力例:
```json
{
  "original_query": "LangGraphとは？",
  "rewrite_query": "LangGraph ライブラリ 概要"
}
```

---

### 2.2 Hybrid Search
目的: vector search だけでなく、keyword 検索の強みを取り込む。

仕様:
- Vector: pgvector の similarity search
- Keyword: Postgres FTS (tsvector / ts_rank) を想定
- スコア正規化後に加重平均

スコア:
```
hybrid_score = alpha * norm_vector + (1 - alpha) * norm_bm25
```

初期値:
- `alpha = 0.6`
- `retrieve_k_vector = 30`
- `retrieve_k_keyword = 30`

---

### 2.3 Confidence Estimation
目的: 検索結果の確信度を定量化し、Dynamic TopK の基準にする。

初期ヒューリスティック:
- `top1 = hybrid_score[0]`
- `top2 = hybrid_score[1]`
- `margin = top1 - top2`

例:
```
confidence = clamp(0.2 + 0.6*top1 + 0.2*margin, 0, 1)
```

---

### 2.4 Dynamic TopK
目的: confidence に応じて文書数を可変化する。

初期ルール:
```
if top1>=0.85 and margin>=0.05: top_k=3
elif top1>=0.70: top_k=5
else: top_k=8
```

設定:
- `TOP_K_MIN=3`
- `TOP_K_MAX=8`

---

### 2.5 Extractive Compression
目的: LLM 入力を最小限にし、ノイズとコストを削減。

仕様:
- 文章単位で relevance を計算
- top_k 文書から関連文のみ抽出
- `source_spans` で citation 追跡可能にする

出力例:
```json
{
  "compressed_text": "…",
  "source_spans": [
    {"doc_id": "architecture.md", "chunk_id": "architecture.md#c12", "sentence_idx": 2}
  ]
}
```

---

## 3. データ構造 (新規)

`RetrievedChunk`
- `doc_id`
- `chunk_id`
- `content`
- `metadata`
- `vector_score`
- `bm25_score`
- `hybrid_score`

`RewriteResult`
- `original_query`
- `rewrite_query`
- `combined_queries`

`CompressionResult`
- `compressed_text`
- `source_spans`

---

## 4. レイテンシ予算 (P95 < 3s)
- Rewrite: 300ms
- Hybrid Search: 700ms
- Confidence + TopK: 100ms
- Compression: 800ms
- LLM/overhead: 1.1s

フォールバック順:
1. Compression skip
2. Rewrite skip
3. Keyword skip (vector only)

---

## 5. 設定項目 (env)
- `HYBRID_ALPHA=0.6`
- `RETRIEVE_K_VECTOR=30`
- `RETRIEVE_K_KEYWORD=30`
- `TOP_K_MIN=3`
- `TOP_K_MAX=8`
- `STAGE_TIMEOUT_MS_*` (rewrite / hybrid / compress)

---

## 6. 影響範囲 (想定変更箇所)
- `domain/services/retrieval_service.py`
  - パイプライン化
  - hybrid / confidence / top_k / compression を統合
- `infrastructure/retrieval/vector_store.py`
  - Keyword 検索用のメソッド追加
- `application/dto/chat_models.py`
  - `sources` に `hybrid_score` 等を追加
  - `confidence` を追加
- `application/services/chat_service.py`
  - `session_id` / query rewrite の経路整理

---

## 7. 成功基準
1. Hybrid Search が動作し、Keyword 単独より高い Recall を示す
2. Confidence に応じた Dynamic TopK が期待通り動く
3. Extractive Compression で入力量を削減できる
4. P95 < 3s を維持できる

---

## 8. 実装順
1. Query Rewrite
2. Hybrid Search
3. Confidence Estimation + Dynamic TopK
4. Extractive Compression

