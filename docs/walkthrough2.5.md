# Phase 2.5 実装 Walkthrough

## 実装概要
Phase 2.5 設計書に基づき、検索パイプラインを5ステージ構成に強化した。

## 変更ファイル一覧

### 新規ファイル（6個）
| ファイル | 内容 |
| :--- | :--- |
| [domain/models/retrieval_models.py](file:///Users/apple/develop/ai-agent-rag/domain/models/retrieval_models.py) | [RetrievedChunk](file:///Users/apple/develop/ai-agent-rag/domain/models/retrieval_models.py#5-18), [RewriteResult](file:///Users/apple/develop/ai-agent-rag/domain/models/retrieval_models.py#20-35), [CompressionResult](file:///Users/apple/develop/ai-agent-rag/domain/models/retrieval_models.py#45-53), [SourceSpan](file:///Users/apple/develop/ai-agent-rag/domain/models/retrieval_models.py#37-43) |
| [domain/services/query_rewriter.py](file:///Users/apple/develop/ai-agent-rag/domain/services/query_rewriter.py) | LLMベースのクエリ書き換え（遅延初期化） |
| [domain/services/hybrid_search.py](file:///Users/apple/develop/ai-agent-rag/domain/services/hybrid_search.py) | Vector + Keyword 並列実行・スコア統合 |
| [domain/services/confidence.py](file:///Users/apple/develop/ai-agent-rag/domain/services/confidence.py) | Confidence算出 + Dynamic TopK |
| [domain/services/compressor.py](file:///Users/apple/develop/ai-agent-rag/domain/services/compressor.py) | 文レベルの抽出圧縮（遅延初期化） |
| [infrastructure/retrieval/keyword_search.py](file:///Users/apple/develop/ai-agent-rag/infrastructure/retrieval/keyword_search.py) | PostgreSQL FTS (tsvector/ts_rank) |

### 変更ファイル（5個）
| ファイル | 変更内容 |
| :--- | :--- |
| [domain/services/retrieval_service.py](file:///Users/apple/develop/ai-agent-rag/domain/services/retrieval_service.py) | 全面書き換え → 5ステージパイプライン |
| [application/dto/chat_models.py](file:///Users/apple/develop/ai-agent-rag/application/dto/chat_models.py) | [Source](file:///Users/apple/develop/ai-agent-rag/application/dto/chat_models.py#4-15) に `hybrid_score`, `vector_score`, `bm25_score` 追加 |
| [application/services/chat_service.py](file:///Users/apple/develop/ai-agent-rag/application/services/chat_service.py) | RetrievalService の `confidence` をそのまま採用 |
| [main.py](file:///Users/apple/develop/ai-agent-rag/main.py) | CLI表示を PH2.5 対応（hybrid/vec/bm25 スコア表示） |
| [.env.example](file:///Users/apple/develop/ai-agent-rag/.env.example) | Hybrid Search / TopK / タイムアウト設定追加 |

### ドキュメント更新
| ファイル | 変更内容 |
| :--- | :--- |
| [README.md](file:///Users/apple/develop/ai-agent-rag/README.md) | Phase 2.5 全面反映（パイプライン図・スコア統合表・設定項目一覧等） |

## 検証結果

### インポートチェーン検証
```
✅ Data models import OK
✅ QueryRewriter import OK
✅ KeywordSearch import OK
✅ HybridSearch import OK
✅ ConfidenceEstimator import OK
✅ ExtractiveCompressor import OK
✅ RetrievalService (pipeline) import OK
✅ Source model OK: hybrid=0.9, vec=0.6, bm25=0.3
✅✅✅ All Phase 2.5 import chains verified!
```

### ロジックテスト
- `ConfidenceEstimator.estimate()`: top1=0.9, margin=0.2 → confidence=0.78, top_k=3 ✅
- `RewriteResult.combined_queries`: original + rewrite の2クエリリスト ✅

## 動作確認手順
```bash
docker compose up -d
uv run python main.py
# 質問を入力すると hybrid/vec/bm25 スコア付きでレスポンスが返る
```
