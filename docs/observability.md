<!--
ファイルの責務:
- 本システム（Agentic RAG with Control Plane）における構造化ログおよび可観測性（Observability）の仕様をドキュメント化する。
- 開発時や本番運用時に出力される構造化ログの各フィールドの意味を分類（Routing, Compare, Prompt, Chat Summary, Budget）ごとに説明する。
- 注意点: 障害追跡や性能分析（ボトルネック特定）を行う際、これらのログフィールドをダッシュボード（Grafana, Kibanaなど）で可視化して監視する想定である。
-->

# 可観測性（Observability）仕様

各ノードは構造化ログ（`logger.info({"event": ...})`）を出力します。以下は記録されている主要な項目です。

## ログフィールド一覧

### Routing

| フィールド | 説明 |
| :--- | :--- |
| `routing_layer` | `heuristic` / `llm` / `fallback` |
| `route_decision_source` | `heuristic_match` / `llm_success` / `llm_timeout_fallback` / `llm_error_fallback` |
| `heuristic_matched` | Heuristic ルールにマッチしたか |
| `heuristic_rule` | マッチしたルール名（`direct_greeting`, `calc_expression`, `compare_keywords`, `definition_keywords`） |
| `route_decision_latency_ms` | ルーティング解決にかかった時間 |
| `route_decision_confidence` | 判定の確信度 |
| `llm_router_invoked` | LLM Router が呼び出されたか |

### Compare Fast-Path

| フィールド | 説明 |
| :--- | :--- |
| `compare_extract_success` | A/B 対象の抽出成功フラグ |
| `compare_targets` | `{target_a, target_b}` |
| `compare_doc_count_a` / `_b` | A/B それぞれの検索ヒット件数 |
| `compare_context_coverage_ok` | merge 後の coverage 判定結果 |
| `compare_route_fallback_used` | agentic_retrieval へのフォールバック有無 |
| `quality_gate_status` | compare fast-path 成功時に現在は `pass` を記録 |

### Prompt / Runtime

| フィールド | 説明 |
| :--- | :--- |
| `resolution_source` | prompt 解決元。現行 runtime は `local` が正本 |
| `prompt_version` | YAML 内の version。Hub 同期後も維持 |
| `used_fallback` | 埋め込み fallback prompt を使ったか |
| `prewarm_phase` | `startup` / `runtime` |
| `critical` | critical prompt かどうか |

### Chat Summary

| フィールド | 説明 |
| :--- | :--- |
| `total_latency_ms` | 1 リクエスト全体の所要時間 |
| `critic_degraded` | critic skip / critic fallback / retrieval_degraded をまとめた要約フラグ |
| `final_confidence` | 最終的に返却した confidence |
| `citation_filtered_count` | 回答文中で参照されず除外された source 件数 |
| `remaining_budget_ms_at_generate` | generate 開始時点の残予算 |

### Budget / Fallback

| フィールド | 説明 |
| :--- | :--- |
| `initial_budget_ms` | クエリに割り当てられた初期予算 |
| `remaining_budget_ms` | 各ノード通過時点の残予算 |
| `fallback_level` | 現在の縮退レベル |
| `skipped_stages` | 予算不足でスキップされたステージ一覧 |
| `budget_pressure_reasons` | 予算圧迫の要因 |
| `must_generate` | Generate 強制遷移フラグ |
| `retrieval_degraded` | 検索品質縮退フラグ |
| `warning_codes` | 内部 warning コード一覧 |
| `retrieval_quality_level` | 検索品質の段階（内部判定） |
| `timeout_stages` / `fallback_stages` | タイムアウト/フォールバックが発生したステージ |
