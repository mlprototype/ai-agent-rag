# Sprint 3: retrieval_complex 最適化 実装設計メモ（レビュー反映版）

## 目的

retrieval_complex 経路において、Timeout / Budget / Fallback を段階的に制御し、レイテンシ悪化と品質低下を最小化する。
特に、全失敗と部分失敗を区別し、必要な処理のみを残しながら generate を確実に実行できる制御面を導入する。

## スコープ

*   retrieval_complex の budget 制御見直し
*   fallback level の導入
*   optional stage の skip 条件明文化
*   partial retrieval の状態管理
*   state / structured log の拡張
*   delay / timeout / partial failure テスト拡充

## スコープ外

*   compare fast-path の再設計
*   heuristic router の再設計
*   definition / direct / calc の route 変更
*   LLM モデル変更

## 現状課題

*   初期 budget が静的で、実行中の再配分が弱い
*   partial failure と total failure の区別が粗い
*   retrieval_degraded=True だけでは低品質要因を追えない
*   rewrite / critic / rerank の skip 条件が曖昧
*   user-facing warning が内部状態と対応付いていない

## 設計方針

### 1. fallback level を導入する

retrieval_complex に対して fallback level を導入する。

*   **Level 0: full_path**
    *   decompose -> parallel_retrieve -> merge -> rerank -> retrieval_critic -> optional rewrite -> generate
*   **Level 1: optimization_skip**
    *   rerank をスキップ
    *   retrieval_critic は実行可
*   **Level 2: critic_skip**
    *   rerank / retrieval_critic / rewrite をスキップ
    *   現在の context で generate
*   **Level 3: single_retrieval_fallback**
    *   decomposition / parallel retrieval を諦め、元クエリ1本で retrieval
    *   generate に進む
*   **Level 4: minimal_answer**
    *   十分な retrieval context が得られない場合の最終防衛線
    *   controlled warning 付きで最小回答、または controlled failure

### 2. stage ごとの予算予約を導入する

全体 budget から、少なくとも generate 用の最低予算を先に予約する。

*   **最低限定義する設定:**
    *   `BUDGET_TOTAL_RETRIEVAL_COMPLEX_MS`
    *   `BUDGET_RESERVED_GENERATE_MS`
    *   `BUDGET_RESERVED_COMMIT_MS`
    *   `BUDGET_MIN_FOR_RERANK_MS`
    *   `BUDGET_MIN_FOR_CRITIC_MS`
    *   `BUDGET_MIN_FOR_REWRITE_MS`
*   **制御ルール:**
    *   generate の予約を侵食しない
    *   残予算が閾値を下回ったら fallback level を引き上げる
    *   must_generate=True の場合は optional stage を打ち切る

### 3. budget checkpoint を各主要 node に追加する

最低限、以下の時点で budget 判定を行う:

*   after decompose
*   after parallel_retrieve
*   after merge
*   before rerank
*   before retrieval_critic
*   before rewrite
*   before generate

各 checkpoint で判断すること:

*   remaining budget
*   fallback level の引き上げ要否
*   optional stage の skip 要否
*   must_generate の設定

### 4. partial retrieval を正式に state 化する

一部 subquery が timeout しても、取得済み結果で継続可能にする。

state に追加する項目:

*   `fallback_level`
*   `partial_retrieval_used`
*   `retrieval_timeout_count`
*   `retrieval_success_count`
*   `skipped_stages`
*   `budget_pressure_reasons`
*   `must_generate`
*   `remaining_budget_ms_at_generate`
*   `warning_codes`
*   `retrieval_quality_level`

### 5. warning を内部 code と表示文言に分離する

*   **内部 code:**
    *   slow_retrieval
    *   decompose_timeout
    *   critic_skipped_budget
    *   rewrite_skipped_budget
    *   single_retrieval_fallback
    *   partial_retrieval_used
*   **表示文言例:**
    *   一部の検索工程を省略して回答を生成しました。
    *   検索結果の一部のみを用いて回答しています。
    *   十分な検索予算を確保できず、簡略化した経路で回答しています。

### 6. rewrite loop は budget 制約下で厳格化する

*   fallback_level >= 2 の場合は rewrite 禁止
*   rewrite 回数は最大1回
*   remaining_budget_ms < BUDGET_MIN_FOR_REWRITE_MS の場合はスキップ

## 実装対象ファイル

*   `application/agents/state.py`
    *   budget / fallback / warning 系 state 追加
*   `application/agents/graph.py`
    *   fallback level 制御
    *   budget checkpoint ロジック
    *   optional stage skip 条件追加
    *   structured log 追加
*   必要に応じて budget ヘルパーモジュール新設
    *   例: `domain/services/retrieval_budget.py`

## テスト計画

### 単体テスト

*   decompose timeout -> fallback level 上昇
*   subquery 1件 timeout -> partial retrieval 継続
*   subquery 多数 timeout -> single retrieval fallback
*   rerank skip 条件
*   critic skip 条件
*   rewrite skip 条件
*   must_generate=True のとき optional stage を実行しない
*   warning code と user-facing warning の整合

### E2E / ベンチ

*   retrieval_complex ケースを 10 件程度用意
*   before / after 比較:
    *   timeout rate
    *   fallback level 分布
    *   warning rate
    *   p50 / p95 latency
    *   degraded rate
    *   answer quality の簡易確認

## 成功条件

*   retrieval_complex の timeout 起因 fallback が減少する
*   partial retrieval を total failure 扱いしなくなる
*   generate 到達率が安定する
*   warning と内部状態の対応が追える
*   p95 latency の悪化を抑えつつ degraded rate を改善する
