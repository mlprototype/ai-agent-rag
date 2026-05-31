# 評価フロー運用ガイド

このドキュメントでは、開発フローにおける評価システムの運用手順を説明します。

## 1. 評価の目的
- 変更による品質の劣化（Regression）を早期に検知する。
- `structured_query_tool` などの複雑なコンポーネントの安全動作（Fail-safe）を監視する。
- 定量的なメトリクス（合格率、レイテンシ、信頼度）に基づいた意思決定を行う。

## 2. 開発フロー
開発者は以下の手順で評価を実施することを推奨します。

### Step 1: 評価の実行
コードの変更後、以下のコマンドで評価を実行します。
```bash
python3 evaluation/evaluate.py
```
実行完了後、`evaluation/eval_results_YYYYMMDD_HHMMSS.json` が生成されます。

### Step 2: レポート生成と比較
最新の結果を、基準となる結果（Baseline）と比較します。
```bash
# Baseline がある場合 (推奨)
uv run python -m evaluation.reporter --baseline evaluation/eval_results_baseline.json --current evaluation/eval_results_latest.json --format both

# Baseline がない場合 (初回など)
uv run python -m evaluation.reporter --current evaluation/eval_results_latest.json --format both
```

### Step 3: 結果の確認と対処
生成されたレポート（既定では `evaluation/reports/` 配下）を確認します。
- **エグゼクティブ・サマリー**: 指標の大きな悪化がないか確認。
- **要対応ケース (Action Required)**: 類似度が低い、または回答不合格となったクエリを個別に見直し、プロンプトや検索ロジックを修正。
- **Structured Query 分析**: 意図しない Fail-safe（ブロック）が多発していないか確認。

## 3. レポートの読み方

各指標やコードの意味は以下の通りです。

| 指標 / 項目 | 意味 |
| :--- | :--- |
| **response_generated_rate** | システムが最終的に何らかの回答テキストを生成できた割合。 |
| **answer_ok_rate** | 回答が品質基準（Critic等）をパスし、品質が担保されている割合。 |
| **fallback_level** | 縮退動作の度合い。`LEVEL_0`（正常）から、予算不足等によるスキップが増えるほど数値が上がります。 |
| **warning_codes** | `TIMEOUT_ROUTER`（ルーター遅延）や `LOW_CONFIDENCE`（低信頼度）など、内部的な注意状態を示す詳細コード。 |
| **reason_code** | 実行結果のステータス。`SUCCESS` 以外に `unsupported_query`（未対応）や `validation_failed`（検証失敗）などがあります。 |
| **source_name** | `structured_query` においては、参照した**データソース識別子**（例: SQLite）を指します。検索における引用元（Citation source）とは別物です。 |

## 4. 保存先と管理ルール

### レポートの保存先
生成されたレポートは、以下のディレクトリに整理して保存することを推奨します。
- JSON データ: `evaluation/results/eval_results_YYYYMMDD_HHMMSS.json`
- レポートファイル: `evaluation/reports/report.md` (または `.html`)

### Baseline の更新・管理ルール
品質の基準点となる Baseline は、以下のルールで運用します。

1.  **更新タイミング**:
    *   `main` ブランチへ大規模なマージが行われ、品質が安定したことを確認できた直後に更新します。
    *   意図的な品質向上（プロンプト改善など）が行われた場合に、新しい基準として固定します。
2.  **固定する状態**:
    *   すべての既存テストケースが期待通りに動作し、`answer_ok_rate` が許容範囲内である状態を Baseline とします。
3.  **禁止事項**:
    *   **評価が失敗（Regressionが発生）している状態では、絶対に Baseline を上書きしてはいけません。**
    *   悪化した状態を Baseline にすると、それ以降の劣化を検知できなくなります。

---
*Last Updated: 2026-04-23*
