# retrieval_complex 詳細図

検索・評価・再試行のループと、予算枯渇時の様々な縮退（Fallback）段階を示す図です。

```mermaid
flowchart TD
    Start([Agentic検索パス開始]) --> Retrieve[初回検索の実行]
    
    Retrieve --> BudgetCheck1{予算・品質チェック<br/>評価をすべきか？}
    
    BudgetCheck1 -->|予算枯渇| ForcedGen[最小回答への縮退<br/>検索を打ち切り強制生成へ]
    BudgetCheck1 -->|高確信度で十分| SkipCritic[一部ステージのスキップ<br/>評価を省いて生成へ]
    BudgetCheck1 -->|余裕あり| Critic[検索品質の評価<br/>Retrieval Critic]
    
    Critic --> CriticResult{評価結果は？}
    CriticResult -->|OK: 情報十分| Gen[回答の生成]
    CriticResult -->|NG: 情報不足| Decompose[不足観点を補うクエリ分解<br/>Decompose / Rewrite]
    
    Decompose --> DecomposeCheck{分解成功か？}
    DecomposeCheck -->|No: タイムアウト等| SingleFallback[単発検索への縮退<br/>現状クエリで簡易検索へ]
    DecomposeCheck -->|Yes| Parallel[分割クエリでの並列検索]
    
    Parallel --> Merge[検索結果の統合とリランク]
    Merge --> BudgetCheck1
    
    SingleFallback --> Merge
    SkipCritic --> Gen
    ForcedGen --> Gen
    
    Gen --> BudgetCheck2{回答生成後の予算}
    BudgetCheck2 -->|枯渇| Commit[最小回答として確定<br/>評価スキップ]
    BudgetCheck2 -->|余裕あり| AnsCritic[回答品質の評価<br/>Answer Critic]
    
    AnsCritic --> AnsResult{回答は十分か？}
    AnsResult -->|OK| Commit
    AnsResult -->|NG: 再試行可能| Decompose
    AnsResult -->|NG: 再試行上限| Commit
    
    Commit --> End([次ノードへ])
```

#### 補足
- **この図の主経路:** 検索 → 評価 (NG) → 分解 → 並列検索 → 統合 → 評価 (OK) → 生成 → 評価 (OK) → 確定
- **この図の fallback / 縮退経路:** 
  1. **一部ステージのスキップ**: 確信度が高い場合に Critic を飛ばす最適化
  2. **単発検索への縮退**: Decompose 等の処理がタイムアウトした場合、分割せずに単一クエリで検索を続行
  3. **最小回答への縮退**: 残り予算（`remaining_budget_ms`）が枯渇した場合、強制的にその時点のコンテキストで回答を生成・確定する
- **この図で重要な state 更新:** `remaining_budget_ms` (残り予算), `retrieval_ok` (検索完了フラグ), `sub_queries` (分割クエリ), `must_generate` (強制生成フラグ), `fallback_level` (現在の縮退レベル)
- **省略したもの:** Chunk や Document モデルの変換処理、タイムアウト秒数の微細な算出ロジック
- **対応する主要実装ファイル:** `application/agents/graph.py` (複雑検索系ノード全体), `domain/services/retrieval_budget.py`
