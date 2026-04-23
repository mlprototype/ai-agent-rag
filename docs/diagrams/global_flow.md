# 全体俯瞰図

リクエストがシステムに入ってから、どのような大きな経路を経て回答に至るかを示す図です。軽量経路と検索系経路を大きく分けて表現しています。

```mermaid
flowchart TD
    Start([API / CLI リクエスト受付]) --> Init[初期状態と予算設定]
    Init --> Route[クエリ意図分析とルート決定]
    
    Route -->|ルート判定| Branch{経路の大きな振り分け}
    
    subgraph 軽量経路
        Branch -->|計算が必要| Calc[計算処理パス]
        Branch -->|単純な質問| Direct[直答パス]
    end

    subgraph 検索系経路
        Branch -->|比較意図明確| Compare[比較専用パス]
        Branch -->|ルーター失敗/予算少| Fallback[1回のみ簡易検索パス]
        Branch -->|複雑な質問| Agentic[複雑な検索・評価ループ]
    end

    Calc --> Gen[回答の生成]
    Direct --> Gen
    Fallback --> Gen
    Agentic --> Gen
    
    Compare -.->|専用フォーマットで生成| Commit
    Gen --> Commit[最終回答の確定と履歴保存]
    Commit --> End([応答返却])
```

#### 補足
- **この図の主経路:** 初期設定 → ルーティング → 複雑な検索・評価ループ (Agentic) → 回答生成 → 確定
- **この図の fallback / 縮退経路:** ルーター失敗時などに Agentic ではなく「1回のみ簡易検索パス」へ落ちる経路
- **この図で重要な state 更新:** `initial_budget_ms` (予算設定), `route` (経路決定), `messages` (最終履歴)
- **省略したもの:** 各種ノード内部の詳細なループや、Compare Fast-Path 内部の抽出・マージロジック
- **対応する主要実装ファイル:** `application/agents/graph.py` (全体の Edge 定義)
