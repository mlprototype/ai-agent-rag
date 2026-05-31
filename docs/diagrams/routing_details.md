# ルーティング詳細図

なぜそのルートが選ばれるのか、LLMを使わずに済む条件やフェイルセーフの仕組みを示す図です。

```mermaid
flowchart TD
    Start([ルーティング開始]) --> Preprocess[入力クエリの前処理<br/>複雑度・種別の事前推定]
    Preprocess --> Heuristic[ヒューリスティック判定<br/>ルールベースでの意図確認]
    
    Heuristic --> H_Check{明らかな意図か？<br/>計算式や明確な比較等}
    H_Check -->|Yes: LLM不要| RouteConfirm[ルート確定]
    
    H_Check -->|No: 要詳細分析| LLMRouter[必要時のみLLMルーティング<br/>GPTによる意図判定]
    LLMRouter --> L_Check{判定成功か？}
    
    L_Check -->|Yes| RouteConfirm
    L_Check -->|No: タイムアウト/例外| FallbackRoute[フェイルセーフルート確定<br/>安全のため検索へ倒す]
    
    RouteConfirm --> Dispatch
    FallbackRoute --> Dispatch
    
    Dispatch{各ルートへ分岐<br/>direct / calc / compare / agentic / fallback} --> End([次ノードへ])
```

#### 補足
- **この図の主経路:** 入力前処理 → ヒューリスティック判定 (No) → LLMルーティング → ルート確定
- **この図の fallback / 縮退経路:** LLMルーティングがタイムアウト・エラーした際に、安全に事実に基づかせるための `fallback_retrieval` へのフォールバック
- **この図で重要な state 更新:** `query_type` (推論された意図), `router_reason` (ルーティング理由), `router_uncertain` (ルーターの自信有無)
- **省略したもの:** 複雑度（low/medium/high）のスコアリングの具体的な条件式、予算の初期化詳細
- **対応する主要実装ファイル:** `application/agents/graph.py` (`router_node`), `domain/services/router.py`
