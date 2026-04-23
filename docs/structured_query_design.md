# Structured Query Tool 設計思想と将来拡張

## 1. 目的と位置づけ

Agentic RAG は、マニュアルや社内規定といった「非構造化ドキュメント」の検索・回答生成に特化した強力なアーキテクチャです。しかし、実際の業務システムにおいては「今月の売上は？」「在庫が少ない商品トップ3は？」といった「構造化データ（業務データ）」に対する集計や問い合わせのニーズが必ず発生します。

構造化データに対して従来のベクトル検索（RAG）を適用すると、正確な集計結果を得ることが困難です。そのため、当プロジェクトでは Control Plane（ルーター）によって質問の性質を判定し、構造化データへの問い合わせを専用の `structured_query_tool` ルートへ振り分けるアプローチを採用しています。

## 2. 現在の設計（SQLite-backed Structured Query Route）

現在は、セキュリティと確実性を最優先し、バリデーション済みの意図（Intent）から安全な SQL を動的に組み立てて SQLite で実行する構成となっています。

処理は以下の 5 ステージに明確に分離（責務分離）されています：

1. **Parse (意図抽出)**
   - ユーザーの自然文から、対象データセット（売上、在庫など）、操作（sum, avg, count, top_k, max, min）、対象指標、およびフィルタ条件を抽出し、`StructuredQueryIntent` モデルにマッピングします。
2. **Validate (安全制御)**
   - 抽出された Intent に対して、厳格な Allowlist ベースのバリデーションを行います。
   - 許可されていない操作やフィールド、データセット（テーブル）は、この段階で安全にブロックします。
3. **SQL Builder (クエリ構築)**
   - バリデーション済みの Intent をもとに、`?` プレースホルダを用いた安全な SQL テンプレートを構築します。
   - SQL インジェクションを構造的に防止します。
4. **Execute (実行)**
   - `SQLiteDataSource` を通じて、`data/structured_query.db` に対して **Read-only** でクエリを実行します。
   - 実行レベルでも、`INSERT/UPDATE/DELETE/DROP` 等の破壊的キーワードが含まれていないかを厳格に監視します。
5. **Format (結果整形)**
   - 実行結果（行データ）を受け取り、人間が理解しやすい自然言語のサマリー（例: 「合計は 1,500,000 です」）に整形します。

### Fail-safe と Reason Codes

解析や実行に失敗した場合は、フォールバック（RAG）に回さず、原因を明示した Fail-safe 応答を返します。
主な `error_code` (reason code) は以下の通りです：

- `unknown_dataset`: 存在しないデータセット（テーブル）が指定された
- `unknown_operation`: サポートされていない操作（集計方法）が指定された
- `unknown_field`: 存在しない、または許可されていないカラムが指定された
- `write_operation_blocked`: 書き込み（更新・削除等）を試みた
- `validation_failed`: 許可されていないフィールドや条件の組み合わせ
- `ValueError` / `ExecutionError`: データベース実行時やバリデーション時の予期せぬエラー

## 3. Calculator との棲み分け

初期実装から存在する `calculator` ルートは、数式評価や四則演算といった deterministic な「単純計算」のための軽量経路として残していますが、現在は **deprecated（非推奨）** の扱いです。

**業務的な価値を持つ構造化問い合わせ（ランキング、集計、平均など）は、すべて `structured_query_tool` に集約**します。RAG（非構造化）、Compare（比較）、Structured Query（構造化）という3本柱でユースケースをカバーする方針です。

## 4. 将来拡張（Future Work）

長期的には本格的なデータベース接続と、より高度な推論を持つ **SQL Agent** への発展を見据えています。

- **Limited Text-to-SQL**
  - 完全な自由生成ではなく、特定のビューや限定されたスキーマに対して、LLM が SQL を生成・実行する機能。
- **Schema-aware Routing**
  - DB のスキーマメタデータを LLM Router に与え、より高度な意図判定とデータセット選択を実現。
- **PostgreSQL / Read-only DB 連携**
  - 現在のローカル SQLite から、本番環境のリードレプリカやデータウェアハウスへの接続への拡張。
- **Stronger SQL Validation**
  - 実行前の SQL を AST（抽象構文木）レベルで解析し、カラム単位の権限管理や複雑な JOIN の可視化を行う。
- **Query Plan Observability**
  - 実際に構築・実行された SQL や、実行時間、スキャン行数などの可視化。
