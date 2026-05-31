"""
このファイルは、評価実行結果のデータ構造（スキーマ）を定義します。
Pydanticモデルを使用し、評価レコード、集計サマリー、および最終的なレポート出力形式を規定します。
主な入力は評価実行時の各クエリ結果であり、出力は構造化されたJSONレポートのベースとなります。
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class EvalRecord(BaseModel):
    """個別のクエリ評価結果を保持するモデル。"""
    query: str
    query_type: str
    route: Optional[str] = None
    answer: str
    confidence: float
    fallback_level: str = "NONE"
    latency_ms: int
    retrieval_quality_level: str = "NONE"
    source_name: Optional[str] = None
    similarity: float
    reason_code: str = "SUCCESS"
    
    # 成功指標の2軸分解
    response_generated: bool = Field(..., description="システムが回答テキストを生成できたか")
    answer_ok: bool = Field(..., description="回答が品質基準（Critic等）を満たしているか")
    
    # 縮退フラグの分解
    degraded: bool = False
    critic_degraded: bool = False
    retrieval_degraded: bool = False
    strict_insufficient_response: bool = False
    
    # 警告関連
    warning: bool = False
    warning_codes: List[str] = Field(default_factory=list)
    
    # Route drift 拡張用
    expected_query_type: Optional[str] = None
    expected_route: Optional[str] = None

class EvalSummary(BaseModel):
    """全体またはカテゴリ別の集計指標を保持するモデル。"""
    total_count: int
    response_generated_rate: float
    answer_ok_rate: float
    warning_rate: float
    fallback_rate: float
    degraded_rate: float
    avg_confidence: float
    avg_similarity: float
    latency_p50_ms: float
    latency_p95_ms: float

class EvalDistributions(BaseModel):
    """各種コードやレベルの発生分布（Top N）を保持するモデル。"""
    warning_codes: Dict[str, int] = Field(default_factory=dict)
    fallback_level: Dict[str, int] = Field(default_factory=dict)
    retrieval_quality_level: Dict[str, int] = Field(default_factory=dict)
    reason_code: Dict[str, int] = Field(default_factory=dict)

class EvalReport(BaseModel):
    """最終的な評価レポートのトップレベル構造。"""
    metadata: Dict[str, Any]
    summary: EvalSummary
    by_query_type: Dict[str, EvalSummary]
    by_route: Dict[str, EvalSummary]
    distributions: EvalDistributions
    records: List[EvalRecord]
