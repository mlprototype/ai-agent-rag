# 責務: 構造化クエリに関連するデータモデル（Intent, Result, ValidationResult）およびデータソースインターフェースの定義
# 主な入出力: ドメイン全体で使用される型定義の提供
# 設計上の注意点: Pydantic を用いて型安全性を担保し、DataSource の抽象基底クラスを定義する

from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel

class StructuredQueryIntent(BaseModel):
    operation: Literal["sum", "avg", "count", "top_k", "max", "min", "unknown"]
    target_metric: str | None
    filters: dict[str, Any]
    target_dataset: Literal["sales", "inventory", "unknown"]

class StructuredQueryResult(BaseModel):
    success: bool
    operation: str
    target_metric: str | None
    filters: dict[str, Any]
    rows: list[dict[str, Any]]
    summary: str
    source_name: str
    error_message: str | None = None

class ValidationResult(BaseModel):
    is_valid: bool
    error_code: str | None = None
    error_message: str | None = None

class StructuredDataSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def execute(self, intent: StructuredQueryIntent) -> list[dict[str, Any]]:
        pass
