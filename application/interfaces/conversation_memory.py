from abc import ABC, abstractmethod
from langgraph.checkpoint.base import BaseCheckpointSaver

class ConversationMemory(ABC):
    """
    基底の会話メモリインターフェース（Abstract Base Class）。
    PostgreSQLやSQLiteなど、今後のメモリ永続化層の切り替えを容易にするための抽象層です。
    """
    
    @abstractmethod
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """
        LangGraphの状態を保存・復元するための BaseCheckpointSaver インスタンスを返します。
        """
        pass
