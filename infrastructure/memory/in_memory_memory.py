from langgraph.checkpoint.memory import MemorySaver
from application.interfaces.conversation_memory import ConversationMemory

class InMemoryConversationMemory(ConversationMemory):
    """
    メモリ（RAM）上で会話の履歴を管理するインメモリ実装。
    LangGraph標準の MemorySaver をラップし、再起動で履歴は失われます。
    """
    
    def __init__(self):
        self._checkpointer = MemorySaver()
        
    def get_checkpointer(self):
        return self._checkpointer

# シングルトンインスタンスを提供（依存性注入を簡単にするため）
in_memory_memory = InMemoryConversationMemory()
