"""Memory 管理——ConversationBufferMemory / ConversationSummaryMemory"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from src.logging_config import get_logger

logger = get_logger(__name__)


class SessionChatHistory(BaseChatMessageHistory):
    """基于 dict 的内存聊天历史，支持 session_id 隔离。

    用于 Streamlit session_state 兼容的内存存储。
    """

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self._messages: list[dict[str, Any]] = []
        logger.debug("SessionChatHistory created: session_id=%s", session_id)

    def add_user_message(self, message: str) -> None:
        self._messages.append({"role": "user", "content": message})

    def add_ai_message(self, message: str) -> None:
        self._messages.append({"role": "assistant", "content": message})

    def get_messages(self) -> list[HumanMessage | AIMessage]:
        result = []
        for msg in self._messages:
            if msg["role"] == "user":
                result.append(HumanMessage(content=msg["content"]))
            else:
                result.append(AIMessage(content=msg["content"]))
        return result

    def clear(self) -> None:
        self._messages.clear()


class MemoryManager:
    """Memory 管理器——管理会话历史

    支持:
        - 基础的 ConversationBufferMemory（完整历史）
        - 预留 ConversationSummaryMemory（摘要压缩）
    """

    def __init__(self, session_id: str = "default") -> None:
        self.session_id = session_id
        self._chat_history: dict[str, SessionChatHistory] = {}
        self._store: dict[str, Any] = {}  # 预留：扩展用

    def get_chat_history(self, session_id: str | None = None) -> SessionChatHistory:
        """获取指定 session 的聊天历史"""
        sid = session_id or self.session_id
        if sid not in self._chat_history:
            self._chat_history[sid] = SessionChatHistory(session_id=sid)
        return self._chat_history[sid]

    def get_history_for_chain(self, session_id: str | None = None) -> list[HumanMessage | AIMessage]:
        """获取历史消息列表（用于 LangChain chain）"""
        history = self.get_chat_history(session_id)
        logger.debug("Loaded history for session: %s, message_count=%d",
                     session_id or self.session_id, len(history._messages))
        return history.get_messages()

    def add_user_message(self, message: str, session_id: str | None = None) -> None:
        """添加用户消息"""
        history = self.get_chat_history(session_id)
        history.add_user_message(message)

    def add_ai_message(self, message: str, session_id: str | None = None) -> None:
        """添加 AI 消息"""
        history = self.get_chat_history(session_id)
        history.add_ai_message(message)

    def clear(self, session_id: str | None = None) -> None:
        """清除指定 session 的历史"""
        sid = session_id or self.session_id
        if sid in self._chat_history:
            self._chat_history[sid].clear()

    def clear_all(self) -> None:
        """清除所有会话历史"""
        for h in self._chat_history.values():
            h.clear()
