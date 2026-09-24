from abc import ABC, abstractmethod
from typing import Optional
from .llm import MyLLM
from .config import Config
from .message import Message


class Agent(ABC):
    """Agent基类"""

    def __init__(
        self,
        name: str,
        llm: MyLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self.history = list[Message] = []  # 用于记录对话历史和工具调用结果

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent"""
        pass

    def add_message(self, message: Message):
        """添加消息到历史记录"""
        self._history.append(message)

    def clear_history(self):
        """清空历史记录"""
        self._history.clear()

    def get_history(self) -> list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name})"
