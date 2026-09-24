# 定义消息角色的类型，限制其取值
from typing import Any, Dict, Literal, Optional
from datetime import datetime
from pydantic import BaseModel

# 定义消息角色的类型，限制其取值
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """消息类"""

    role: MessageRole
    content: str
    timestamp: datetime = None  # 可选的时间戳字段，默认为 None
    metadata: Optional[Dict[str, Any]] = None  # 可选的元数据字段，默认为 None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        将消息对象转换为字典形式
        """
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }

    def __str__(self):
        return f"[{self.timestamp}] {self.role}: {self.content}"
