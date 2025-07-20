"""
Agent基类

定义所有Agent的通用接口和功能
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import ReasoningStep
from ..services import GraphService, LLMService, VectorService


class BaseAgent(ABC):
    """Agent基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化Agent
        
        Args:
            name: Agent名称
            config: Agent配置
        """
        self.name = name
        self.config = config
        self.graph_service: Optional[GraphService] = None
        self.vector_service: Optional[VectorService] = None
        self.llm_service: Optional[LLMService] = None
        self.tools: List[Any] = []
        self.context: Dict[str, Any] = {}
        self._initialized = False
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理输入数据"""
        pass
    
    async def setup(self, graph_service: GraphService, vector_service: VectorService, llm_service: LLMService) -> None:
        """设置Agent依赖的服务"""
        self.graph_service = graph_service
        self.vector_service = vector_service
        self.llm_service = llm_service
        self._initialized = True
    
    async def initialize(self) -> None:
        """初始化Agent"""
        if not self._initialized:
            raise RuntimeError("Agent必须先设置服务依赖")
    
    async def shutdown(self) -> None:
        """关闭Agent"""
        self._initialized = False
    
    def add_tool(self, tool: Any) -> None:
        """添加工具"""
        self.tools.append(tool)
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """获取工具"""
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name == tool_name:
                return tool
        return None
    
    def update_context(self, key: str, value: Any) -> None:
        """更新上下文"""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """获取上下文"""
        return self.context.get(key, default)
    
    def clear_context(self) -> None:
        """清除上下文"""
        self.context.clear()
    
    async def create_reasoning_step(self, step_type: str, input_data: str, output_data: str, confidence: float = 1.0) -> ReasoningStep:
        """创建推理步骤"""
        return ReasoningStep(
            id=str(uuid.uuid4()),
            path_id=self.get_context("reasoning_path_id", ""),
            step_type=step_type,
            input=input_data,
            output=output_data,
            step_order=len(self.get_context("current_steps", [])),
            confidence=confidence,
            metadata={
                "agent": self.name,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """设置配置值"""
        self.config[key] = value
    
    def is_initialized(self) -> bool:
        """检查Agent是否已初始化"""
        return self._initialized
    
    def get_agent_info(self) -> Dict[str, Any]:
        """获取Agent信息"""
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "initialized": self._initialized,
            "tools_count": len(self.tools),
            "context_keys": list(self.context.keys())
        }
    
    async def handle_error(self, error: Exception) -> None:
        """处理Agent错误"""
        error_info = {
            "agent": self.name,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        # 更新上下文中的错误信息
        self.update_context("last_error", error_info)
        
        # 子类可以重写此方法来实现自定义错误处理
        raise error
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据"""
        # 子类可以重写此方法来实现输入验证
        return True
    
    async def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理输入数据"""
        # 子类可以重写此方法来实现预处理
        return input_data
    
    async def postprocess(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理输出数据"""
        # 子类可以重写此方法来实现后处理
        return output_data 