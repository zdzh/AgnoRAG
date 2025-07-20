"""
基础服务抽象类

定义所有服务的通用接口和功能
"""

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional


class BaseService(ABC):
    """基础服务抽象类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化服务
        
        Args:
            config: 服务配置字典
        """
        self.config = config
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化服务"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """关闭服务"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass
    
    @asynccontextmanager
    async def get_session(self):
        """获取服务会话的上下文管理器"""
        try:
            if not self._initialized:
                await self.initialize()
            yield self
        except Exception as e:
            await self.handle_error(e)
            raise
        finally:
            # 注意：这里不关闭服务，因为可能还有其他地方在使用
            pass
    
    async def handle_error(self, error: Exception) -> None:
        """处理服务错误"""
        # 子类可以重写此方法来实现自定义错误处理
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """设置配置值"""
        self.config[key] = value
    
    def is_initialized(self) -> bool:
        """检查服务是否已初始化"""
        return self._initialized
    
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息"""
        return {
            "service_type": self.__class__.__name__,
            "initialized": self._initialized,
            "config_keys": list(self.config.keys())
        } 