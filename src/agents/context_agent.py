"""
上下文管理Agent

负责管理会话上下文和状态
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent


class ContextAgent(BaseAgent):
    """上下文管理Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ContextAgent", config)
        self.max_context_size = self.get_config("max_context_size", 1000)
        self.session_timeout = self.get_config("session_timeout", 3600)  # 1小时
        self.context_history: List[Dict[str, Any]] = []
        self.session_start_time = datetime.now()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理上下文管理请求"""
        try:
            # 验证输入
            if not await self.validate_input(input_data):
                raise ValueError("输入数据验证失败")
            
            # 预处理
            processed_input = await self.preprocess(input_data)
            
            # 获取操作类型
            operation = processed_input.get("operation", "update")
            context_data = processed_input.get("context_data", {})
            
            # 执行上下文操作
            context_result = await self._manage_context(operation, context_data)
            
            # 创建推理步骤
            reasoning_step = await self.create_reasoning_step(
                step_type="context_management",
                input_data=str(processed_input),
                output_data=str(context_result),
                confidence=1.0
            )
            
            # 构建输出
            output = {
                "operation": operation,
                "context_data": context_data,
                "result": context_result,
                "reasoning_step": reasoning_step,
                "success": True
            }
            
            # 更新上下文
            self.update_context("context_operation", output)
            
            # 后处理
            return await self.postprocess(output)
            
        except Exception as e:
            await self.handle_error(e)
            return {
                "success": False,
                "error": str(e),
                "operation": input_data.get("operation", "")
            }
    
    async def _manage_context(self, operation: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """管理上下文"""
        if operation == "update":
            return await self._update_context(context_data)
        elif operation == "get":
            return await self._get_context(context_data)
        elif operation == "clear":
            return await self._clear_context()
        elif operation == "history":
            return await self._get_history()
        else:
            raise ValueError(f"不支持的上下文操作: {operation}")
    
    async def _update_context(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """更新上下文"""
        # 添加时间戳
        context_data["timestamp"] = datetime.now().isoformat()
        
        # 更新当前上下文
        for key, value in context_data.items():
            if key != "timestamp":
                self.update_context(key, value)
        
        # 添加到历史记录
        self.context_history.append({
            "operation": "update",
            "data": context_data,
            "timestamp": datetime.now().isoformat()
        })
        
        # 限制历史记录大小
        if len(self.context_history) > self.max_context_size:
            self.context_history = self.context_history[-self.max_context_size:]
        
        return {
            "status": "updated",
            "context_keys": list(self.context.keys()),
            "history_size": len(self.context_history)
        }
    
    async def _get_context(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """获取上下文"""
        keys = context_data.get("keys", [])
        
        if not keys:
            # 返回所有上下文
            return {
                "context": self.context.copy(),
                "session_duration": (datetime.now() - self.session_start_time).total_seconds()
            }
        else:
            # 返回指定键的上下文
            result = {}
            for key in keys:
                if key in self.context:
                    result[key] = self.context[key]
            
            return {
                "context": result,
                "requested_keys": keys,
                "found_keys": list(result.keys())
            }
    
    async def _clear_context(self) -> Dict[str, Any]:
        """清除上下文"""
        cleared_keys = list(self.context.keys())
        self.clear_context()
        
        # 添加到历史记录
        self.context_history.append({
            "operation": "clear",
            "cleared_keys": cleared_keys,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "status": "cleared",
            "cleared_keys": cleared_keys,
            "remaining_history_size": len(self.context_history)
        }
    
    async def _get_history(self) -> Dict[str, Any]:
        """获取历史记录"""
        return {
            "history": self.context_history,
            "total_entries": len(self.context_history),
            "session_start": self.session_start_time.isoformat(),
            "session_duration": (datetime.now() - self.session_start_time).total_seconds()
        }
    
    def get_session_info(self) -> Dict[str, Any]:
        """获取会话信息"""
        return {
            "session_start": self.session_start_time.isoformat(),
            "session_duration": (datetime.now() - self.session_start_time).total_seconds(),
            "context_size": len(self.context),
            "history_size": len(self.context_history),
            "context_keys": list(self.context.keys())
        }
    
    def is_session_expired(self) -> bool:
        """检查会话是否过期"""
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        return session_duration > self.session_timeout
    
    def reset_session(self) -> None:
        """重置会话"""
        self.session_start_time = datetime.now()
        self.clear_context()
        self.context_history.clear()
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据"""
        operation = input_data.get("operation", "")
        context_data = input_data.get("context_data", {})
        
        return (
            operation in ["update", "get", "clear", "history"] and
            isinstance(context_data, dict)
        )
    
    async def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理输入数据"""
        return {
            "operation": input_data.get("operation", "update"),
            "context_data": input_data.get("context_data", {})
        }
    
    async def postprocess(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理输出数据"""
        output_data["metadata"] = {
            "agent": self.name,
            "processing_time": asyncio.get_event_loop().time(),
            "session_duration": (datetime.now() - self.session_start_time).total_seconds(),
            "context_size": len(self.context),
            "history_size": len(self.context_history)
        }
        return output_data 