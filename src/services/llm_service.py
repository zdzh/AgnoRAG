"""
LLM服务

提供大语言模型的抽象接口和模拟实现
"""

import asyncio
import json
import os
import random
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseService


class LLMService(BaseService):
    """LLM服务抽象类"""
    
    @abstractmethod
    async def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """生成文本"""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, text: str) -> List[float]:
        """生成文本嵌入"""
        pass
    
    @abstractmethod
    async def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        """聊天完成"""
        pass
    
    @abstractmethod
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取实体"""
        pass
    
    @abstractmethod
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询"""
        pass


class MockLLMService(LLMService):
    """模拟LLM服务实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = self.get_config("model_name", "mock-llm")
        self.embedding_dimension = self.get_config("embedding_dimension", 768)
        self.response_delay = self.get_config("response_delay", 0.1)  # 模拟响应延迟
        self.storage_file = self.get_config("storage_file", "data/llm_service/responses.json")
        self.response_cache: Dict[str, str] = {}
    
    async def initialize(self) -> None:
        """初始化服务"""
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        await self._load_cache()
        self._initialized = True
    
    async def shutdown(self) -> None:
        """关闭服务"""
        await self._save_cache()
        self._initialized = False
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "cache_size": len(self.response_cache),
            "initialized": self._initialized
        }
    
    async def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """生成文本"""
        await asyncio.sleep(self.response_delay)  # 模拟延迟
        
        # 检查缓存
        cache_key = f"text:{prompt[:100]}:{max_tokens}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # 根据提示生成模拟响应
        response = await self._generate_mock_response(prompt)
        
        # 缓存响应
        self.response_cache[cache_key] = response
        return response
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """生成文本嵌入"""
        await asyncio.sleep(self.response_delay * 0.5)  # 嵌入生成通常更快
        
        # 检查缓存
        cache_key = f"embedding:{text[:100]}"
        if cache_key in self.response_cache:
            return json.loads(self.response_cache[cache_key])
        
        # 生成模拟嵌入向量
        embeddings = [random.uniform(-1, 1) for _ in range(self.embedding_dimension)]
        
        # 缓存响应
        self.response_cache[cache_key] = json.dumps(embeddings)
        return embeddings
    
    async def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        """聊天完成"""
        await asyncio.sleep(self.response_delay)
        
        # 检查缓存
        cache_key = f"chat:{str(messages)[:100]}:{max_tokens}"
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # 根据消息历史生成模拟响应
        last_message = messages[-1]["content"] if messages else ""
        response = await self._generate_mock_chat_response(messages, last_message)
        
        # 缓存响应
        self.response_cache[cache_key] = response
        return response
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """提取实体"""
        await asyncio.sleep(self.response_delay * 0.3)
        
        # 模拟实体提取
        entities = []
        
        # 简单的规则基础实体提取
        words = text.split()
        for word in words:
            if len(word) > 1 and word[0].isupper():
                entities.append({
                    "text": word,
                    "type": "PERSON" if random.random() > 0.5 else "ORGANIZATION",
                    "confidence": random.uniform(0.7, 1.0)
                })
        
        return entities[:5]  # 限制返回数量
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询"""
        await asyncio.sleep(self.response_delay * 0.2)
        
        # 模拟查询分析
        analysis = {
            "query_type": "multi_hop" if "参与" in query or "项目" in query else "single_hop",
            "entities": [],
            "intent": "information_retrieval",
            "complexity": "medium",
            "estimated_hops": random.randint(1, 3)
        }
        
        # 提取实体
        entities = await self.extract_entities(query)
        analysis["entities"] = entities
        
        return analysis
    
    async def _generate_mock_response(self, prompt: str) -> str:
        """生成模拟响应"""
        if "张三" in prompt:
            return "张三参与了飞天项目。"
        elif "李四" in prompt:
            return "李四与张三在一个项目中工作。"
        elif "项目" in prompt:
            return "飞天项目是一个重要的技术项目，由王五负责管理。"
        elif "参与" in prompt:
            return "根据查询，张三参与了飞天项目。"
        else:
            return f"这是对'{prompt}'的模拟响应。"
    
    async def _generate_mock_chat_response(self, messages: List[Dict[str, str]], last_message: str) -> str:
        """生成模拟聊天响应"""
        if "张三" in last_message:
            return "张三参与了飞天项目，他与李四在同一个项目中工作。"
        elif "李四" in last_message:
            return "李四是飞天项目的成员，与张三一起工作。"
        elif "项目" in last_message:
            return "飞天项目由王五负责管理，张三和李四都是项目成员。"
        else:
            return "我理解您的问题，让我为您提供相关信息。"
    
    async def _save_cache(self) -> None:
        """保存缓存到文件"""
        try:
            data = {
                "responses": self.response_cache,
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "cache_size": len(self.response_cache)
                }
            }
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            await self.handle_error(e)
    
    async def _load_cache(self) -> None:
        """从文件加载缓存"""
        try:
            if not os.path.exists(self.storage_file):
                return
            
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.response_cache = data.get("responses", {})
        
        except Exception as e:
            await self.handle_error(e)
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self.response_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            "cache_size": len(self.response_cache),
            "cache_keys": list(self.response_cache.keys())[:10]  # 只显示前10个键
        } 