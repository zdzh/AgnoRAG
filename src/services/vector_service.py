"""
向量服务

提供向量存储和检索的抽象接口和内存实现
"""

import json
import os
import pickle
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..models import Entity
from .base import BaseService


class VectorService(BaseService):
    """向量服务抽象类"""
    
    @abstractmethod
    async def add_embeddings(self, entity_id: str, embeddings: List[float]) -> bool:
        """添加向量嵌入"""
        pass
    
    @abstractmethod
    async def get_embeddings(self, entity_id: str) -> Optional[List[float]]:
        """获取向量嵌入"""
        pass
    
    @abstractmethod
    async def search_similar(self, query_embeddings: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """搜索相似向量"""
        pass
    
    @abstractmethod
    async def update_embeddings(self, entity_id: str, embeddings: List[float]) -> bool:
        """更新向量嵌入"""
        pass
    
    @abstractmethod
    async def delete_embeddings(self, entity_id: str) -> bool:
        """删除向量嵌入"""
        pass
    
    @abstractmethod
    async def get_all_embeddings(self) -> Dict[str, List[float]]:
        """获取所有向量嵌入"""
        pass


class InMemoryVectorService(VectorService):
    """内存向量服务实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.embeddings: Dict[str, List[float]] = {}
        self.storage_file = self.get_config("storage_file", "data/vector_store/embeddings.pkl")
        self.dimension = self.get_config("dimension", 768)  # 默认向量维度
    
    async def initialize(self) -> None:
        """初始化服务"""
        # 确保存储目录存在
        os.makedirs(os.path.dirname(self.storage_file), exist_ok=True)
        
        # 尝试从文件加载数据
        await self._load_from_file()
        self._initialized = True
    
    async def shutdown(self) -> None:
        """关闭服务"""
        await self._save_to_file()
        self._initialized = False
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            "status": "healthy",
            "embedding_count": len(self.embeddings),
            "dimension": self.dimension,
            "initialized": self._initialized
        }
    
    async def add_embeddings(self, entity_id: str, embeddings: List[float]) -> bool:
        """添加向量嵌入"""
        try:
            if len(embeddings) != self.dimension:
                raise ValueError(f"向量维度不匹配，期望{self.dimension}，实际{len(embeddings)}")
            
            self.embeddings[entity_id] = embeddings
            return True
        except Exception as e:
            await self.handle_error(e)
            return False
    
    async def get_embeddings(self, entity_id: str) -> Optional[List[float]]:
        """获取向量嵌入"""
        return self.embeddings.get(entity_id)
    
    async def search_similar(self, query_embeddings: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """搜索相似向量（使用余弦相似度）"""
        if not self.embeddings:
            return []
        
        if len(query_embeddings) != self.dimension:
            raise ValueError(f"查询向量维度不匹配，期望{self.dimension}，实际{len(query_embeddings)}")
        
        # 计算余弦相似度
        similarities = []
        query_norm = np.linalg.norm(query_embeddings)
        
        for entity_id, embeddings in self.embeddings.items():
            if len(embeddings) == self.dimension:
                # 计算余弦相似度
                dot_product = np.dot(query_embeddings, embeddings)
                embeddings_norm = np.linalg.norm(embeddings)
                
                if query_norm > 0 and embeddings_norm > 0:
                    similarity = dot_product / (query_norm * embeddings_norm)
                    similarities.append((entity_id, float(similarity)))
        
        # 按相似度排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    async def update_embeddings(self, entity_id: str, embeddings: List[float]) -> bool:
        """更新向量嵌入"""
        if entity_id not in self.embeddings:
            return False
        
        return await self.add_embeddings(entity_id, embeddings)
    
    async def delete_embeddings(self, entity_id: str) -> bool:
        """删除向量嵌入"""
        if entity_id in self.embeddings:
            del self.embeddings[entity_id]
            return True
        return False
    
    async def get_all_embeddings(self) -> Dict[str, List[float]]:
        """获取所有向量嵌入"""
        return self.embeddings.copy()
    
    async def batch_add_embeddings(self, embeddings_dict: Dict[str, List[float]]) -> Dict[str, bool]:
        """批量添加向量嵌入"""
        results = {}
        for entity_id, embeddings in embeddings_dict.items():
            results[entity_id] = await self.add_embeddings(entity_id, embeddings)
        return results
    
    async def search_by_entity_name(self, entity_name: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """根据实体名称搜索相似向量"""
        # 这里可以实现基于实体名称的语义搜索
        # 暂时返回空列表，需要结合LLM服务来实现
        return []
    
    async def _save_to_file(self) -> None:
        """保存数据到文件"""
        try:
            data = {
                "embeddings": self.embeddings,
                "dimension": self.dimension,
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "embedding_count": len(self.embeddings)
                }
            }
            
            with open(self.storage_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            await self.handle_error(e)
    
    async def _load_from_file(self) -> None:
        """从文件加载数据"""
        try:
            if not os.path.exists(self.storage_file):
                return
            
            with open(self.storage_file, 'rb') as f:
                data = pickle.load(f)
            
            self.embeddings = data.get("embeddings", {})
            self.dimension = data.get("dimension", self.dimension)
        
        except Exception as e:
            await self.handle_error(e)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取向量存储统计信息"""
        if not self.embeddings:
            return {
                "total_embeddings": 0,
                "average_dimension": 0,
                "dimension_variance": 0
            }
        
        dimensions = [len(emb) for emb in self.embeddings.values()]
        return {
            "total_embeddings": len(self.embeddings),
            "average_dimension": np.mean(dimensions),
            "dimension_variance": np.var(dimensions)
        } 