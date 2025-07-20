"""
图数据库服务

提供知识图谱的抽象接口和内存实现
"""

import json
import os
from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from ..models import Entity, Relation
from .base import BaseService


class GraphService(BaseService):
    """图数据库服务抽象类"""
    
    @abstractmethod
    async def add_entity(self, entity: Entity) -> bool:
        """添加实体"""
        pass
    
    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        pass
    
    @abstractmethod
    async def update_entity(self, entity: Entity) -> bool:
        """更新实体"""
        pass
    
    @abstractmethod
    async def delete_entity(self, entity_id: str) -> bool:
        """删除实体"""
        pass
    
    @abstractmethod
    async def add_relation(self, relation: Relation) -> bool:
        """添加关系"""
        pass
    
    @abstractmethod
    async def get_relations(self, entity_id: str, relation_type: Optional[str] = None) -> List[Relation]:
        """获取实体的关系"""
        pass
    
    @abstractmethod
    async def find_path(self, source_id: str, target_id: str, max_hops: int = 3) -> List[Relation]:
        """查找两个实体间的路径"""
        pass
    
    @abstractmethod
    async def search_entities(self, query: str, entity_type: Optional[str] = None, limit: int = 10) -> List[Entity]:
        """搜索实体"""
        pass
    
    @abstractmethod
    async def get_entity_neighbors(self, entity_id: str, relation_type: Optional[str] = None) -> List[Entity]:
        """获取实体的邻居"""
        pass


class InMemoryGraphService(GraphService):
    """内存图数据库服务实现"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.entity_name_index: Dict[str, Set[str]] = {}  # 名称到ID的索引
        self.entity_type_index: Dict[str, Set[str]] = {}  # 类型到ID的索引
        self.relation_index: Dict[str, List[str]] = {}  # 实体ID到关系ID的索引
        self.storage_file = self.get_config("storage_file", "data/knowledge_graph/graph_data.json")
    
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
            "entity_count": len(self.entities),
            "relation_count": len(self.relations),
            "initialized": self._initialized
        }
    
    async def add_entity(self, entity: Entity) -> bool:
        """添加实体"""
        try:
            self.entities[entity.id] = entity
            
            # 更新索引
            if entity.name not in self.entity_name_index:
                self.entity_name_index[entity.name] = set()
            self.entity_name_index[entity.name].add(entity.id)
            
            if entity.type not in self.entity_type_index:
                self.entity_type_index[entity.type] = set()
            self.entity_type_index[entity.type].add(entity.id)
            
            return True
        except Exception as e:
            await self.handle_error(e)
            return False
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self.entities.get(entity_id)
    
    async def update_entity(self, entity: Entity) -> bool:
        """更新实体"""
        if entity.id not in self.entities:
            return False
        
        # 更新索引
        old_entity = self.entities[entity.id]
        if old_entity.name != entity.name:
            self.entity_name_index[old_entity.name].discard(entity.id)
            if not self.entity_name_index[old_entity.name]:
                del self.entity_name_index[old_entity.name]
            
            if entity.name not in self.entity_name_index:
                self.entity_name_index[entity.name] = set()
            self.entity_name_index[entity.name].add(entity.id)
        
        if old_entity.type != entity.type:
            self.entity_type_index[old_entity.type].discard(entity.id)
            if not self.entity_type_index[old_entity.type]:
                del self.entity_type_index[old_entity.type]
            
            if entity.type not in self.entity_type_index:
                self.entity_type_index[entity.type] = set()
            self.entity_type_index[entity.type].add(entity.id)
        
        self.entities[entity.id] = entity
        return True
    
    async def delete_entity(self, entity_id: str) -> bool:
        """删除实体"""
        if entity_id not in self.entities:
            return False
        
        entity = self.entities[entity_id]
        
        # 删除相关的关系
        relations_to_delete = []
        for relation_id, relation in self.relations.items():
            if relation.source_id == entity_id or relation.target_id == entity_id:
                relations_to_delete.append(relation_id)
        
        for relation_id in relations_to_delete:
            await self.delete_relation(relation_id)
        
        # 更新索引
        self.entity_name_index[entity.name].discard(entity_id)
        if not self.entity_name_index[entity.name]:
            del self.entity_name_index[entity.name]
        
        self.entity_type_index[entity.type].discard(entity_id)
        if not self.entity_type_index[entity.type]:
            del self.entity_type_index[entity.type]
        
        del self.entities[entity_id]
        return True
    
    async def add_relation(self, relation: Relation) -> bool:
        """添加关系"""
        try:
            # 验证关系
            if not relation.is_valid():
                return False
            
            # 检查实体是否存在
            if relation.source_id not in self.entities or relation.target_id not in self.entities:
                return False
            
            self.relations[relation.id] = relation
            
            # 更新关系索引
            if relation.source_id not in self.relation_index:
                self.relation_index[relation.source_id] = []
            self.relation_index[relation.source_id].append(relation.id)
            
            return True
        except Exception as e:
            await self.handle_error(e)
            return False
    
    async def delete_relation(self, relation_id: str) -> bool:
        """删除关系"""
        if relation_id not in self.relations:
            return False
        
        relation = self.relations[relation_id]
        
        # 更新关系索引
        if relation.source_id in self.relation_index:
            self.relation_index[relation.source_id] = [
                rid for rid in self.relation_index[relation.source_id] 
                if rid != relation_id
            ]
        
        del self.relations[relation_id]
        return True
    
    async def get_relations(self, entity_id: str, relation_type: Optional[str] = None) -> List[Relation]:
        """获取实体的关系"""
        if entity_id not in self.relation_index:
            return []
        
        relation_ids = self.relation_index[entity_id]
        relations = []
        
        for rid in relation_ids:
            if rid in self.relations:
                relation = self.relations[rid]
                if relation_type is None or relation.relation_type == relation_type:
                    relations.append(relation)
        
        return relations
    
    async def find_path(self, source_id: str, target_id: str, max_hops: int = 3) -> List[Relation]:
        """查找两个实体间的路径（使用BFS）"""
        if source_id == target_id:
            return []
        
        visited = set()
        queue = [(source_id, [])]
        
        while queue and len(queue[0][1]) < max_hops:
            current_id, path = queue.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # 获取当前实体的所有关系
            relations = await self.get_relations(current_id)
            
            for relation in relations:
                next_id = relation.target_id if relation.source_id == current_id else relation.source_id
                
                if next_id == target_id:
                    return path + [relation]
                
                if next_id not in visited:
                    queue.append((next_id, path + [relation]))
        
        return []
    
    async def search_entities(self, query: str, entity_type: Optional[str] = None, limit: int = 10) -> List[Entity]:
        """搜索实体"""
        results = []
        
        # 按名称搜索
        for name, entity_ids in self.entity_name_index.items():
            if query.lower() in name.lower():
                for entity_id in entity_ids:
                    if entity_id in self.entities:
                        entity = self.entities[entity_id]
                        if entity_type is None or entity.type == entity_type:
                            results.append(entity)
        
        # 按类型过滤
        if entity_type and entity_type in self.entity_type_index:
            for entity_id in self.entity_type_index[entity_type]:
                if entity_id in self.entities:
                    entity = self.entities[entity_id]
                    if entity not in results:
                        results.append(entity)
        
        # 限制结果数量
        return results[:limit]
    
    async def get_entity_neighbors(self, entity_id: str, relation_type: Optional[str] = None) -> List[Entity]:
        """获取实体的邻居"""
        relations = await self.get_relations(entity_id, relation_type)
        neighbors = []
        
        for relation in relations:
            neighbor_id = relation.target_id if relation.source_id == entity_id else relation.source_id
            if neighbor_id in self.entities:
                neighbors.append(self.entities[neighbor_id])
        
        return neighbors
    
    async def _save_to_file(self) -> None:
        """保存数据到文件"""
        try:
            data = {
                "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
                "relations": {rid: relation.to_dict() for rid, relation in self.relations.items()},
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "entity_count": len(self.entities),
                    "relation_count": len(self.relations)
                }
            }
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            await self.handle_error(e)
    
    async def _load_from_file(self) -> None:
        """从文件加载数据"""
        try:
            if not os.path.exists(self.storage_file):
                return
            
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载实体
            for eid, entity_data in data.get("entities", {}).items():
                entity = Entity.from_dict(entity_data)
                self.entities[eid] = entity
                
                # 重建索引
                if entity.name not in self.entity_name_index:
                    self.entity_name_index[entity.name] = set()
                self.entity_name_index[entity.name].add(entity.id)
                
                if entity.type not in self.entity_type_index:
                    self.entity_type_index[entity.type] = set()
                self.entity_type_index[entity.type].add(entity.id)
            
            # 加载关系
            for rid, relation_data in data.get("relations", {}).items():
                relation = Relation.from_dict(relation_data)
                self.relations[rid] = relation
                
                # 重建关系索引
                if relation.source_id not in self.relation_index:
                    self.relation_index[relation.source_id] = []
                self.relation_index[relation.source_id].append(relation.id)
        
        except Exception as e:
            await self.handle_error(e) 