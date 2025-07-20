"""
图数据库Agent

负责在图数据库中执行复杂的查询和路径查找
"""

import asyncio
from typing import Any, Dict, List, Optional

from ..models import Entity, Relation
from .base_agent import BaseAgent


class GraphAgent(BaseAgent):
    """图数据库Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("GraphAgent", config)
        self.max_path_length = self.get_config("max_path_length", 5)
        self.max_neighbors = self.get_config("max_neighbors", 10)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理图数据库查询"""
        try:
            # 验证输入
            if not await self.validate_input(input_data):
                raise ValueError("输入数据验证失败")
            
            # 预处理
            processed_input = await self.preprocess(input_data)
            
            # 获取查询参数
            query_type = processed_input.get("query_type", "entity_search")
            entities = processed_input.get("entities", [])
            target_entities = processed_input.get("target_entities", [])
            
            # 执行图查询
            graph_results = await self._execute_graph_query(query_type, entities, target_entities)
            
            # 创建推理步骤
            reasoning_step = await self.create_reasoning_step(
                step_type="graph_query",
                input_data=str(processed_input),
                output_data=str(graph_results),
                confidence=graph_results.get("confidence", 0.8)
            )
            
            # 构建输出
            output = {
                "query_type": query_type,
                "entities": entities,
                "target_entities": target_entities,
                "results": graph_results,
                "reasoning_step": reasoning_step,
                "success": True
            }
            
            # 更新上下文
            self.update_context("graph_results", output)
            
            # 后处理
            return await self.postprocess(output)
            
        except Exception as e:
            await self.handle_error(e)
            return {
                "success": False,
                "error": str(e),
                "query_type": input_data.get("query_type", "")
            }
    
    async def _execute_graph_query(self, query_type: str, entities: List[Dict[str, Any]], target_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行图数据库查询"""
        if not self.graph_service:
            return {"confidence": 0.0, "results": []}
        
        results = {
            "confidence": 0.0,
            "results": [],
            "paths": [],
            "relations": []
        }
        
        if query_type == "entity_search":
            results = await self._search_entities(entities)
        elif query_type == "path_finding":
            results = await self._find_paths(entities, target_entities)
        elif query_type == "relation_search":
            results = await self._search_relations(entities)
        elif query_type == "neighbor_search":
            results = await self._search_neighbors(entities)
        
        return results
    
    async def _search_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """搜索实体"""
        results = []
        
        for entity_info in entities:
            entity_name = entity_info.get("text", "")
            if entity_name:
                found_entities = await self.graph_service.search_entities(entity_name)
                for entity in found_entities:
                    results.append({
                        "entity": entity.to_dict(),
                        "source_query": entity_name,
                        "confidence": entity_info.get("confidence", 0.8)
                    })
        
        return {
            "confidence": 0.9 if results else 0.0,
            "results": results,
            "paths": [],
            "relations": []
        }
    
    async def _find_paths(self, source_entities: List[Dict[str, Any]], target_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """查找路径"""
        paths = []
        
        for source_info in source_entities:
            source_name = source_info.get("text", "")
            source_entities_found = await self.graph_service.search_entities(source_name)
            
            for target_info in target_entities:
                target_name = target_info.get("text", "")
                target_entities_found = await self.graph_service.search_entities(target_name)
                
                for source_entity in source_entities_found:
                    for target_entity in target_entities_found:
                        path = await self.graph_service.find_path(
                            source_entity.id, 
                            target_entity.id, 
                            max_hops=self.max_path_length
                        )
                        
                        if path:
                            paths.append({
                                "source": source_entity.to_dict(),
                                "target": target_entity.to_dict(),
                                "path": [rel.to_dict() for rel in path],
                                "length": len(path),
                                "confidence": min(source_info.get("confidence", 0.8), target_info.get("confidence", 0.8))
                            })
        
        return {
            "confidence": 0.8 if paths else 0.0,
            "results": [],
            "paths": paths,
            "relations": []
        }
    
    async def _search_relations(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """搜索关系"""
        relations = []
        
        for entity_info in entities:
            entity_name = entity_info.get("text", "")
            if entity_name:
                found_entities = await self.graph_service.search_entities(entity_name)
                for entity in found_entities:
                    entity_relations = await self.graph_service.get_relations(entity.id)
                    for relation in entity_relations:
                        relations.append({
                            "relation": relation.to_dict(),
                            "source_entity": entity.to_dict(),
                            "confidence": entity_info.get("confidence", 0.8)
                        })
        
        return {
            "confidence": 0.9 if relations else 0.0,
            "results": [],
            "paths": [],
            "relations": relations
        }
    
    async def _search_neighbors(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """搜索邻居"""
        neighbors = []
        
        for entity_info in entities:
            entity_name = entity_info.get("text", "")
            if entity_name:
                found_entities = await self.graph_service.search_entities(entity_name)
                for entity in found_entities:
                    entity_neighbors = await self.graph_service.get_entity_neighbors(entity.id)
                    for neighbor in entity_neighbors[:self.max_neighbors]:
                        neighbors.append({
                            "neighbor": neighbor.to_dict(),
                            "source_entity": entity.to_dict(),
                            "confidence": entity_info.get("confidence", 0.8)
                        })
        
        return {
            "confidence": 0.9 if neighbors else 0.0,
            "results": neighbors,
            "paths": [],
            "relations": []
        }
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据"""
        query_type = input_data.get("query_type", "")
        entities = input_data.get("entities", [])
        
        return (
            query_type in ["entity_search", "path_finding", "relation_search", "neighbor_search"] and
            isinstance(entities, list)
        )
    
    async def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理输入数据"""
        return {
            "query_type": input_data.get("query_type", "entity_search"),
            "entities": input_data.get("entities", []),
            "target_entities": input_data.get("target_entities", [])
        }
    
    async def postprocess(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理输出数据"""
        output_data["metadata"] = {
            "agent": self.name,
            "processing_time": asyncio.get_event_loop().time(),
            "results_count": len(output_data.get("results", {}).get("results", [])),
            "paths_count": len(output_data.get("results", {}).get("paths", [])),
            "relations_count": len(output_data.get("results", {}).get("relations", []))
        }
        return output_data 