"""
搜索Agent

负责在知识图谱和向量存储中搜索相关信息
"""

import asyncio
from typing import Any, Dict, List, Optional

from ..models import Entity, Relation
from .base_agent import BaseAgent


class SearchAgent(BaseAgent):
    """搜索Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SearchAgent", config)
        self.max_results = self.get_config("max_results", 20)
        self.similarity_threshold = self.get_config("similarity_threshold", 0.5)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理搜索请求"""
        try:
            # 验证输入
            if not await self.validate_input(input_data):
                raise ValueError("输入数据验证失败")
            
            # 预处理
            processed_input = await self.preprocess(input_data)
            
            # 获取搜索参数
            query = processed_input.get("query", "")
            entities = processed_input.get("entities", [])
            search_type = processed_input.get("search_type", "hybrid")
            
            # 执行搜索
            search_results = await self._perform_search(query, entities, search_type)
            
            # 创建推理步骤
            reasoning_step = await self.create_reasoning_step(
                step_type="information_search",
                input_data=str(processed_input),
                output_data=str(search_results),
                confidence=search_results.get("confidence", 0.8)
            )
            
            # 构建输出
            output = {
                "query": query,
                "entities": entities,
                "search_type": search_type,
                "results": search_results,
                "reasoning_step": reasoning_step,
                "success": True
            }
            
            # 更新上下文
            self.update_context("search_results", output)
            
            # 后处理
            return await self.postprocess(output)
            
        except Exception as e:
            await self.handle_error(e)
            return {
                "success": False,
                "error": str(e),
                "query": input_data.get("query", "")
            }
    
    async def _perform_search(self, query: str, entities: List[Dict[str, Any]], search_type: str) -> Dict[str, Any]:
        """执行搜索"""
        results = {
            "graph_results": [],
            "vector_results": [],
            "combined_results": [],
            "confidence": 0.0,
            "total_found": 0
        }
        
        # 图数据库搜索
        if search_type in ["graph", "hybrid"]:
            graph_results = await self._search_graph(query, entities)
            results["graph_results"] = graph_results
        
        # 向量搜索
        if search_type in ["vector", "hybrid"]:
            vector_results = await self._search_vector(query)
            results["vector_results"] = vector_results
        
        # 合并结果
        if search_type == "hybrid":
            results["combined_results"] = await self._combine_results(
                results["graph_results"], 
                results["vector_results"]
            )
        
        # 计算总体置信度
        results["confidence"] = self._calculate_confidence(results)
        results["total_found"] = len(results["combined_results"])
        
        return results
    
    async def _search_graph(self, query: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """在图数据库中搜索"""
        if not self.graph_service:
            return []
        
        graph_results = []
        
        # 根据实体搜索
        for entity_info in entities:
            entity_name = entity_info.get("text", "")
            if entity_name:
                # 搜索实体
                entities_found = await self.graph_service.search_entities(entity_name)
                
                for entity in entities_found:
                    # 获取实体的关系
                    relations = await self.graph_service.get_relations(entity.id)
                    
                    # 获取邻居实体
                    neighbors = await self.graph_service.get_entity_neighbors(entity.id)
                    
                    graph_results.append({
                        "entity": entity.to_dict(),
                        "relations": [rel.to_dict() for rel in relations],
                        "neighbors": [neighbor.to_dict() for neighbor in neighbors],
                        "source": "graph_search",
                        "confidence": entity_info.get("confidence", 0.8)
                    })
        
        return graph_results[:self.max_results]
    
    async def _search_vector(self, query: str) -> List[Dict[str, Any]]:
        """在向量存储中搜索"""
        if not self.vector_service or not self.llm_service:
            return []
        
        # 生成查询向量
        query_embeddings = await self.llm_service.generate_embeddings(query)
        
        # 搜索相似向量
        similar_entities = await self.vector_service.search_similar(
            query_embeddings, 
            top_k=self.max_results
        )
        
        vector_results = []
        for entity_id, similarity in similar_entities:
            if similarity >= self.similarity_threshold:
                # 获取实体信息
                if self.graph_service:
                    entity = await self.graph_service.get_entity(entity_id)
                    if entity:
                        vector_results.append({
                            "entity": entity.to_dict(),
                            "similarity": similarity,
                            "source": "vector_search",
                            "confidence": similarity
                        })
        
        return vector_results
    
    async def _combine_results(self, graph_results: List[Dict[str, Any]], vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并搜索结果"""
        combined = []
        entity_ids = set()
        
        # 添加图搜索结果
        for result in graph_results:
            entity_id = result["entity"]["id"]
            if entity_id not in entity_ids:
                combined.append(result)
                entity_ids.add(entity_id)
        
        # 添加向量搜索结果
        for result in vector_results:
            entity_id = result["entity"]["id"]
            if entity_id not in entity_ids:
                combined.append(result)
                entity_ids.add(entity_id)
        
        # 按置信度排序
        combined.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return combined[:self.max_results]
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """计算搜索置信度"""
        if not results["combined_results"]:
            return 0.0
        
        confidences = [result.get("confidence", 0) for result in results["combined_results"]]
        return sum(confidences) / len(confidences)
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据"""
        query = input_data.get("query", "")
        entities = input_data.get("entities", [])
        search_type = input_data.get("search_type", "hybrid")
        
        return (
            isinstance(query, str) and len(query.strip()) > 0 and
            isinstance(entities, list) and
            search_type in ["graph", "vector", "hybrid"]
        )
    
    async def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理输入数据"""
        return {
            "query": input_data.get("query", "").strip(),
            "entities": input_data.get("entities", []),
            "search_type": input_data.get("search_type", "hybrid")
        }
    
    async def postprocess(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理输出数据"""
        output_data["metadata"] = {
            "agent": self.name,
            "processing_time": asyncio.get_event_loop().time(),
            "results_count": output_data.get("results", {}).get("total_found", 0),
            "search_confidence": output_data.get("results", {}).get("confidence", 0.0)
        }
        return output_data 