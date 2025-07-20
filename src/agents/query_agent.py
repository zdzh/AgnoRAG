"""
查询分析Agent

负责分析用户查询，提取实体和意图
"""

import asyncio
from typing import Any, Dict, List, Optional

from ..models import ReasoningStep
from .base_agent import BaseAgent


class QueryAgent(BaseAgent):
    """查询分析Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("QueryAgent", config)
        self.max_entities = self.get_config("max_entities", 10)
        self.confidence_threshold = self.get_config("confidence_threshold", 0.7)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理查询分析"""
        try:
            # 验证输入
            if not await self.validate_input(input_data):
                raise ValueError("输入数据验证失败")
            
            # 预处理
            processed_input = await self.preprocess(input_data)
            
            # 提取查询文本
            query = processed_input.get("query", "")
            if not query:
                raise ValueError("查询文本不能为空")
            
            # 分析查询
            analysis_result = await self._analyze_query(query)
            
            # 提取实体
            entities = await self._extract_entities(query)
            
            # 确定推理路径
            reasoning_plan = await self._plan_reasoning(query, entities, analysis_result)
            
            # 创建推理步骤
            reasoning_step = await self.create_reasoning_step(
                step_type="query_analysis",
                input_data=query,
                output_data=str(analysis_result),
                confidence=analysis_result.get("confidence", 0.8)
            )
            
            # 构建输出
            output = {
                "query": query,
                "analysis": analysis_result,
                "entities": entities,
                "reasoning_plan": reasoning_plan,
                "reasoning_step": reasoning_step,
                "success": True
            }
            
            # 更新上下文
            self.update_context("query_analysis", output)
            self.update_context("extracted_entities", entities)
            self.update_context("reasoning_plan", reasoning_plan)
            
            # 后处理
            return await self.postprocess(output)
            
        except Exception as e:
            await self.handle_error(e)
            return {
                "success": False,
                "error": str(e),
                "query": input_data.get("query", "")
            }
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """分析查询"""
        if not self.llm_service:
            raise RuntimeError("LLM服务未初始化")
        
        # 使用LLM服务分析查询
        analysis = await self.llm_service.analyze_query(query)
        
        # 增强分析结果
        enhanced_analysis = {
            "query_type": analysis.get("query_type", "unknown"),
            "intent": analysis.get("intent", "information_retrieval"),
            "complexity": analysis.get("complexity", "medium"),
            "estimated_hops": analysis.get("estimated_hops", 1),
            "confidence": 0.9,  # 模拟置信度
            "requires_multi_hop": analysis.get("query_type") == "multi_hop",
            "entities_count": len(analysis.get("entities", [])),
            "timestamp": asyncio.get_event_loop().time()
        }
        
        return enhanced_analysis
    
    async def _extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """提取实体"""
        if not self.llm_service:
            raise RuntimeError("LLM服务未初始化")
        
        # 使用LLM服务提取实体
        entities = await self.llm_service.extract_entities(query)
        
        # 过滤低置信度的实体
        filtered_entities = [
            entity for entity in entities 
            if entity.get("confidence", 0) >= self.confidence_threshold
        ]
        
        # 限制实体数量
        return filtered_entities[:self.max_entities]
    
    async def _plan_reasoning(self, query: str, entities: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """规划推理路径"""
        reasoning_plan = {
            "query": query,
            "entities": entities,
            "analysis": analysis,
            "steps": [],
            "estimated_duration": 0,
            "complexity": analysis.get("complexity", "medium")
        }
        
        # 根据查询类型和实体数量确定推理步骤
        if analysis.get("requires_multi_hop", False):
            reasoning_plan["steps"] = [
                {"type": "entity_search", "description": "搜索初始实体"},
                {"type": "relation_search", "description": "查找相关关系"},
                {"type": "path_finding", "description": "寻找推理路径"},
                {"type": "information_integration", "description": "整合信息"},
                {"type": "answer_generation", "description": "生成答案"}
            ]
            reasoning_plan["estimated_duration"] = 5.0  # 秒
        else:
            reasoning_plan["steps"] = [
                {"type": "entity_search", "description": "搜索实体"},
                {"type": "direct_answer", "description": "直接生成答案"}
            ]
            reasoning_plan["estimated_duration"] = 2.0  # 秒
        
        return reasoning_plan
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据"""
        query = input_data.get("query", "")
        return isinstance(query, str) and len(query.strip()) > 0
    
    async def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理输入数据"""
        # 清理查询文本
        query = input_data.get("query", "").strip()
        return {"query": query}
    
    async def postprocess(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理输出数据"""
        # 添加元数据
        output_data["metadata"] = {
            "agent": self.name,
            "processing_time": asyncio.get_event_loop().time(),
            "entities_count": len(output_data.get("entities", [])),
            "analysis_confidence": output_data.get("analysis", {}).get("confidence", 0.0)
        }
        return output_data
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        analysis = self.get_context("query_analysis")
        if not analysis:
            return {}
        
        return {
            "query": analysis.get("query", ""),
            "query_type": analysis.get("analysis", {}).get("query_type", "unknown"),
            "entities_count": len(analysis.get("entities", [])),
            "requires_multi_hop": analysis.get("analysis", {}).get("requires_multi_hop", False),
            "estimated_hops": analysis.get("analysis", {}).get("estimated_hops", 1)
        } 