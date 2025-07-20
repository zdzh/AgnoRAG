"""
Agentic RAG 系统

多跳问答系统的核心实现
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .agents import (
    AnswerAgent,
    ContextAgent,
    GraphAgent,
    QueryAgent,
    ReasoningAgent,
    SearchAgent,
)
from .models import ReasoningPath, ReasoningStep
from .services import InMemoryGraphService, InMemoryVectorService, MockLLMService


class AgenticRAGSystem:
    """Agentic RAG 系统主类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化RAG系统
        
        Args:
            config: 系统配置
        """
        self.config = config
        self.session_id = str(uuid.uuid4())
        
        # 初始化服务
        self.graph_service = InMemoryGraphService(config.get("graph_service", {}))
        self.vector_service = InMemoryVectorService(config.get("vector_service", {}))
        self.llm_service = MockLLMService(config.get("llm_service", {}))
        
        # 初始化Agent
        self.query_agent = QueryAgent(config.get("query_agent", {}))
        self.search_agent = SearchAgent(config.get("search_agent", {}))
        self.graph_agent = GraphAgent(config.get("graph_agent", {}))
        self.reasoning_agent = ReasoningAgent(config.get("reasoning_agent", {}))
        self.context_agent = ContextAgent(config.get("context_agent", {}))
        self.answer_agent = AnswerAgent(config.get("answer_agent", {}))
        
        # 推理路径
        self.current_reasoning_path: Optional[ReasoningPath] = None
        
        # 系统状态
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化系统"""
        try:
            # 初始化服务
            await self.graph_service.initialize()
            await self.vector_service.initialize()
            await self.llm_service.initialize()
            
            # 设置Agent依赖
            await self.query_agent.setup(self.graph_service, self.vector_service, self.llm_service)
            await self.search_agent.setup(self.graph_service, self.vector_service, self.llm_service)
            await self.graph_agent.setup(self.graph_service, self.vector_service, self.llm_service)
            await self.reasoning_agent.setup(self.graph_service, self.vector_service, self.llm_service)
            await self.context_agent.setup(self.graph_service, self.vector_service, self.llm_service)
            await self.answer_agent.setup(self.graph_service, self.vector_service, self.llm_service)
            
            # 初始化Agent
            await self.query_agent.initialize()
            await self.search_agent.initialize()
            await self.graph_agent.initialize()
            await self.reasoning_agent.initialize()
            await self.context_agent.initialize()
            await self.answer_agent.initialize()
            
            self._initialized = True
            print("✅ RAG系统初始化完成")
            
        except Exception as e:
            print(f"❌ RAG系统初始化失败: {e}")
            raise
    
    async def shutdown(self) -> None:
        """关闭系统"""
        try:
            await self.graph_service.shutdown()
            await self.vector_service.shutdown()
            await self.llm_service.shutdown()
            
            await self.query_agent.shutdown()
            await self.search_agent.shutdown()
            await self.graph_agent.shutdown()
            await self.reasoning_agent.shutdown()
            await self.context_agent.shutdown()
            await self.answer_agent.shutdown()
            
            self._initialized = False
            print("✅ RAG系统已关闭")
            
        except Exception as e:
            print(f"❌ RAG系统关闭失败: {e}")
    
    async def query(self, user_query: str) -> Dict[str, Any]:
        """
        处理用户查询
        
        Args:
            user_query: 用户查询文本
            
        Returns:
            查询结果字典
        """
        if not self._initialized:
            raise RuntimeError("系统未初始化")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 创建推理路径
            self.current_reasoning_path = ReasoningPath(
                query=user_query,
                created_at=datetime.now()
            )
            
            # 1. 查询分析
            print("🔍 步骤1: 查询分析")
            query_result = await self.query_agent.process({"query": user_query})
            if not query_result.get("success"):
                return self._create_error_response(user_query, "查询分析失败", query_result.get("error"))
            
            # 设置推理路径ID
            reasoning_path_id = self.current_reasoning_path.id
            self.query_agent.update_context("reasoning_path_id", reasoning_path_id)
            self.search_agent.update_context("reasoning_path_id", reasoning_path_id)
            self.graph_agent.update_context("reasoning_path_id", reasoning_path_id)
            self.reasoning_agent.update_context("reasoning_path_id", reasoning_path_id)
            self.context_agent.update_context("reasoning_path_id", reasoning_path_id)
            self.answer_agent.update_context("reasoning_path_id", reasoning_path_id)
            
            # 添加推理步骤
            if query_result.get("reasoning_step"):
                self.current_reasoning_path.add_step(query_result["reasoning_step"])
            
            # 2. 信息搜索
            print("🔍 步骤2: 信息搜索")
            search_result = await self.search_agent.process({
                "query": user_query,
                "entities": query_result.get("entities", []),
                "search_type": "hybrid"
            })
            if not search_result.get("success"):
                return self._create_error_response(user_query, "信息搜索失败", search_result.get("error"))
            
            if search_result.get("reasoning_step"):
                self.current_reasoning_path.add_step(search_result["reasoning_step"])
            
            # 3. 图数据库查询
            print("🔍 步骤3: 图数据库查询")
            graph_result = await self.graph_agent.process({
                "query_type": "entity_search",
                "entities": query_result.get("entities", [])
            })
            if not graph_result.get("success"):
                return self._create_error_response(user_query, "图数据库查询失败", graph_result.get("error"))
            
            if graph_result.get("reasoning_step"):
                self.current_reasoning_path.add_step(graph_result["reasoning_step"])
            
            # 4. 逻辑推理
            print("🔍 步骤4: 逻辑推理")
            reasoning_result = await self.reasoning_agent.process({
                "query": user_query,
                "search_results": search_result.get("results", {}),
                "graph_results": graph_result.get("results", {})
            })
            if not reasoning_result.get("success"):
                return self._create_error_response(user_query, "逻辑推理失败", reasoning_result.get("error"))
            
            if reasoning_result.get("reasoning_step"):
                self.current_reasoning_path.add_step(reasoning_result["reasoning_step"])
            
            # 5. 上下文更新
            print("🔍 步骤5: 上下文更新")
            context_result = await self.context_agent.process({
                "operation": "update",
                "context_data": {
                    "query": user_query,
                    "search_results": search_result,
                    "graph_results": graph_result,
                    "reasoning_result": reasoning_result
                }
            })
            
            if context_result.get("reasoning_step"):
                self.current_reasoning_path.add_step(context_result["reasoning_step"])
            
            # 6. 答案生成
            print("🔍 步骤6: 答案生成")
            answer_result = await self.answer_agent.process({
                "query": user_query,
                "reasoning_result": reasoning_result.get("reasoning_result", {}),
                "search_results": search_result.get("results", {})
            })
            if not answer_result.get("success"):
                return self._create_error_response(user_query, "答案生成失败", answer_result.get("error"))
            
            if answer_result.get("reasoning_step"):
                self.current_reasoning_path.add_step(answer_result["reasoning_step"])
            
            # 设置最终答案
            final_answer = answer_result.get("answer", {})
            self.current_reasoning_path.set_final_answer(
                final_answer.get("answer", ""),
                final_answer.get("confidence", 0.0)
            )
            
            # 计算处理时间
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time
            
            # 构建最终响应
            response = {
                "success": True,
                "session_id": self.session_id,
                "query": user_query,
                "answer": final_answer.get("answer", ""),
                "confidence": final_answer.get("confidence", 0.0),
                "explanation": final_answer.get("explanation", ""),
                "reasoning_process": final_answer.get("reasoning_process", ""),
                "evidence": final_answer.get("evidence", []),
                "reasoning_path": self.current_reasoning_path.to_dict(),
                "processing_time": processing_time,
                "metadata": {
                    "query_analysis": query_result.get("analysis", {}),
                    "search_results_count": search_result.get("results", {}).get("total_found", 0),
                    "graph_results_count": len(graph_result.get("results", {}).get("results", [])),
                    "reasoning_steps_count": len(reasoning_result.get("reasoning_result", {}).get("reasoning_steps", [])),
                    "total_reasoning_steps": len(self.current_reasoning_path.steps)
                }
            }
            
            print(f"✅ 查询处理完成，耗时: {processing_time:.2f}秒")
            return response
            
        except Exception as e:
            error_time = asyncio.get_event_loop().time() - start_time
            print(f"❌ 查询处理失败，耗时: {error_time:.2f}秒")
            return self._create_error_response(user_query, "系统处理失败", str(e))
    
    def _create_error_response(self, query: str, error_type: str, error_message: str) -> Dict[str, Any]:
        """创建错误响应"""
        return {
            "success": False,
            "session_id": self.session_id,
            "query": query,
            "error_type": error_type,
            "error_message": error_message,
            "answer": "抱歉，无法处理您的查询。",
            "confidence": 0.0,
            "reasoning_path": self.current_reasoning_path.to_dict() if self.current_reasoning_path else None
        }
    
    async def load_mock_data(self) -> None:
        """加载模拟数据"""
        from .models import Entity, Relation
        
        # 创建实体
        entities = [
            Entity(name="张三", type="Person", attributes={"role": "工程师"}),
            Entity(name="李四", type="Person", attributes={"role": "项目经理"}),
            Entity(name="王五", type="Person", attributes={"role": "技术总监"}),
            Entity(name="飞天项目", type="Project", attributes={"status": "进行中", "description": "重要的技术项目"}),
            Entity(name="项目A", type="Project", attributes={"status": "已完成", "description": "基础建设项目"}),
            Entity(name="项目管理", type="Skill", attributes={"level": "高级"}),
            Entity(name="Python", type="Technology", attributes={"category": "编程语言"}),
        ]
        
        # 添加到图数据库
        for entity in entities:
            await self.graph_service.add_entity(entity)
        
        # 创建关系
        relations = [
            Relation(source_id=entities[0].id, target_id=entities[1].id, relation_type="works_with"),
            Relation(source_id=entities[0].id, target_id=entities[3].id, relation_type="participates_in"),
            Relation(source_id=entities[1].id, target_id=entities[3].id, relation_type="participates_in"),
            Relation(source_id=entities[3].id, target_id=entities[2].id, relation_type="managed_by"),
            Relation(source_id=entities[2].id, target_id=entities[5].id, relation_type="has_skill"),
            Relation(source_id=entities[5].id, target_id=entities[3].id, relation_type="used_in"),
        ]
        
        # 添加关系到图数据库
        for relation in relations:
            await self.graph_service.add_relation(relation)
        
        # 生成向量嵌入
        for entity in entities:
            embeddings = await self.llm_service.generate_embeddings(entity.name)
            await self.vector_service.add_embeddings(entity.id, embeddings)
            entity.update_embeddings(embeddings)
        
        print(f"✅ 已加载 {len(entities)} 个实体和 {len(relations)} 个关系")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "initialized": self._initialized,
            "session_id": self.session_id,
            "services": {
                "graph_service": self.graph_service.health_check(),
                "vector_service": self.vector_service.health_check(),
                "llm_service": self.llm_service.health_check()
            },
            "agents": {
                "query_agent": self.query_agent.get_agent_info(),
                "search_agent": self.search_agent.get_agent_info(),
                "graph_agent": self.graph_agent.get_agent_info(),
                "reasoning_agent": self.reasoning_agent.get_agent_info(),
                "context_agent": self.context_agent.get_agent_info(),
                "answer_agent": self.answer_agent.get_agent_info()
            }
        } 