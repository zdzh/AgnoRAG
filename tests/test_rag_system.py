"""
RAG系统测试

测试Agentic RAG系统的核心功能
"""

import asyncio
import os
import sys
from typing import Any, Dict

import pytest

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_system import AgenticRAGSystem


class TestAgenticRAGSystem:
    """RAG系统测试类"""
    
    @pytest.fixture
    async def rag_system(self):
        """创建RAG系统实例"""
        config = {
            "graph_service": {
                "storage_file": "test_data/knowledge_graph/test_graph.json"
            },
            "vector_service": {
                "storage_file": "test_data/vector_store/test_embeddings.pkl",
                "dimension": 768
            },
            "llm_service": {
                "model_name": "mock-llm",
                "embedding_dimension": 768,
                "response_delay": 0.01  # 测试时使用更快的响应
            },
            "query_agent": {
                "max_entities": 5,
                "confidence_threshold": 0.7
            },
            "search_agent": {
                "max_results": 10,
                "similarity_threshold": 0.5
            },
            "graph_agent": {
                "max_path_length": 3,
                "max_neighbors": 5
            },
            "reasoning_agent": {
                "max_reasoning_steps": 3,
                "confidence_threshold": 0.6
            },
            "context_agent": {
                "max_context_size": 100,
                "session_timeout": 3600
            },
            "answer_agent": {
                "max_answer_length": 200,
                "include_reasoning": True,
                "include_confidence": True
            }
        }
        
        system = AgenticRAGSystem(config)
        await system.initialize()
        await system.load_mock_data()
        
        yield system
        
        await system.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, rag_system):
        """测试系统初始化"""
        assert rag_system._initialized == True
        assert rag_system.session_id is not None
        
        # 检查服务状态
        status = rag_system.get_system_status()
        assert status["initialized"] == True
        assert status["services"]["graph_service"]["status"] == "healthy"
        assert status["services"]["vector_service"]["status"] == "healthy"
        assert status["services"]["llm_service"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_mock_data_loading(self, rag_system):
        """测试模拟数据加载"""
        status = rag_system.get_system_status()
        
        # 检查图数据库
        graph_status = status["services"]["graph_service"]
        assert graph_status["entity_count"] > 0
        assert graph_status["relation_count"] > 0
        
        # 检查向量存储
        vector_status = status["services"]["vector_service"]
        assert vector_status["embedding_count"] > 0
    
    @pytest.mark.asyncio
    async def test_single_hop_query(self, rag_system):
        """测试单跳查询"""
        query = "张三参与了哪个项目？"
        result = await rag_system.query(query)
        
        assert result["success"] == True
        assert "answer" in result
        assert "confidence" in result
        assert "processing_time" in result
        assert result["confidence"] > 0.0
        
        # 检查推理路径
        reasoning_path = result.get("reasoning_path")
        assert reasoning_path is not None
        assert len(reasoning_path.get("steps", [])) > 0
    
    @pytest.mark.asyncio
    async def test_multi_hop_query(self, rag_system):
        """测试多跳查询"""
        query = "张三参与了哪个项目？"
        result = await rag_system.query(query)
        
        assert result["success"] == True
        assert "飞天项目" in result.get("answer", "")
        
        # 检查元数据
        metadata = result.get("metadata", {})
        assert metadata.get("total_reasoning_steps", 0) > 0
    
    @pytest.mark.asyncio
    async def test_entity_search_query(self, rag_system):
        """测试实体搜索查询"""
        query = "李四和谁一起工作？"
        result = await rag_system.query(query)
        
        assert result["success"] == True
        assert len(result.get("answer", "")) > 0
    
    @pytest.mark.asyncio
    async def test_project_query(self, rag_system):
        """测试项目相关查询"""
        query = "飞天项目的负责人是谁？"
        result = await rag_system.query(query)
        
        assert result["success"] == True
        assert "王五" in result.get("answer", "")
    
    @pytest.mark.asyncio
    async def test_skill_query(self, rag_system):
        """测试技能相关查询"""
        query = "王五有什么技能？"
        result = await rag_system.query(query)
        
        assert result["success"] == True
        assert len(result.get("answer", "")) > 0
    
    @pytest.mark.asyncio
    async def test_empty_query(self, rag_system):
        """测试空查询"""
        query = ""
        result = await rag_system.query(query)
        
        assert result["success"] == False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_unknown_query(self, rag_system):
        """测试未知查询"""
        query = "未知实体参与了什么项目？"
        result = await rag_system.query(query)
        
        # 即使没有找到相关信息，系统也应该返回一个合理的响应
        assert "answer" in result
        assert len(result.get("answer", "")) > 0
    
    @pytest.mark.asyncio
    async def test_reasoning_path_completeness(self, rag_system):
        """测试推理路径完整性"""
        query = "张三参与了哪个项目？"
        result = await rag_system.query(query)
        
        reasoning_path = result.get("reasoning_path")
        assert reasoning_path is not None
        
        # 检查推理路径是否完整
        assert reasoning_path.get("query") == query
        assert len(reasoning_path.get("steps", [])) > 0
        assert reasoning_path.get("final_answer") != ""
        assert reasoning_path.get("confidence") > 0.0
    
    @pytest.mark.asyncio
    async def test_processing_time(self, rag_system):
        """测试处理时间"""
        query = "张三参与了哪个项目？"
        result = await rag_system.query(query)
        
        processing_time = result.get("processing_time", 0)
        assert processing_time > 0
        assert processing_time < 10.0  # 应该在合理时间内完成
    
    @pytest.mark.asyncio
    async def test_confidence_scores(self, rag_system):
        """测试置信度分数"""
        query = "张三参与了哪个项目？"
        result = await rag_system.query(query)
        
        confidence = result.get("confidence", 0)
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_agent_collaboration(self, rag_system):
        """测试Agent协作"""
        query = "张三参与了哪个项目？"
        result = await rag_system.query(query)
        
        # 检查各个Agent的贡献
        metadata = result.get("metadata", {})
        assert metadata.get("search_results_count", 0) >= 0
        assert metadata.get("graph_results_count", 0) >= 0
        assert metadata.get("reasoning_steps_count", 0) >= 0
        assert metadata.get("total_reasoning_steps", 0) > 0


async def run_tests():
    """运行测试"""
    print("🧪 开始运行RAG系统测试...")
    
    # 创建测试实例
    config = {
        "graph_service": {
            "storage_file": "test_data/knowledge_graph/test_graph.json"
        },
        "vector_service": {
            "storage_file": "test_data/vector_store/test_embeddings.pkl",
            "dimension": 768
        },
        "llm_service": {
            "model_name": "mock-llm",
            "embedding_dimension": 768,
            "response_delay": 0.01
        }
    }
    
    rag_system = AgenticRAGSystem(config)
    
    try:
        await rag_system.initialize()
        await rag_system.load_mock_data()
        
        # 运行测试用例
        test_cases = [
            ("张三参与了哪个项目？", "飞天项目"),
            ("李四和谁一起工作？", "张三"),
            ("飞天项目的负责人是谁？", "王五"),
            ("王五有什么技能？", "项目管理")
        ]
        
        passed = 0
        total = len(test_cases)
        
        for query, expected_keyword in test_cases:
            print(f"\n🔍 测试查询: {query}")
            result = await rag_system.query(query)
            
            if result.get("success"):
                answer = result.get("answer", "")
                if expected_keyword in answer:
                    print(f"✅ 通过: 找到关键词 '{expected_keyword}'")
                    passed += 1
                else:
                    print(f"❌ 失败: 未找到关键词 '{expected_keyword}'")
                    print(f"   实际答案: {answer}")
            else:
                print(f"❌ 失败: 查询处理失败")
                print(f"   错误: {result.get('error_message', '未知错误')}")
        
        print(f"\n📊 测试结果: {passed}/{total} 通过")
        
        if passed == total:
            print("🎉 所有测试通过！")
        else:
            print("⚠️  部分测试失败")
    
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
    
    finally:
        await rag_system.shutdown()


if __name__ == "__main__":
    asyncio.run(run_tests()) 