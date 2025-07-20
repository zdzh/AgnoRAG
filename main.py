"""
Agentic RAG 系统主程序

多跳问答系统的演示程序
"""

import asyncio
import json
from typing import Any, Dict

from src.rag_system import AgenticRAGSystem


async def main():
    """主程序"""
    print("🚀 启动 Agentic RAG 多跳问答系统")
    print("=" * 50)
    
    # 系统配置
    config = {
        "graph_service": {
            "storage_file": "data/knowledge_graph/graph_data.json"
        },
        "vector_service": {
            "storage_file": "data/vector_store/embeddings.pkl",
            "dimension": 768
        },
        "llm_service": {
            "model_name": "mock-llm",
            "embedding_dimension": 768,
            "response_delay": 0.1
        },
        "query_agent": {
            "max_entities": 10,
            "confidence_threshold": 0.7
        },
        "search_agent": {
            "max_results": 20,
            "similarity_threshold": 0.5
        },
        "graph_agent": {
            "max_path_length": 5,
            "max_neighbors": 10
        },
        "reasoning_agent": {
            "max_reasoning_steps": 5,
            "confidence_threshold": 0.6
        },
        "context_agent": {
            "max_context_size": 1000,
            "session_timeout": 3600
        },
        "answer_agent": {
            "max_answer_length": 500,
            "include_reasoning": True,
            "include_confidence": True
        }
    }
    
    # 创建RAG系统
    rag_system = AgenticRAGSystem(config)
    
    try:
        # 初始化系统
        print("📋 初始化系统...")
        await rag_system.initialize()
        
        # 加载模拟数据
        print("📊 加载模拟数据...")
        await rag_system.load_mock_data()
        
        # 显示系统状态
        print("\n📈 系统状态:")
        status = rag_system.get_system_status()
        print(f"  - 初始化状态: {'✅' if status['initialized'] else '❌'}")
        print(f"  - 会话ID: {status['session_id']}")
        print(f"  - 图数据库实体数: {status['services']['graph_service']['entity_count']}")
        print(f"  - 向量存储数量: {status['services']['vector_service']['embedding_count']}")
        
        # 演示查询
        print("\n🎯 开始演示查询...")
        print("=" * 50)
        
        # 测试查询列表
        test_queries = [
            "张三参与了哪个项目？",
            "李四和谁一起工作？",
            "飞天项目的负责人是谁？",
            "王五有什么技能？"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 查询 {i}: {query}")
            print("-" * 30)
            
            # 执行查询
            result = await rag_system.query(query)
            
            # 显示结果
            if result.get("success"):
                print(f"✅ 答案: {result.get('answer', '')}")
                print(f"📊 置信度: {result.get('confidence', 0):.2f}")
                print(f"⏱️  处理时间: {result.get('processing_time', 0):.2f}秒")
                
                if result.get("explanation"):
                    print(f"💡 解释: {result.get('explanation')}")
                
                if result.get("reasoning_process"):
                    print(f"🧠 推理过程:\n{result.get('reasoning_process')}")
                
                # 显示元数据
                metadata = result.get("metadata", {})
                print(f"📈 统计信息:")
                print(f"  - 搜索到 {metadata.get('search_results_count', 0)} 个结果")
                print(f"  - 图数据库返回 {metadata.get('graph_results_count', 0)} 个结果")
                print(f"  - 推理步骤数: {metadata.get('reasoning_steps_count', 0)}")
                print(f"  - 总推理步骤: {metadata.get('total_reasoning_steps', 0)}")
            else:
                print(f"❌ 查询失败: {result.get('error_message', '未知错误')}")
            
            print("-" * 30)
        
        # 交互式查询
        print("\n🎮 进入交互模式 (输入 'quit' 退出)")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n请输入您的查询: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not user_input:
                    print("请输入有效的查询")
                    continue
                
                print(f"\n🔍 处理查询: {user_input}")
                result = await rag_system.query(user_input)
                
                if result.get("success"):
                    print(f"\n✅ 答案: {result.get('answer', '')}")
                    print(f"📊 置信度: {result.get('confidence', 0):.2f}")
                    print(f"⏱️  处理时间: {result.get('processing_time', 0):.2f}秒")
                    
                    if result.get("explanation"):
                        print(f"💡 解释: {result.get('explanation')}")
                    
                    if result.get("reasoning_process"):
                        print(f"🧠 推理过程:\n{result.get('reasoning_process')}")
                else:
                    print(f"❌ 查询失败: {result.get('error_message', '未知错误')}")
                
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，正在退出...")
                break
            except Exception as e:
                print(f"❌ 处理查询时发生错误: {e}")
    
    except Exception as e:
        print(f"❌ 系统运行错误: {e}")
    
    finally:
        # 关闭系统
        print("\n🔄 正在关闭系统...")
        await rag_system.shutdown()
        print("👋 系统已关闭，再见！")


if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main()) 