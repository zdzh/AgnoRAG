"""
答案生成Agent

负责生成最终答案和解释
"""

import asyncio
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent


class AnswerAgent(BaseAgent):
    """答案生成Agent"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("AnswerAgent", config)
        self.max_answer_length = self.get_config("max_answer_length", 500)
        self.include_reasoning = self.get_config("include_reasoning", True)
        self.include_confidence = self.get_config("include_confidence", True)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理答案生成请求"""
        try:
            # 验证输入
            if not await self.validate_input(input_data):
                raise ValueError("输入数据验证失败")
            
            # 预处理
            processed_input = await self.preprocess(input_data)
            
            # 获取输入参数
            query = processed_input.get("query", "")
            reasoning_result = processed_input.get("reasoning_result", {})
            search_results = processed_input.get("search_results", {})
            
            # 生成答案
            answer_result = await self._generate_answer(query, reasoning_result, search_results)
            
            # 创建推理步骤
            reasoning_step = await self.create_reasoning_step(
                step_type="answer_generation",
                input_data=str(processed_input),
                output_data=str(answer_result),
                confidence=answer_result.get("confidence", 0.8)
            )
            
            # 构建输出
            output = {
                "query": query,
                "answer": answer_result,
                "reasoning_step": reasoning_step,
                "success": True
            }
            
            # 更新上下文
            self.update_context("final_answer", output)
            
            # 后处理
            return await self.postprocess(output)
            
        except Exception as e:
            await self.handle_error(e)
            return {
                "success": False,
                "error": str(e),
                "query": input_data.get("query", "")
            }
    
    async def _generate_answer(self, query: str, reasoning_result: Dict[str, Any], search_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成答案"""
        answer_result = {
            "answer": "",
            "confidence": 0.0,
            "explanation": "",
            "evidence": [],
            "reasoning_process": "",
            "metadata": {}
        }
        
        # 从推理结果中提取答案
        if reasoning_result.get("answer"):
            answer_result["answer"] = reasoning_result["answer"]
            answer_result["confidence"] = reasoning_result.get("confidence", 0.0)
            answer_result["explanation"] = reasoning_result.get("conclusion", "")
        
        # 生成详细解释
        detailed_explanation = await self._generate_explanation(query, reasoning_result, search_results)
        answer_result["explanation"] = detailed_explanation
        
        # 收集证据
        evidence = await self._collect_evidence(reasoning_result, search_results)
        answer_result["evidence"] = evidence
        
        # 生成推理过程
        reasoning_process = await self._format_reasoning_process(reasoning_result)
        answer_result["reasoning_process"] = reasoning_process
        
        # 添加元数据
        answer_result["metadata"] = {
            "query_type": "multi_hop" if "参与" in query and "项目" in query else "single_hop",
            "evidence_count": len(evidence),
            "reasoning_steps_count": len(reasoning_result.get("reasoning_steps", [])),
            "generation_time": asyncio.get_event_loop().time()
        }
        
        return answer_result
    
    async def _generate_explanation(self, query: str, reasoning_result: Dict[str, Any], search_results: Dict[str, Any]) -> str:
        """生成详细解释"""
        if not self.llm_service:
            return reasoning_result.get("conclusion", "")
        
        # 构建解释提示
        prompt = self._build_explanation_prompt(query, reasoning_result, search_results)
        
        # 使用LLM生成解释
        explanation = await self.llm_service.generate_text(prompt, max_tokens=300)
        
        return explanation
    
    async def _collect_evidence(self, reasoning_result: Dict[str, Any], search_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """收集证据"""
        evidence = []
        
        # 从推理结果中收集证据
        if reasoning_result.get("evidence"):
            evidence.extend(reasoning_result["evidence"])
        
        # 从搜索结果中收集证据
        if search_results.get("results", {}).get("combined_results"):
            for result in search_results["results"]["combined_results"][:3]:  # 只取前3个
                evidence.append({
                    "type": "search_evidence",
                    "content": result.get("entity", {}).get("name", "Unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "source": result.get("source", "search")
                })
        
        return evidence
    
    async def _format_reasoning_process(self, reasoning_result: Dict[str, Any]) -> str:
        """格式化推理过程"""
        steps = reasoning_result.get("reasoning_steps", [])
        
        if not steps:
            return "推理过程：基于搜索结果进行逻辑推理"
        
        process_lines = ["推理过程："]
        for i, step in enumerate(steps, 1):
            content = step.get("content", "")
            if content:
                process_lines.append(f"{i}. {content}")
        
        return "\n".join(process_lines)
    
    def _build_explanation_prompt(self, query: str, reasoning_result: Dict[str, Any], search_results: Dict[str, Any]) -> str:
        """构建解释提示"""
        evidence_text = ""
        if reasoning_result.get("evidence"):
            evidence_text = "\n".join([
                f"- {e.get('content', {}).get('entity', {}).get('name', 'Unknown')}"
                for e in reasoning_result["evidence"][:3]
            ])
        
        prompt = f"""
基于以下信息为查询提供详细解释：

查询: {query}

推理结果: {reasoning_result.get("answer", "")}

证据:
{evidence_text}

请提供一个清晰、详细的解释，说明如何得出这个结论。
"""
        return prompt
    
    def format_final_answer(self, answer_result: Dict[str, Any]) -> str:
        """格式化最终答案"""
        answer = answer_result.get("answer", "")
        confidence = answer_result.get("confidence", 0.0)
        explanation = answer_result.get("explanation", "")
        
        formatted_answer = f"答案: {answer}\n"
        
        if self.include_confidence:
            formatted_answer += f"置信度: {confidence:.2f}\n"
        
        if self.include_reasoning and explanation:
            formatted_answer += f"\n解释: {explanation}"
        
        return formatted_answer
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """验证输入数据"""
        query = input_data.get("query", "")
        reasoning_result = input_data.get("reasoning_result", {})
        search_results = input_data.get("search_results", {})
        
        return (
            isinstance(query, str) and len(query.strip()) > 0 and
            isinstance(reasoning_result, dict) and
            isinstance(search_results, dict)
        )
    
    async def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理输入数据"""
        return {
            "query": input_data.get("query", "").strip(),
            "reasoning_result": input_data.get("reasoning_result", {}),
            "search_results": input_data.get("search_results", {})
        }
    
    async def postprocess(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理输出数据"""
        output_data["metadata"] = {
            "agent": self.name,
            "processing_time": asyncio.get_event_loop().time(),
            "answer_length": len(output_data.get("answer", {}).get("answer", "")),
            "confidence": output_data.get("answer", {}).get("confidence", 0.0),
            "evidence_count": len(output_data.get("answer", {}).get("evidence", []))
        }
        return output_data 