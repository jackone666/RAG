"""
异步质量监控中心 - 大模型裁判评估与坏例沉淀模块

评估指标（参考字节 §5.4「检索效果评估与调优」）：
1. Precision（精确度）：检索到的文档中真正相关的比例
2. Recall（召回率）：预估的相关文档中被检索到的比例
3. MRR（平均倒数排名）：第一个相关文档的排名倒数，越靠前越高
4. Hit Rate（命中率）：Top-K 中是否存在相关文档
5. Faithfulness（忠实度）：回答是否严格基于上下文
6. Relevance（相关性）：检索结果与问题的整体相关程度

使用原生 openai SDK 以兼容 DeepSeek。
"""
import json
import re

from loguru import logger
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.rag_params import rag_params
from src.config.settings import settings


class AsyncEvaluator:
    """异步大模型裁判评估器——检索 + 生成双维质量评估。"""

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        return self._client

    def _extract_json(self, text: str) -> dict:
        """从 LLM 返回文本中提取 JSON。"""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        logger.warning(f"无法提取 JSON: {text[:300]}")
        return {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=30), reraise=True)
    async def _judge(self, prompt: str) -> dict:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=settings.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=rag_params.judge_temperature,
            max_tokens=rag_params.judge_max_tokens,
        )
        return self._extract_json(response.choices[0].message.content or "{}")

    async def evaluate(
            self, query: str, context_nodes: list[str], answer: str, tenant_id: str
    ) -> dict | None:
        """执行完整评估：检索指标 + 生成指标。

        Returns:
            包含所有指标的 dict，或 None（评估失败时）
        """
        # 跳过空回答或降级回答
        skip_phrases = ["知识服务暂时不可用", "未找到相关文档"]
        if not answer.strip() or any(p in answer for p in skip_phrases):
            return None

        import asyncio
        try:
            retrieval_scores, generation_scores = await asyncio.gather(
                self._evaluate_retrieval(query, context_nodes),
                self._evaluate_generation(query, context_nodes, answer),
            )
        except Exception as e:
            logger.error(f"裁判模型评估失败: {e}")
            return None

        result = {**retrieval_scores, **generation_scores}
        # overall = 加权平均（检索 0.4 + 生成 0.6），仅取浮点字段
        gen_nums = [v for v in generation_scores.values() if isinstance(v, (int, float))]
        ret_nums = [v for v in retrieval_scores.values() if isinstance(v, (int, float))]
        gen_avg = sum(gen_nums) / len(gen_nums) if gen_nums else 0
        ret_avg = sum(ret_nums) / len(ret_nums) if ret_nums else 0
        result["overall"] = round(ret_avg * 0.4 + gen_avg * 0.6, 4)
        result["passing"] = result["overall"] >= rag_params.eval_score_threshold

        if not result["passing"]:
            bad_case = {
                "tenant_id": tenant_id, "query": query,
                "context_nodes": context_nodes, "answer": answer, **result,
            }
            try:
                from src.storage.pg_store import insert_bad_case
                insert_bad_case(bad_case)
            except Exception as e:
                logger.warning(f"坏例写入失败: {e}")
            logger.warning(
                f"检测到坏例: tenant={tenant_id}, "
                f"precision={result.get('precision', 0):.2f}, "
                f"recall={result.get('recall', 0):.2f}, "
                f"hit_rate={result.get('hit_rate', 0):.2f}, "
                f"overall={result['overall']:.2f}"
            )

        return result

    async def _evaluate_retrieval(self, query: str, docs: list[str]) -> dict:
        """评估检索质量：让裁判判断每个文档是否相关。"""
        if not docs:
            return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "hit_rate": 0.0}

        docs_text = "\n\n".join(
            f"[文档 {i}]\n{d[:1000]}" for i, d in enumerate(docs)
        )
        prompt = f"""评估以下检索结果的质量。对于每个文档，判断它是否与查询相关。

## 查询
{query}

## 检索到的文档（共 {len(docs)} 个）
{docs_text}

## 要求
对每个文档判断相关性（relevant: true/false），然后计算：
1. **precision**: 相关文档数 / 总检索文档数（0.0~1.0）
2. **recall**: 预估检索到的相关文档占比（0.0~1.0，1.0=所有可能的相关信息都在检索结果中）
3. **mrr**: 1 / (第一个相关文档的排名)，排名从1开始，无相关则为0
4. **hit_rate**: Top-10 中是否有相关文档（1.0=有，0.0=无）

严格输出 JSON：
```json
{{
  "relevant_docs": [0, 3, 5],
  "precision": 0.0,
  "recall": 0.0,
  "mrr": 0.0,
  "hit_rate": 0.0
}}
```"""

        result = await self._judge(prompt)
        for key in ["precision", "recall", "mrr", "hit_rate"]:
            if key not in result:
                result[key] = 0.0
            try:
                result[key] = float(result[key])
            except (TypeError, ValueError):
                result[key] = 0.0
        return {
            "precision": min(result["precision"], 1.0),
            "recall": min(result["recall"], 1.0),
            "mrr": min(result["mrr"], 1.0),
            "hit_rate": min(result["hit_rate"], 1.0),
        }

    async def _evaluate_generation(self, query: str, context: list[str], answer: str) -> dict:
        """评估生成质量：忠实度 + 相关性。"""
        ctx_text = "\n\n".join(c[:1000] for c in context[:5])
        prompt = f"""评估以下 RAG 回答的质量：

## 上下文
{ctx_text}

## 问题
{query}

## 回答
{answer}

## 评估
1. **faithfulness（忠实度）**：回答是否严格基于上下文？无编造=1.0，完全编造=0.0
2. **relevance（相关性）**：回答与问题的相关程度？高度相关=1.0，完全无关=0.0

严格输出 JSON：
```json
{{
  "faithfulness": 0.0,
  "faithfulness_reason": "理由",
  "relevance": 0.0,
  "relevance_reason": "理由"
}}
```"""

        result = await self._judge(prompt)
        for key in ["faithfulness", "relevance"]:
            if key not in result:
                result[key] = 0.0
            try:
                result[key] = float(result[key])
            except (TypeError, ValueError):
                result[key] = 0.0
        return {
            "faithfulness": min(result["faithfulness"], 1.0),
            "faithfulness_reason": result.get("faithfulness_reason", ""),
            "relevance": min(result["relevance"], 1.0),
            "relevance_reason": result.get("relevance_reason", ""),
        }


evaluator = AsyncEvaluator()
