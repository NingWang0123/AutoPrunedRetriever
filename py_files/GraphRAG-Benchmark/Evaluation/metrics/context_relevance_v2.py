from typing import List, Callable, Awaitable
import numpy as np
import re
from typing import List, Dict, Any, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks
import json
import re
from Evaluation.metrics.utils import JSONHandler

CONTEXT_RELEVANCE_JUDGE_BY_EVIDENCE_PROMPT = """
### Task
You are given a Context, a Question, and a list of Evidence. Your task is to evaluate how relevant the Context is to the Question and Evidence.

Score based on the following criteria:
- **2 (Highly Relevant)**: The Context directly answers the Question or is essential for understanding the Evidence.
- **1 (Partially Relevant)**: The Context provides related background information but does not directly answer the Question, or is only tangentially related.
- **0 (Not Relevant)**: The Context is about a completely different topic.

Respond ONLY with a JSON object containing:
- "reason": A brief explanation (1 sentence) for your score.
- "relevance_score": Your relevance score (0, 1, or 2).

### Example
Input:
Context: "The capital of Australia is Canberra, a planned city located between Sydney and Melbourne."
Evidence: ["Canberra is the capital of Australia"]
Question: "What is the capital of Australia?"

Output:
{{
  "reason": "The context directly confirms that Canberra is the capital of Australia, which is the core of the question and evidence.",
  "score": 2
}}

### Actual Input
Context: "{context}"

Evidence: {evidence}

Question: "{question}"

### Your Response:
"""
 
async def compute_context_relevance(
    question: str,
    contexts: List[str],
    evidence: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 3
) -> float:
    """
    Evaluate relevance of contexts to question/evidence with chunking for long texts.
    Returns average score between 0.0 (irrelevant) and 1.0 (fully relevant).
    """
    # Handle edge cases
    if not question.strip() or not contexts or not any(c.strip() for c in contexts):
        return 0.0

    all_scores = []
    
    for context in contexts:
        if not context.strip():
            continue
            
        # Check for exact match with question
        if context.strip() == question.strip():
            all_scores.append(0.0)
            continue
            
        # Process long contexts with chunking
        chunks = _chunk_context(context)
        
        # Evaluate each chunk
        chunk_scores = []
        for chunk in chunks:
            result = await _evaluate_single_context(
                question, evidence, chunk, llm, callbacks, max_retries
            )
            if "relevance_score" in result:
                # Convert 0-2 scale to 0-1 scale
                chunk_score = result["relevance_score"] / 2.0
                chunk_scores.append(chunk_score)
        
        # Calculate average for this context
        if chunk_scores:
            context_avg = sum(chunk_scores) / len(chunk_scores)
            all_scores.append(context_avg)
    
    # Calculate final average across all contexts
    return sum(all_scores) / len(all_scores) if all_scores else 0.0

def _chunk_context(context: str, max_length: int = 21000, chunk_size: int = 3000) -> List[str]:
    """
    Split long contexts into manageable chunks:
    - If > max_length: truncate then chunk
    - If > chunk_size: split into chunks
    - Otherwise: return as single chunk
    """
    # Truncate extremely long contexts first
    processed = context[:max_length] if len(context) > max_length else context
    
    # Split into chunks if needed
    if len(processed) > 5000:
        chunks = []
        for i in range(0, len(processed), chunk_size):
            chunk_end = min(i + chunk_size, len(processed))
            chunks.append(processed[i:chunk_end])
        return chunks
    return [processed]

async def _evaluate_single_context(
    question: str,
    evidence: List[str],
    context: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> Dict[str, Union[int, str]]:
    """
    Evaluate a single context chunk and return relevance score with reason.
    Uses RobustJSONHandler for resilient JSON parsing.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    evidence_str = json.dumps(evidence)
    truncated_context = context[:20000]

    prompt = CONTEXT_RELEVANCE_JUDGE_BY_EVIDENCE_PROMPT.format(
        context=truncated_context,
        evidence=evidence_str,
        question=question
    )

    for _ in range(max_retries):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )

            return _normalize_relevance_response(parsed)
        except Exception:
            continue

    return {"reason": "Evaluation failed after retries", "relevance_score": 0}


def _normalize_relevance_response(parsed: Union[Dict, List, None]) -> Dict[str, Union[int, str]]:
    """
    Normalize parsed JSON (or text) to ensure relevance_score and reason are returned.
    """
    if isinstance(parsed, dict):
        score = parsed.get("relevance_score", parsed.get("score"))
        reason = parsed.get("reason", "")
        if score is not None:
            try:
                return {
                    "reason": str(reason),
                    "relevance_score": int(score)
                }
            except (TypeError, ValueError):
                pass

    # Fallback: try to extract a 0-2 score from text
    if isinstance(parsed, str):
        match = re.search(r"\b[0-2]\b", parsed)
        if match:
            return {
                "reason": "Score extracted from text",
                "relevance_score": int(match.group())
            }

    return {"reason": "Failed to parse response", "relevance_score": 0}