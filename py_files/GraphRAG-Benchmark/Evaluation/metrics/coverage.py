import json
import re
import numpy as np
from typing import List, Dict, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks
from Evaluation.metrics.utils import JSONHandler

FACT_EXTRACTION_PROMPT = """
You are given a question and a reference answer. Break down the reference answer into a list of distinct factual statements (facts) that could be independently verified. 
Output them as a JSON list of strings under the 'facts' field.

Example
Input:
  Question: "What causes seasons?"
  Reference: "Seasonal changes result from Earth's axial tilt. This tilt causes different hemispheres to receive varying sunlight."

Output:
{{
  "facts": [
    "Seasonal changes result from Earth's axial tilt",
    "The axial tilt causes different hemispheres to receive varying sunlight"
  ]
}}

### Actual Input
Question: "{question}"
Reference Answer: "{reference}"

### Your Response:
"""

FACT_COVERAGE_PROMPT = """
### Task
For each factual statement from the reference, determine if it's covered in the response.
Respond ONLY with a JSON object containing a "classifications" list. Each item should have:
- "statement": the exact fact from reference
- "attributed": 1 if covered, 0 if not

### Example
Response: "Seasons are caused by Earth's tilted axis"
Reference Facts: [
  "Seasonal changes result from Earth's axial tilt",
  "The axial tilt causes different hemispheres to receive varying sunlight"
]

Output:
{{
  "classifications": [
    {{"statement": "Seasonal changes result from Earth's axial tilt", "attributed": 1}},
    {{"statement": "The axial tilt causes different hemispheres to receive varying sunlight", "attributed": 0}}
  ]
}}

### Actual Input
Question: "{question}"
Response: "{response}"
Reference Facts: {facts}

### Your Response:
"""

async def compute_coverage_score(
    question: str,
    reference: str,
    response: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Calculate coverage score (0.0-1.0) by measuring what percentage of 
    reference facts are covered in the response.
    """
    # Handle edge cases
    if not reference.strip():
        return 1.0  # Perfect coverage for empty reference
    
    # Step 1: Extract facts from reference
    facts = await _extract_facts(
        question, reference, llm, callbacks, max_retries
    )
    
    if not facts:
        return np.nan  # Failed to extract facts
    
    # Step 2: Check fact coverage in response
    coverage = await _check_fact_coverage(
        question, facts, response, llm, callbacks, max_retries
    )
    
    # Calculate coverage score
    if coverage:
        attributed = [c["attributed"] for c in coverage]
        return sum(attributed) / len(attributed)
    return np.nan

async def _extract_facts(
    question: str,
    reference: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[str]:
    """
    Extract factual statements from the reference answer using an LLM.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    prompt = FACT_EXTRACTION_PROMPT.format(
        question=question,
        reference=reference[:3000]  # Avoid overly long prompts
    )

    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            return _validate_facts(
                parsed.get("facts", []) if isinstance(parsed, dict) else parsed
            )
        except Exception:
            continue

    return []


def _validate_facts(facts: List) -> List[str]:
    """Ensure extracted facts are valid non-empty strings."""
    if not isinstance(facts, list):
        return []
    return [str(f).strip() for f in facts if isinstance(f, (str, int, float)) and str(f).strip()]


async def _check_fact_coverage(
    question: str,
    facts: List[str],
    response_text: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[Dict]:
    """
    Check which facts are covered in the given response using an LLM.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    prompt = FACT_COVERAGE_PROMPT.format(
        question=question,
        response=response_text[:3000],
        facts=json.dumps(facts)
    )

    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            return _validate_classifications(
                parsed.get("classifications", []) if isinstance(parsed, dict) else parsed
            )
        except Exception:
            continue

    return []


def _validate_classifications(classifications: List) -> List[Dict]:
    """Ensure each classification entry contains required fields with proper types."""
    if not isinstance(classifications, list):
        return []
    valid = []
    for item in classifications:
        if not isinstance(item, dict):
            continue
        try:
            if "statement" in item and "attributed" in item and item["attributed"] in {0, 1}:
                valid.append({
                    "statement": str(item["statement"]),
                    "attributed": int(item["attributed"])
                })
        except (TypeError, ValueError):
            continue
    return valid