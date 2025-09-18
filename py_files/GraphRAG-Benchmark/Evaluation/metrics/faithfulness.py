import json
import numpy as np
from typing import List, Dict, Optional, Union
import re
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import Callbacks
from Evaluation.metrics.utils import JSONHandler

STATEMENT_GENERATOR_PROMPT = """
Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Format the outputs in JSON.

Example Input: 
Question: Who was Albert Einstein and what is he best known for?
Answer: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.

Example Output:
["Albert Einstein was a German-born theoretical physicist.", "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.","Albert Einstein was best known for developing the theory of relativity.","Albert Einstein also made important contributions to the development of the theory of quantum mechanics."]

Input Text:
Question:{question}
Answer: {answer}

Generated Statements:
"""

FAITHFULNESS_EXAMPLES = [
    {
        "input": {
            "context": "John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.",
            "statements": [
                "John is majoring in Biology.",
                "John is taking a course on Artificial Intelligence.",
                "John is a dedicated student.",
                "John has a part-time job.",
            ],
        },
        "output": [
            {"statement": "John is majoring in Biology.", "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.", "verdict": 0},
            {"statement": "John is taking a course on Artificial Intelligence.", "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.", "verdict": 0},
            {"statement": "John is a dedicated student.", "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.", "verdict": 1},
            {"statement": "John has a part-time job.", "reason": "There is no information given in the context about John having a part-time job.", "verdict": 0},
        ],
    },
    {
        "input": {
            "context": "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
            "statements": [
                "Albert Einstein was a genius.",
            ],
        },
        "output": [
            {"statement": "Albert Einstein was a genius.", "reason": "The context and statement are unrelated", "verdict": 0},
        ],
    },
]

FAITHFULNESS_EVALUATION_PROMPT = """
Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.

Examples:
{examples}

Current Analysis:
Context: {context}
Statements: {statements}
"""

async def compute_faithfulness_score(
    question: str,
    answer: str,
    contexts: List[str],
    llm: BaseLanguageModel,
    callbacks: Callbacks = None,
    max_retries: int = 2
) -> float:
    """
    Calculate faithfulness score (0.0-1.0) by measuring what percentage of 
    answer statements are supported by the context.
    """
    # Step 1: Generate atomic statements from answer
    statements = await _generate_statements(
        question, answer, llm, callbacks, max_retries
    )
    
    # Handle edge cases
    if not statements:
        return 1.0 if not answer.strip() else np.nan
    
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0  # No context means no support
    
    # Step 2: Evaluate statement faithfulness
    verdicts = await _evaluate_statements(
        statements, context_str, llm, callbacks, max_retries
    )
    
    # Calculate faithfulness score
    if verdicts:
        supported = [v["verdict"] for v in verdicts]
        return sum(supported) / len(supported)
    return np.nan

async def _generate_statements(
    question: str,
    answer: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[str]:
    """
    Break down the answer into atomic statements using LLM.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    prompt = STATEMENT_GENERATOR_PROMPT.format(
        question=question[:500],
        answer=answer[:3000]
    )

    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            if isinstance(parsed, list):
                return [str(s).strip() for s in parsed if str(s).strip()]
            return []
        except Exception:
            continue

    return []


async def _evaluate_statements(
    statements: List[str],
    context: str,
    llm: BaseLanguageModel,
    callbacks: Callbacks,
    max_retries: int,
    self_healing: bool = False
) -> List[Dict]:
    """
    Evaluate which statements are supported by the context using LLM.
    """
    parser = JSONHandler(max_retries=max_retries, self_healing=self_healing)

    # Prepare examples for prompt
    examples = "\n".join(
        f"Input: {json.dumps(ex['input'])}\nOutput: {json.dumps(ex['output'])}"
        for ex in FAITHFULNESS_EXAMPLES
    )

    prompt = FAITHFULNESS_EVALUATION_PROMPT.format(
        examples=examples,
        context=context[:10000],
        statements=json.dumps(statements)[:5000]
    )

    for _ in range(max_retries + 1):
        try:
            response = await llm.ainvoke(prompt, config={"callbacks": callbacks})
            parsed = await parser.parse_with_fallbacks(
                response.content,
                llm=llm if self_healing else None,
                callbacks=callbacks
            )
            return _validate_verdicts(
                parsed if isinstance(parsed, list) else [parsed]
            )        
        except Exception:
            continue

    return []


def _validate_verdicts(verdicts: Union[List, Dict]) -> List[Dict]:
    """
    Ensure verdicts have required fields and correct types.
    Accepts either a list of dicts or a single dict.
    """
    if isinstance(verdicts, dict):
        verdicts = [verdicts]
    elif not isinstance(verdicts, list):
        return []

    valid = []
    for item in verdicts:
        if not isinstance(item, dict):
            continue
        try:
            if (
                "statement" in item
                and "verdict" in item and item["verdict"] in {0, 1}
                and "reason" in item
            ):
                valid.append({
                    "statement": str(item["statement"]),
                    "verdict": int(item["verdict"]),
                    "reason": str(item["reason"])
                })
        except (TypeError, ValueError):
            continue
    return valid
