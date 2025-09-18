import re
import json
import json5
import json_repair
from typing import List, Optional, Any, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks.base import Callbacks


class JSONHandler:
    """
    Robust JSON parser with multi-tier repair strategies.
    Supports standard JSON, JSON5, json_repair, and optional LLM-based self-healing.
    """

    def __init__(self, max_retries: int = 2, self_healing: bool = False):
        self.max_retries = max_retries
        self.self_healing = self_healing

    @staticmethod
    def safe_json_parse(text: str) -> dict:
        """Try JSON → JSON5 → json_repair parsing."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        try:
            return json5.loads(text)
        except Exception:
            pass
        try:
            repaired = json_repair.repair_json(text)
            return json.loads(repaired)
        except Exception:
            return {}

    @staticmethod
    def extract_json_block(text: str) -> str:
        """Extract first JSON-like block."""
        match = re.search(r"\{[\s\S]*\}", text)
        return match.group(0) if match else text

    @staticmethod
    def extract_array_fallback(text: str) -> List[str]:
        """Fallback: extract array from text."""
        match = re.search(r"\[([\s\S]*?)\]", text)
        if not match:
            return []
        items = re.split(r",\s*", match.group(1))
        return [i.strip(" \"'") for i in items if i.strip()]

    @staticmethod
    def validate_list(items: Any) -> List[Any]:
        """Ensure clean list of strings or dicts."""
        if not isinstance(items, list):
            return []
        cleaned = []
        for i in items:
            if isinstance(i, str) and i.strip():
                cleaned.append(i.strip())
            elif isinstance(i, dict):  # 保留 dict，不要丢掉
                cleaned.append(i)
        return cleaned


    async def parse_with_fallbacks(
        self,
        raw_text: str,
        key: Optional[str] = None,
        llm: Optional[BaseLanguageModel] = None,
        callbacks: Optional[Callbacks] = None
    ) -> Union[List[str], dict]:
        """Parse with multiple fallback strategies and optional LLM repair."""
        content = re.sub(r"```(?:json)?|```", "", raw_text).strip()

        # 1. Direct parse
        data = self.safe_json_parse(content)
        if key and key in data:
            return self.validate_list(data[key])
        elif not key and data:
            return data

        # 2. Extract block
        json_block = self.extract_json_block(content)
        data = self.safe_json_parse(json_block)
        if key and key in data:
            return self.validate_list(data[key])
        elif not key and data:
            return data

        # 3. Fallback array
        if key:
            fallback_array = self.extract_array_fallback(content)
            if fallback_array:
                return self.validate_list(fallback_array)

        # 4. Self-healing
        if self.self_healing and llm is not None:
            healed = await self.heal_with_llm(raw_text, key, llm, callbacks)
            if healed:
                return healed

        return [] if key else {}

    async def heal_with_llm(
        self,
        invalid_text: str,
        key: Optional[str],
        llm: BaseLanguageModel,
        callbacks: Optional[Callbacks]
    ) -> Union[List[str], dict]:
        """Ask LLM to return valid JSON."""
        repair_prompt = f"""
Return ONLY valid JSON{f" with a key '{key}'" if key else ""}.
Invalid output was:
{invalid_text}
"""
        for _ in range(self.max_retries):
            try:
                response = await llm.ainvoke(repair_prompt, config={"callbacks": callbacks})
                repaired_text = re.sub(r"```(?:json)?|```", "", response.content).strip()
                data = self.safe_json_parse(repaired_text)
                if key and key in data:
                    return self.validate_list(data[key])
                elif not key and data:
                    return data
            except Exception:
                continue
        return [] if key else {}
