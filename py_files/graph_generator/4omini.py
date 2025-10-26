import openai
import os
import json
import time
from functools import lru_cache
from typing import Optional, Union, Set, Tuple
import asyncio
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

# OpenAI client setup
def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY or LLM_API_KEY environment variable")
    return openai.OpenAI(api_key=api_key)

@lru_cache(maxsize=2)
def get_triplet_extractor(device: Optional[Union[str, int]] = None):
    """Returns OpenAI client for GPT-4o mini triplet extraction."""
    return get_openai_client()

def _truncate_to_max_tokens(text: str, max_tokens: int = 8000) -> str:
    """Truncate text to approximate token limit for GPT-4o mini (rough estimation)."""
    # GPT-4o mini has ~128k context, but we use conservative limit
    # Rough estimation: 1 token ≈ 4 characters for English text
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

def _extract_triplets_from_generated(text: str) -> Set[Tuple[str, str, str]]:
    """Legacy REBEL format parser - kept for compatibility."""
    triplets = []
    relation, subject, object_ = "", "", ""
    current = None
    for tok in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if tok == "<triplet>":
            current = "t"
            if subject and relation and object_:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            subject = relation = object_ = ""
        elif tok == "<subj>":
            current = "s"
            if subject and relation and object_:
                triplets.append((subject.strip(), relation.strip(), object_.strip()))
            object_ = ""
        elif tok == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":   subject += (" " if subject else "") + tok
            elif current == "s": object_ += (" " if object_ else "") + tok
            elif current == "o": relation += (" " if relation else "") + tok
    if subject and relation and object_:
        triplets.append((subject.strip(), relation.strip(), object_.strip()))
    return { (h.strip(), r.strip(), t.strip()) for (h, r, t) in triplets if h and r and t }

def _extract_triplets_from_gpt_response(text: str) -> Set[Tuple[str, str, str]]:
    """Parse triplets from GPT-4o mini response."""
    triplets = []
    
    # Try to parse the structured format first
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('*'):
            continue
            
        # Look for the structured format: <triplet> subject <subj> object <obj> relation </s>
        if '<triplet>' in line and '<subj>' in line and '<obj>' in line:
            try:
                # Extract components using the tags
                parts = line.split('<triplet>')[-1].split('</s>')[0]  # Get content between tags
                
                if '<subj>' in parts and '<obj>' in parts:
                    # Split by the tags
                    subj_split = parts.split('<subj>')
                    subject = subj_split[0].strip()
                    
                    obj_split = subj_split[1].split('<obj>')
                    object_ = obj_split[0].strip()
                    relation = obj_split[1].strip()
                    
                    if subject and relation and object_:
                        triplets.append((subject, relation, object_))
            except:
                continue
        
        # Fallback: try to parse simple format like "subject | relation | object"
        elif '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 3 and all(parts):
                triplets.append((parts[0], parts[1], parts[2]))
    
    return { (h.strip(), r.strip(), t.strip()) for (h, r, t) in triplets if h and r and t }


def _coerce_to_text(x: Any) -> str:
    """Make sure the pipeline input is a UTF-8 string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, (pd.Series, pd.Index)):
        x = x.astype(str).tolist()
    if isinstance(x, np.ndarray):
        x = x.flatten().tolist()
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        # join list/tuple of chunks into one doc
        return " ".join(_coerce_to_text(t) for t in x)
    if isinstance(x, Mapping):
        # turn dict into a readable text blob
        return " ".join(f"{k}: {_coerce_to_text(v)}" for k, v in x.items())
    return str(x)


def triplet_parser(text: str, *, device: Optional[Union[str,int]] = None, max_new_tokens: int = 256):
    """Extract triplets using GPT-4o mini API."""
    client = get_triplet_extractor(device)
    text = _coerce_to_text(text)
    truncated = _truncate_to_max_tokens(text, max_tokens=8000)
    
    # Create prompt for triplet extraction
    prompt = f"""Extract relationship triplets from the following text. 
Return triplets in the format: <triplet> subject <subj> object <obj> relation </s>

Examples:
- "John works at Google" → <triplet> John <subj> Google <obj> works at </s>
- "The cat sits on the mat" → <triplet> cat <subj> mat <obj> sits on </s>

Text to analyze:
{truncated}

Extracted triplets:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured relationship triplets from text. Always follow the exact format specified."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_new_tokens,
            temperature=0.1,
            top_p=1.0
        )
        
        generated = response.choices[0].message.content or ""
        return _extract_triplets_from_gpt_response(generated)
        
    except Exception as e:
        print(f"Error calling GPT-4o mini: {e}")
        return set()  # Return empty set on error

# ---------------------------------------------------------------------------
# Installation requirements:
# pip install openai

# Usage example:
if __name__ == "__main__":
    # Set your OpenAI API key first:
    # export OPENAI_API_KEY="your-api-key-here"
    
    test_text = "John works at Google. Mary is the CEO of Tesla. The cat sits on the mat."
    triples = triplet_parser(test_text, max_new_tokens=512)
    print(f"{len(triples)} triples extracted:")
    for subject, relation, obj in triples:
        print(f"  {subject} --[{relation}]--> {obj}")

# python graph_generator/4oparser.py