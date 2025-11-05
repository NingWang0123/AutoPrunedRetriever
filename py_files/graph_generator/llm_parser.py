import openai
import os
import json
import time
from functools import lru_cache
from typing import Optional, Union, Set, Tuple, List
import asyncio
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

# Type definitions to match REBEL interface
Triplet = Tuple[str, str, str]

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



def _client_from_api(api: Optional[Union[str, Mapping]] = None):
    """
    Build an OpenAI client from:
      - str: treated as API key
      - Mapping: may include {"api_key": "...", "base_url": "..."} (base_url optional)
      - None: fall back to env-based client
    """
    if api is None:
        return get_triplet_extractor(None)  # existing cached env-based client

    if isinstance(api, str):
        return openai.OpenAI(api_key=api)

    if isinstance(api, Mapping):
        api_key = api.get("api_key") or api.get("key") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = api.get("base_url") or api.get("endpoint")
        if base_url:
            return openai.OpenAI(api_key=api_key, base_url=base_url)
        return openai.OpenAI(api_key=api_key)

    # Fallback
    return get_triplet_extractor(None)


def triplet_parser(
    text_or_list: Union[str, List[str]],
    *,
    device: Optional[Union[str, int]] = None,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    num_beams: int = 1,
) -> Union[Set[Triplet], List[Set[Triplet]]]:
    """Extract triplets using GPT-4o mini API - compatible with REBEL interface."""
    client = get_triplet_extractor(device)
    
    # Handle single string input
    if isinstance(text_or_list, str):
        text = _coerce_to_text(text_or_list)
        truncated = _truncate_to_max_tokens(text, max_tokens=8000)
        return _extract_triplets_single(client, truncated, max_new_tokens)
    
    # Handle list input with batching
    texts: List[str] = [_coerce_to_text(t) for t in text_or_list]
    results: List[Set[Triplet]] = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = _extract_triplets_batch(client, batch, max_new_tokens)
        results.extend(batch_results)
    
    return results


def triplet_parser_llm(
    text_or_list: Union[str, List[str]],
    *,
    device: Optional[Union[str, int]] = None,
    batch_size: int = 8,
    max_new_tokens: int = 1200,
    do_sample: bool = False,
    num_beams: int = 1,
    api: Optional[Union[str, Mapping]] = None,   
    client: Optional[Any] = None,                
    model: str = "gpt-4o-mini",                  
) -> Union[Set[Triplet], List[Set[Triplet]]]:
    """
    Extract triplets using GPT-4o mini API - compatible with REBEL interface.

    Parameters new:
      - api: str (API key) or Mapping (e.g., {"api_key": "...", "base_url": "..."})
      - client: pre-initialized OpenAI client; takes precedence over `api`
      - model: model name (default "gpt-4o-mini")
    """
    # Resolve client preference: client > api > env
    _client = client or _client_from_api(api)

    # Bind model name into the single-call helper via a small wrapper
    def _single(text: str) -> Set[Triplet]:
        return _extract_triplets_single(_client, text, max_new_tokens, model=model)

    # Handle single string input
    if isinstance(text_or_list, str):
        text = _coerce_to_text(text_or_list)
        truncated = _truncate_to_max_tokens(text, max_tokens=8000)
        return _single(truncated)

    # Handle list input with batching
    texts: List[str] = [_coerce_to_text(t) for t in text_or_list]
    results: List[Set[Triplet]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = _extract_triplets_batch(_client, batch, max_new_tokens, model=model)
        results.extend(batch_results)
    return results

def _extract_triplets_single(client, text: str, max_new_tokens: int,model: str = "gpt-4o-mini") -> Set[Triplet]:
    """Extract triplets from a single text using GPT-4o mini."""
    # Create prompt for triplet extraction
    prompt = f"""Extract relationship triplets from the following text. 
Return triplets in the format: <triplet> subject <subj> object <obj> relation </s> and If you know the information give the exact information to replace he/she/we/they/it... with the exact info.

Examples:
- "John works at Google" → <triplet> John <subj> Google <obj> works at </s> if it follows "he likes apples", you should replace he with John.
- "The cat sits on the mat" → <triplet> cat <subj> mat <obj> sits on </s>

Text to analyze:
{text}

Extracted triplets:"""

    try:
        response = client.chat.completions.create(
            model=model,
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

def _extract_triplets_batch(client, texts: List[str], max_new_tokens: int,model: str = "gpt-4o-mini") -> List[Set[Triplet]]:
    """Extract triplets from multiple texts using GPT-4o mini."""
    results = []
    
    for text in texts:
        truncated = _truncate_to_max_tokens(text, max_tokens=8000)
        result = _extract_triplets_single(client, truncated, max_new_tokens,model)
        results.append(result)
        
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    return results

# ---------------------------------------------------------------------------
# Installation requirements:
# pip install openai

# Usage example matching REBEL interface:
# if __name__ == "__main__":
#     # Set your OpenAI API key first:
#     # export OPENAI_API_KEY="your-api-key-here"
    
#     # Single string input (like REBEL)
#     s = "About basal cell skin cancer What is basal cell skin cancer? How is basal cell skin cancer treated? What can you do to get the best care? Basal cell skin cancer, also known as basal cell carcinoma (BCC), is the most common type of skin cancer. About 3 million cases of basal cell skin cancer are diagnosed every year in the United States. The good news is it can be cured in most cases. Treatment usually involves surgery to remove the cancer. Keep reading to find out more. What is basal cell skin cancer? Basal cell skin cancer is the most common of all skin cancer types. If caught early, it is easily treatable and curable. This is because it rarely metastasizes (spreads). Skin cancers often occur in the top layer of the skin (epidermis) and less commonly in the middle layer of the skin (dermis). The epidermis is made up of basal cells and other cells."
#     print("Single text result:")
#     print(triplet_parser(s, device="mps"))            

#     # List input with batch processing (like REBEL)
#     lst = [
#         "About basal cell skin cancer What is basal cell skin cancer?",
#         "How is basal cell skin cancer treated? What can you do to get the best care?",
#         "Basal cell skin cancer, also known as basal cell carcinoma (BCC), is the most common type of skin cancer.",
#         "About 3 million cases of basal cell skin cancer are diagnosed every year in the United States.",
#         "The good news is it can be cured in most cases.",
#         "Treatment usually involves surgery to remove the cancer.",
#         "Keep reading to find out more.",
#         "What is basal cell skin cancer?",
#         "Basal cell skin cancer is the most common of all skin cancer types.",
#         "If caught early, it is easily treatable and curable.",
#         "This is because it rarely metastasizes (spreads).",
#     ]
#     print("\nBatch processing result:")
#     batch_results = triplet_parser(lst, device="mps", batch_size=4)
#     for i, result in enumerate(batch_results):
#         print(f"Text {i+1}: {len(result)} triplets")
#         for triplet in list(result)[:3]:  # Show first 3 triplets
#             print(f"  {triplet}")

# python graph_generator/4omini.py