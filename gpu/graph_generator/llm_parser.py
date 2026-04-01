import openai
import os
import json
import time
from functools import lru_cache
from typing import Optional, Union, Set, Tuple, List, Dict
import asyncio
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd


# -----------------------------
# 1) Global token tracker
# -----------------------------
TOKEN_STATS = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}

def record_usage(usage):
    """usage is response.usage from OpenAI; update global totals."""
    if not usage:
        return
    TOKEN_STATS["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
    TOKEN_STATS["completion_tokens"] += getattr(usage, "completion_tokens", 0)
    TOKEN_STATS["total_tokens"] += getattr(usage, "total_tokens", 0)


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



_client_cache: Dict[Tuple[str, Optional[str]], Any] = {}

def _client_from_api(api: Optional[Union[str, Mapping]] = None):
    """
    Build an OpenAI client from:
      - str: treated as API key
      - Mapping: may include {"api_key": "...", "base_url": "..."} (base_url optional)
      - None: fall back to env-based client
    Caches clients by (api_key, base_url) to avoid re-creating ~100-200ms per call.
    """
    if api is None:
        return get_triplet_extractor(None)  # existing cached env-based client

    if isinstance(api, str):
        cache_key = (api, None)
        if cache_key not in _client_cache:
            _client_cache[cache_key] = openai.OpenAI(api_key=api)
        return _client_cache[cache_key]

    if isinstance(api, Mapping):
        api_key = api.get("api_key") or api.get("key") or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        base_url = api.get("base_url") or api.get("endpoint")
        cache_key = (api_key, base_url)
        if cache_key not in _client_cache:
            if base_url:
                _client_cache[cache_key] = openai.OpenAI(api_key=api_key, base_url=base_url)
            else:
                _client_cache[cache_key] = openai.OpenAI(api_key=api_key)
        return _client_cache[cache_key]

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
    max_new_tokens: int = 256,
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

        record_usage(response.usage)
        
        generated = response.choices[0].message.content or ""
        return _extract_triplets_from_gpt_response(generated)
        
    except Exception as e:
        print(f"Error calling GPT-4o mini: {e}")
        return set()  # Return empty set on error

def _extract_triplets_batch(client, texts: List[str], max_new_tokens: int,model: str = "gpt-4o-mini") -> List[Set[Triplet]]:
    """Extract triplets from multiple texts using GPT-4o mini (sequential)."""
    truncated = [_truncate_to_max_tokens(t, max_tokens=8000) for t in texts]
    results = []
    for text in truncated:
        triplets = _extract_triplets_single(client, text, max_new_tokens, model)
        results.append(triplets)
    return results


# ============================================================
# Question-specific triple extraction
# ============================================================
#
# Questions need fundamentally different triples than factual text.
# Factual text:  "John works at Google" → (John, works at, Google)
# Question text: "Who is the mother of the director of Film X?"
#   BAD:  (director, of film, Film X)  +  (mother, of, director)
#         → "director" and "mother" are useless as entity matches
#   GOOD: (mother of the director, of, Film X)
#         → "Film X" is the anchor entity, matches codebook
#
# The question parser produces triples where:
#   - Named entities (proper nouns, titles) go in entity slots
#   - Relational chains ("mother of the director") pack into subject/relation
#   - Comparison questions get one triple per compared entity

_QUESTION_TRIPLE_PROMPT = """You are a knowledge graph query planner. Given a question, extract structured triples that can be used to search a knowledge graph.

RULES:
1. Keep all named entities (people, places, films, organizations, dates) as-is — never replace them with generic words.
2. Pack descriptive chains into the SUBJECT. The OBJECT should always be a specific named entity from the question.
3. The RELATION should describe the connection type between subject and object.
4. For comparison questions ("Do X and Y share..."), produce one triple per entity.
5. Output format: <triplet> subject <subj> object <obj> relation </s>

EXAMPLES:
Q: "Who is the mother of the director of film Polish-Russian War?"
→ <triplet> mother of the director <subj> Polish-Russian War <obj> of film </s>

Q: "Where was the director of Ronnie Rocket born?"
→ <triplet> birthplace of the director <subj> Ronnie Rocket <obj> of film </s>

Q: "Do The Raven's Dance and Keita! The Heritage of the Griot have the same directors from the same country?"
→ <triplet> director country <subj> The Raven's Dance <obj> of film </s>
<triplet> director country <subj> Keita! The Heritage of the Griot <obj> of film </s>

Q: "When did John V, Prince of Anhalt-Zerbst's father die?"
→ <triplet> death date of father <subj> John V, Prince of Anhalt-Zerbst <obj> of </s>

Q: "Which film has the director born first, The Millerson Case or 711 Ocean Drive?"
→ <triplet> director birth date <subj> The Millerson Case <obj> of film </s>
<triplet> director birth date <subj> 711 Ocean Drive <obj> of film </s>

Q: "Are Kaufland and Otrag both headquartered in the same country?"
→ <triplet> headquarters country <subj> Kaufland <obj> of </s>
<triplet> headquarters country <subj> Otrag <obj> of </s>

Q: "Who is the maternal grandfather of Antiochus X Eusebes?"
→ <triplet> maternal grandfather <subj> Antiochus X Eusebes <obj> of </s>

Now extract triples for:
Q: "{question}"
"""




_QUESTION_TRIPLE_EXPANDED_PROMPT = """You are a knowledge graph query planner. Given a question, extract expanded multi-hop query triples.

RULES:
1. Keep named entities exactly as they appear.
2. Prefer explicit intermediate variables for relational chains (use ?variable notation).
3. Use the most concrete relation possible.
4. For simple single-hop questions, one triple is enough.
5. For comparison questions ("Do X and Y share..."), produce one triple per entity — NO intermediate variables needed.
6. Output format: <triplet> subject <subj> object <obj> relation </s>

EXAMPLES:
Q: "Who is the mother of the director of film Polish-Russian War?"
→ <triplet> Polish-Russian War <subj> ?director <obj> directed by </s>
<triplet> ?director <subj> ?mother <obj> mother </s>

Q: "Where was the director of Ronnie Rocket born?"
→ <triplet> Ronnie Rocket <subj> ?director <obj> directed by </s>
<triplet> ?director <subj> ?place <obj> born in </s>

Q: "Do The Raven's Dance and Keita! The Heritage of the Griot have the same directors from the same country?"
→ <triplet> The Raven's Dance <subj> ?country1 <obj> director country </s>
<triplet> Keita! The Heritage of the Griot <subj> ?country2 <obj> director country </s>

Q: "When did John V, Prince of Anhalt-Zerbst's father die?"
→ <triplet> John V, Prince of Anhalt-Zerbst <subj> ?father <obj> father </s>
<triplet> ?father <subj> ?date <obj> death date </s>

Q: "Are Kaufland and Otrag both headquartered in the same country?"
→ <triplet> Kaufland <subj> ?country1 <obj> headquarters country </s>
<triplet> Otrag <subj> ?country2 <obj> headquarters country </s>

Q: "Which film has the director born first, The Millerson Case or 711 Ocean Drive?"
→ <triplet> The Millerson Case <subj> ?director1 <obj> directed by </s>
<triplet> ?director1 <subj> ?date1 <obj> birth date </s>
<triplet> 711 Ocean Drive <subj> ?director2 <obj> directed by </s>
<triplet> ?director2 <subj> ?date2 <obj> birth date </s>

Now extract triples for:
Q: "{question}"
"""


def _extract_triplets_single_question_expanded(
    client, text: str, max_new_tokens: int, model: str = "gpt-4o-mini"
) -> Set[Triplet]:
    prompt = _QUESTION_TRIPLE_EXPANDED_PROMPT.format(question=text)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You extract expanded multi-hop query triples from questions. Output ONLY triples."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        record_usage(response.usage)
        generated = response.choices[0].message.content or ""
        return _extract_triplets_from_gpt_response(generated)
    except Exception as e:
        print(f"Error in expanded question triple extraction: {e}")
        return set()

def _extract_triplets_single_question(
    client, text: str, max_new_tokens: int, model: str = "gpt-4o-mini"
) -> Set[Triplet]:
    """Extract triples from a QUESTION using the question-specific prompt."""
    prompt = _QUESTION_TRIPLE_PROMPT.format(question=text)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You extract structured knowledge graph query triples from questions. "
                            "Always follow the exact format specified. Output ONLY triples, no explanations."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        record_usage(response.usage)
        generated = response.choices[0].message.content or ""
        return _extract_triplets_from_gpt_response(generated)

    except Exception as e:
        print(f"Error in question triple extraction: {e}")
        return set()




# ============================================================
# Unified structured question parser: HOPS + RETRIEVAL + GROUPS
# ============================================================

_QUESTION_STRUCTURED_PROMPT = """You are a knowledge graph query planner. Given a question, output three sections: HOPS, RETRIEVAL, and GROUPS.

HOPS: The question broken into minimal single-hop triples with ?variable placeholders for unknowns.
  - Each hop is one edge: subject | relation | object
  - Named entities stay as-is. Unknowns use ?variable names.
  - Label each hop H1, H2, etc.

RETRIEVAL: Retrieval-friendly compressed triples where descriptive chains are packed into the subject.
  - These are for searching a knowledge base — keep the named entity as object.
  - Label each R1, R2, etc. One R per independent branch.

GROUPS: Which hops chain together (share a ?variable).
  - Chain: [H1 > H2] means H1 must resolve before H2.
  - Independent: [H3] alone.

EXAMPLES:

Q: "Who is the mother of the director of film Polish-Russian War?"
HOPS:
H1: Polish-Russian War | directed_by | ?director
H2: ?director | mother | ?mother
RETRIEVAL:
R1: mother of the director | of film | Polish-Russian War
GROUPS:
[H1 > H2]

Q: "Do The Raven's Dance and Keita! The Heritage of the Griot have the same directors from the same country?"
HOPS:
H1: The Raven's Dance | director_country | ?country1
H2: Keita! The Heritage of the Griot | director_country | ?country2
RETRIEVAL:
R1: director country | of film | The Raven's Dance
R2: director country | of film | Keita! The Heritage of the Griot
GROUPS:
[H1]
[H2]

Q: "Which film has the director born first, The Millerson Case or 711 Ocean Drive?"
HOPS:
H1: The Millerson Case | directed_by | ?director1
H2: ?director1 | birth_date | ?date1
H3: 711 Ocean Drive | directed_by | ?director2
H4: ?director2 | birth_date | ?date2
RETRIEVAL:
R1: director birth date | of film | The Millerson Case
R2: director birth date | of film | 711 Ocean Drive
GROUPS:
[H1 > H2]
[H3 > H4]

Q: "When did John V, Prince of Anhalt-Zerbst's father die?"
HOPS:
H1: John V, Prince of Anhalt-Zerbst | father | ?father
H2: ?father | death_date | ?date
RETRIEVAL:
R1: death date of father | of | John V, Prince of Anhalt-Zerbst
GROUPS:
[H1 > H2]

Q: "Where was Film A filmed?"
HOPS:
H1: Film A | filmed_in | ?location
RETRIEVAL:
R1: filmed in | of | Film A
GROUPS:
[H1]

Now output HOPS, RETRIEVAL, and GROUPS for:
Q: "{question}"
"""


def _parse_structured_response(text: str) -> Dict[str, Any]:
    """Parse the structured HOPS/RETRIEVAL/GROUPS response from the LLM."""
    hops: Dict[str, Tuple[str, str, str]] = {}
    retrieval: Dict[str, Tuple[str, str, str]] = {}
    groups: List[List[str]] = []

    current_section = None
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith('HOPS'):
            current_section = 'hops'
            continue
        elif upper.startswith('RETRIEVAL'):
            current_section = 'retrieval'
            continue
        elif upper.startswith('GROUPS') or upper.startswith('GROUP'):
            current_section = 'groups'
            continue

        if current_section in ('hops', 'retrieval'):
            # Parse "H1: subject | relation | object" or "R1: ..."
            if ':' in line:
                label, rest = line.split(':', 1)
                label = label.strip()
            else:
                label = f"{'H' if current_section == 'hops' else 'R'}{len(hops if current_section == 'hops' else retrieval) + 1}"
                rest = line
            parts = [p.strip() for p in rest.split('|')]
            if len(parts) >= 3:
                triple = (parts[0], parts[1], parts[2])
                if current_section == 'hops':
                    hops[label] = triple
                else:
                    retrieval[label] = triple
            elif len(parts) == 2:
                # Fallback: treat as (subject, relation, "?")
                triple = (parts[0], parts[1], "?")
                if current_section == 'hops':
                    hops[label] = triple
                else:
                    retrieval[label] = triple

        elif current_section == 'groups':
            # Parse "[H1 > H2]" or "[H3]"
            line = line.strip('[]() ')
            if not line:
                continue
            group_labels = [g.strip() for g in line.split('>')]
            group_labels = [g for g in group_labels if g]
            if group_labels:
                groups.append(group_labels)

    # If no groups parsed, infer: each hop is independent
    if not groups and hops:
        groups = [[label] for label in hops]

    # If no retrieval parsed, fall back to hop triples
    if not retrieval and hops:
        for label, triple in hops.items():
            rlabel = label.replace('H', 'R')
            retrieval[rlabel] = triple

    return {
        "hops": hops,
        "retrieval": retrieval,
        "groups": groups,
    }


def triplet_parser_llm_question_structured(
    question: str,
    *,
    api: Optional[Union[str, Mapping]] = None,
    client: Optional[Any] = None,
    model: str = "gpt-4o-mini",
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Unified structured question parser.

    Single LLM call → HOPS + RETRIEVAL + GROUPS.

    Returns:
        {
            "hops": {"H1": (s, r, o), "H2": (s, r, o), ...},
            "retrieval": {"R1": (s, r, o), ...},
            "groups": [["H1", "H2"], ["H3"], ...],
            "hops_triples": Set[Triplet],       # for codebook building
            "retrieval_triples": Set[Triplet],   # for codebook building
        }
    """
    import time as _time
    _t_total = _time.perf_counter()

    _t0 = _time.perf_counter()
    _client = client or _client_from_api(api)
    _t_client = _time.perf_counter() - _t0

    _t0 = _time.perf_counter()
    text = _coerce_to_text(question)
    truncated = _truncate_to_max_tokens(text, max_tokens=4000)
    prompt = _QUESTION_STRUCTURED_PROMPT.format(question=truncated)
    _t_prep = _time.perf_counter() - _t0

    try:
        _t0 = _time.perf_counter()
        response = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are a knowledge graph query planner. "
                            "Output ONLY the HOPS, RETRIEVAL, and GROUPS sections. No explanations."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        _t_api = _time.perf_counter() - _t0

        _t0 = _time.perf_counter()
        record_usage(response.usage)
        generated = response.choices[0].message.content or ""
        parsed = _parse_structured_response(generated)

        # Build triple sets for codebook construction
        parsed["hops_triples"] = set(parsed["hops"].values())
        parsed["retrieval_triples"] = set(parsed["retrieval"].values())
        _t_parse = _time.perf_counter() - _t0

        n_hops = len(parsed["hops"])
        n_ret = len(parsed["retrieval"])
        n_groups = len(parsed["groups"])
        _t_total_elapsed = _time.perf_counter() - _t_total
        print(f"[structured_parser] {n_hops} hops, {n_ret} retrieval, {n_groups} groups "
              f"| client={_t_client*1000:.0f}ms api={_t_api*1000:.0f}ms "
              f"prep={_t_prep*1000:.0f}ms parse={_t_parse*1000:.0f}ms "
              f"total={_t_total_elapsed*1000:.0f}ms")

        return parsed

    except Exception as e:
        print(f"Error in structured question parsing: {e}")
        # Fallback: use dual parser
        _compressed = _extract_triplets_single_question(_client, truncated, max_new_tokens, model)
        _expanded = _extract_triplets_single_question_expanded(_client, truncated, max_new_tokens, model)
        return {
            "hops": {f"H{i+1}": t for i, t in enumerate(_expanded or _compressed)},
            "retrieval": {f"R{i+1}": t for i, t in enumerate(_compressed)},
            "groups": [[f"H{i+1}"] for i in range(len(_expanded or _compressed))],
            "hops_triples": _expanded or _compressed,
            "retrieval_triples": _compressed,
        }


def triplet_parser_llm_question_dual(
    question: str,
    *,
    api: Optional[Union[str, Mapping]] = None,
    client: Optional[Any] = None,
    model: str = "gpt-4o-mini",
    max_new_tokens: int = 256,
) -> Dict[str, Set[Triplet]]:
    """
    Return both retrieval-friendly compressed triples and structure-friendly expanded triples.

    compressed:
        packs descriptive chains into subject/relation and keeps the named anchor entity intact.
    expanded:
        attempts to expose explicit intermediate variables for downstream structure inference.
    """
    _client = client or _client_from_api(api)
    text = _coerce_to_text(question)
    truncated = _truncate_to_max_tokens(text, max_tokens=4000)
    compressed = _extract_triplets_single_question(_client, truncated, max_new_tokens, model)
    expanded = _extract_triplets_single_question_expanded(_client, truncated, max_new_tokens, model)
    return {
        "compressed": compressed,
        "expanded": expanded if expanded else compressed,
    }

def triplet_parser_llm_question(
    question: str,
    *,
    api: Optional[Union[str, Mapping]] = None,
    client: Optional[Any] = None,
    model: str = "gpt-4o-mini",
    max_new_tokens: int = 256,
) -> Set[Triplet]:
    """
    Question-specific triple extraction.

    Unlike triplet_parser_llm (designed for factual passages), this uses a
    prompt that keeps named entities as objects and packs relational chains
    into subjects — producing triples aligned with knowledge graph retrieval.

    Parameters:
      question       : the raw question string
      api            : str (API key) or Mapping {"api_key": ..., "base_url": ...}
      client         : pre-built OpenAI client (takes precedence over api)
      model          : model name (default "gpt-4o-mini")
      max_new_tokens : response length cap

    Returns:
      Set of (subject, relation, object) triples
    """
    _client = client or _client_from_api(api)
    text = _coerce_to_text(question)
    truncated = _truncate_to_max_tokens(text, max_tokens=4000)
    return _extract_triplets_single_question(_client, truncated, max_new_tokens, model)

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