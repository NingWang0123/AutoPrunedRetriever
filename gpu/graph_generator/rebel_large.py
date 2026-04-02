from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from functools import lru_cache
from typing import Optional, Union, Set, Tuple
import torch
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

def _disable_sdpa_on_cuda():
    if torch.cuda.is_available():
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel.enable_flash_sdp(False)
            sdp_kernel.enable_mem_efficient_sdp(False)
            sdp_kernel.enable_math_sdp(True)
        except Exception:
            pass

@lru_cache(maxsize=2)
def get_triplet_extractor(device: Optional[Union[str, int]] = "auto"):
    # device: "auto" | None | "cuda" | "mps" | "cpu" | int
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = 0  # CUDA
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = -1  # CPU
    elif isinstance(device, str):
        if device.lower().startswith("cuda"):
            device = 0
        elif device.lower() == "mps":
            device = torch.device("mps")
        elif device.lower() == "cpu":
            device = -1

    _disable_sdpa_on_cuda()

    model_name = "Babelscape/rebel-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Prefer new kw `dtype`; fall back to `torch_dtype` if needed
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=torch.float32, low_cpu_mem_usage=True
        )
    except TypeError:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )

    return pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        # set defaults explicitly to avoid warnings and surprises
        do_sample=False,
        num_beams=1,
    )

def _truncate_to_max_tokens(text: str, tok: AutoTokenizer) -> str:
    max_len = getattr(tok, "model_max_length", 1024) or 1024
    # encode + truncate, then decode back (skips specials)
    ids = tok.encode(text or "", add_special_tokens=True, truncation=True, max_length=max_len)
    return tok.decode(ids, skip_special_tokens=True)

def _extract_triplets_from_generated(text: str) -> Set[Tuple[str, str, str]]:
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


def triplet_parser(text: str, *, device: Optional[Union[str,int]] = "auto", max_new_tokens: int = 256):
    pipe = get_triplet_extractor(device)
    text = _coerce_to_text(text)                         # ensure string
    truncated = _truncate_to_max_tokens(text, pipe.tokenizer)

    try:
        # Ask the pipeline for token IDs (don't auto-decode to text)
        out = pipe(truncated, max_new_tokens=max_new_tokens,
                   return_tensors=True, return_text=False)
    except RuntimeError as e:
        if "CUDA error" in str(e) or "device-side assert" in str(e):
            get_triplet_extractor.cache_clear()
            cpu_pipe = get_triplet_extractor(-1)
            out = cpu_pipe(truncated, max_new_tokens=max_new_tokens,
                           return_tensors=True, return_text=False)
        else:
            raise

    # --- CRITICAL: decode ourselves and KEEP special tokens ---
    gen_ids = out[0]["generated_token_ids"]
    generated = pipe.tokenizer.batch_decode([gen_ids], skip_special_tokens=False)[0]

    return _extract_triplets_from_generated(generated)

# ---------------------------------------------------------------------------


# # 
# if __name__ == "__main__":
#     long_text ="Basal cell carcinoma (BCC) arises." * 2000
#     triples = triplet_parser(long_text, device=0, max_new_tokens=256)
#     print(len(triples), "triples extracted")

# python graph_generator/rebel_large.py