from transformers import pipeline
from functools import lru_cache
from typing import Optional, Union
import torch

@lru_cache(maxsize=3)
def get_triplet_extractor(device: Optional[Union[str, int]] = None):
    """
    device:
      - 'mps'  -> Apple MPS
      - -1     -> CPU
      - 0/1... -> CUDA GPU
      - None   -> 自动检测（优先 mps，否则 CPU）
    """
    if device is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = 0
        else:
            device = -1
    elif isinstance(device, str) and device.lower() == "mps":
        device = torch.device("mps")

    # 关键点：用 device=...，不要用 torch_device=...
    return pipeline(
        task="text2text-generation",
        model="Babelscape/rebel-large",
        tokenizer="Babelscape/rebel-large",
        device=device,                 # ✅ 正确姿势
        # torch_dtype=torch.float32,   # 可选：MPS 上通常用 fp32 更稳
    )

def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
    triplets = {
        (d['head'].strip(), d['type'].strip(), d['tail'].strip())
        for d in triplets
        if isinstance(d, dict) and d.get('head') and d.get('type') and d.get('tail')
    }
    return triplets

def triplet_parser(text: str, *, device: Optional[str|int] = None, max_new_tokens: int = 256):
    extractor = get_triplet_extractor(device)
    out = extractor(text, return_text=True, return_tensors=True, max_new_tokens=max_new_tokens)
    rec = out[0]
    if "generated_token_ids" in rec and hasattr(extractor, "tokenizer"):
        decoded = extractor.tokenizer.batch_decode([rec["generated_token_ids"]])
        gen = decoded[0]
    else:
        gen = rec.get("generated_text") or str(rec)
    return extract_triplets(gen)

# if __name__ == "__main__":
#     print(triplet_parser("Punta Cana is a resort town in the municipality of Higuey, in La Altagracia Province, the eastern most province of the Dominican Republic"))
#     print(len(triplet_parser("Punta Cana is a resort town in the municipality of Higuey, in La Altagracia Province, the eastern most province of the Dominican Republic")))
#     long_text ="Basal cell carcinoma (BCC) arises." * 2
#     triples = triplet_parser(long_text, device=0)
#     print(len(triples), "triples extracted")

# python graph_generator/rebel.py