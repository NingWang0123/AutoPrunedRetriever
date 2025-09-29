import json
import re 
import pandas as pd

cr_reuslts_path = 'results/cr_results/compressrag_medical_data.json'

# load the json file

with open(cr_reuslts_path, 'r', encoding='utf-8') as f:
    cr_results = json.load(f)


start_with_context = []

i=0
for d in cr_results:

    generated_answer = d['generated_answer']

    if '<<<RETRIEVED_CONTEXT_START>>>' in generated_answer or 'no answer' in generated_answer:
        start_with_context.append(i)

    i+=1


print(f'cr has {len(start_with_context)} missings')



light_rag_path =  'results/light_rag_results/predictions_Medical_lightrag.json'


with open(light_rag_path, 'r', encoding='utf-8') as f:
    light_rag_results = json.load(f)


contain_idont_know = []

i = 0
for d in light_rag_results:   # don't use `dict` as a variable name
    generated_answer = d['generated_answer']

    if "I don't know" in generated_answer:
        contain_idont_know.append(i)

    i += 1

print(f'light rag has {len(contain_idont_know)} missings')


same_missings = []
for i in contain_idont_know:
    if i in start_with_context:
        same_missings.append(i)


print(f'Totaly {len(same_missings)} same missings')



# pattern to catch variations like "I don't know", "i dont know", extra spaces, etc.
IDK_PATTERN = re.compile(r"\bi\s*do(?:n'?t)?\s*know\b", re.IGNORECASE)

rows = []
for i, d in enumerate(light_rag_results):
    ans = d.get("generated_answer", "") or ""
    if not IDK_PATTERN.search(ans):  # keep only answers WITHOUT "i don't know"
        gi = d.get("gen_info", {}) or {}
        ev = d.get("eval", {}) or {}
        rows.append({
            "index": i,
            "generated_answer": ans,

            # ---- gen_info ----
            "input_tokens": gi.get("input_tokens"),
            "output_tokens": gi.get("output_tokens"),
            "total_tokens": gi.get("total_tokens"),
            "latency_sec": gi.get("latency_sec"),
            "gen_latency_sec": gi.get("gen_latency_sec"),
            "retrieval_latency_sec": gi.get("retrieval_latency_sec"),
            "peak_vram_MiB": gi.get("peak_vram_MiB"),
            "prompt_chars": gi.get("prompt_chars"),
            "throughput_tok_per_s": gi.get("throughput_tok_per_s"),
            "prompt_tok_per_s": gi.get("prompt_tok_per_s"),
            "device": gi.get("device"),
            "dtype": gi.get("dtype"),
            "model_name": gi.get("model_name"),
            "timestamp_start": gi.get("timestamp_start"),
            "timestamp_end": gi.get("timestamp_end"),

            # ---- eval ----
            "correctness_sbert": ev.get("correctness_sbert"),
            "context_similarity_sbert": ev.get("context_similarity_sbert"),
        })

df_no_idk = pd.DataFrame(rows)

print(f"Kept {len(df_no_idk)} / {len(light_rag_results)} rows without 'I don't know'.")
# Optional: quick peek
print(df_no_idk.head())

#  python filter_ungenerated_results.py
