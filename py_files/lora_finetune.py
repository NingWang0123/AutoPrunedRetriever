# fine_tune_lora_adapter_only.py
# -*- coding: utf-8 -*-

import os
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer.sft_config import SFTConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model


from CompressRag_rl_v2 import (
    get_code_book, merging_codebook, slice_for_final_merged_json
)
from WordEmb import Word2VecEmbeddings

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16  
    device = "cpu"
    dtype = torch.float32

print(f"[INFO] Using device={device}, dtype={dtype}")

BASE_MODEL_ID = "Qwen/Qwen3-4B"
DATA_PATH     = "context/medical_questions.json"
OUTPUT_DIR    = "lora_qwen_sft_adapter"   
LOG_DIR       = os.path.join(OUTPUT_DIR, "logs")
DATA_SLICE    = 100
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

raw_data = raw_data[:DATA_SLICE]
print(f"[INFO] Loaded {len(raw_data)} samples for training")

def build_prompt(question: str, evidence_list) -> str:
    system_msg = (
        "You are a precise QA agent that answers by expressing facts as short, "
        "plain English statements. Keep outputs concise and factual."
    )

    ev_lines = []
    if evidence_list:
        for ev in evidence_list:
            ev = " ".join(str(ev).split())
            ev_lines.append(f"- {ev}")
    else:
        ev_lines.append("- (no evidence provided)")

    # evidence -> codebook -> 压缩 JSON 片段
    fact_cb = get_code_book(ev_lines, type='facts', rule="Store factual statements.")
    word_emb = Word2VecEmbeddings(model_name="word2vec-google-news-300")

    combined = {
        "e": list(fact_cb.get("e", [])),
        "r": list(fact_cb.get("r", [])),
        "edge_matrix": list(fact_cb.get("edges([e,r,e])", [])),
        "facts(edges[i])": [lst for lst in fact_cb.get("facts(edges[i])", [])],
        "questions_lst": [], "answers_lst": [], "thinkings_lst": [],
        "rule": fact_cb.get("rule", "Store factual statements."),
        "e_embeddings": [], "r_embeddings": [],
    }
    combined_facts_cb = merging_codebook(None, combined, 'facts', word_emb, False)
    combined_facts_cb.pop("e_embeddings", None)
    combined_facts_cb.pop("r_embeddings", None)
    final_merged_json = slice_for_final_merged_json(combined_facts_cb, False)

    ctx_lines = [
        "<<<RETRIEVED_CONTEXT_START>>>",
        "The system searched for a related question in the database. Below are related question's facts and prior statements as reference. You don't have to follow it completely, just use it as a reference.",
        f"{final_merged_json}",
        "<<<RETRIEVED_CONTEXT_END>>>",
    ]

    user_msg = "\n".join(ctx_lines) + "\n"
    user_msg += (
        f"[CURRENT QUESTION]: {question} \n"
        "[TASK]: You are a QA assistant for open-ended questions.\n"
        "- Give a short, direct answer in 2–3 sentences."
        "- Do NOT restrict to yes/no.\n"
        "[FORMAT]: Write complete sentences (not a single word). "
        "Avoid starting with just 'Yes.' or 'No.'; if the question is yes/no-style, "
        "state the conclusion AND 1–2 short reasons.\n"
        "[ANSWER]: "
    )
    return system_msg + "\n\n" + user_msg


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def to_sample(ex):
    return {
        "prompt": build_prompt(ex["question"], ex.get("evidence", [])),
        "completion": ex["answer"].rstrip() + tokenizer.eos_token
    }

formatted = [to_sample(x) for x in raw_data]
dataset  = Dataset.from_list(formatted).train_test_split(test_size=0.1, seed=42)
print(dataset)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, torch_dtype=dtype, trust_remote_code=True
).to(device)


peft_cfg = LoraConfig(
    r=8,                          
    lora_alpha=16,                 
    lora_dropout=0.1,              
    bias="none",                 
    task_type="CAUSAL_LM",         
    target_modules=[              
        "q_proj","k_proj","v_proj",
        "o_proj","up_proj","down_proj","gate_proj"
    ],
    modules_to_save=["lm_head"],  
)


model = get_peft_model(base_model, peft_cfg)

train_size = len(dataset["train"])
effective_bs = 1 * 16  # per_device_train_batch_size=1 × gradient_accumulation_steps=16
steps_per_epoch = train_size // effective_bs
print(f"[INFO] steps_per_epoch = {steps_per_epoch}")

# ========== 训练参数 (SFTConfig) ==========
sft_args = SFTConfig(
    # ========== 基础保存 ==========
    output_dir=OUTPUT_DIR,
    save_strategy="epoch",
    save_total_limit=5,  

    # ========== 训练超参 ==========
    per_device_train_batch_size=1,   # 每个设备上的 batch size
    gradient_accumulation_steps=16,  # 梯度累积 → 等效大 batch
    num_train_epochs=500,              # 训练轮数
    learning_rate=2e-4,              # 学习率

    # ========== 日志 & 监控 ==========
    logging_steps=1,                # 每 1 step 打印一次日志
    report_to=["tensorboard"],                # 可选: "tensorboard", "wandb", "none"
    logging_dir=LOG_DIR,             # 日志存放目录

    # ===== 验证评估 =====
    eval_strategy="epoch",        
    eval_steps=None,              
    
    # ========== 其他 ==========
    max_length=768,                  # 最大输入长度
    packing=False,                   # 不拼接多个样本
    padding_free=False,              # 关闭 padding-free 模式
    dataset_text_field="text",       # 数据字段名（实际用 formatting_func 拼接）
    completion_only_loss=None,       # 默认自动推断
    pad_token=tokenizer.pad_token,   # pad token
    eos_token=tokenizer.eos_token,   # 结束 token
    dataset_num_proc=1,              # 数据处理进程数
    seed=42,                         # 随机种子，保证复现
)

trainer = SFTTrainer(
    model=model,                 
    args=sft_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer, 
)


#trainer.train()
trainer.train(resume_from_checkpoint="lora_qwen_sft_adapter/checkpoint-900")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"[OK] LoRA adapter saved to: {OUTPUT_DIR}")