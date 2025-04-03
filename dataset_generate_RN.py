import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_scheduler
from datasets import load_dataset
from tqdm import tqdm
import random

# ========== Dataset ==========



def prepare_dpo_dataset(path, sample_size=1024, seed=42):
    raw_dataset = load_dataset("json", data_files=path)["train"]
    raw_dataset = raw_dataset.shuffle(seed=seed).select(range(min(sample_size, len(raw_dataset))))
    texts = [d["output"] for d in raw_dataset]

    dpo_samples = []
    for d in raw_dataset:
        prompt = f"### Instruction:\n{d['instruction']}\n"
        if d["input"].strip():
            prompt += f"### Input:\n{d['input']}\n"
        prompt += "### Response:"
        positive = d["output"].strip()
        negative = positive
        while negative == positive:
            negative = random.choice(texts)
        dpo_samples.append({
            "prompt": prompt,
            "chosen": positive,
            "rejected": negative
        })
    return dpo_samples

# ========== Loss ==========




if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True

    seed = 0
    train_sample_size = 1024
    valid_sample_size = 256
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    output_path = f"/scratch/user/chuanhsin0110/test_0321/output/random_sample/"
    train_data_path = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/train.json"
    valid_data_path = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/valid.json"



    # Prepare data
    train_data = prepare_dpo_dataset(train_data_path, sample_size=train_sample_size)
    valid_data = prepare_dpo_dataset(valid_data_path, sample_size=valid_sample_size)

    with open(os.path.join(output_path, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_path, "valid.json"), "w", encoding="utf-8") as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)

    print(f"\nDPO dataset saved to {output_path}")
