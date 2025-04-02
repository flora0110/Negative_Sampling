"""
Clustering + Exposure Balanced Sampling Dataset Generator

1. 使用 Beam Search 生成 Top-K Hard Negative -> 下一個，用very hard neg再做(size<8)，先用topk
2. Long-tail Negative 根據曝光度排序
3. 若找不到 Hard Negative，再生一次
4. 最後儲存三種 DPO 資料格式：
    - dpo_hard.json
    - dpo_long_tail.json
    - dpo_two_negatives.json -> 做 S-DPO
"""

import re
import json
import os
import random
from tqdm import tqdm
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Config ===
method_name = "Clustering-Exposure_Balanced_Sampling_run1"
BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TUNED_MODEL = "smolLM2-1.7B-lora-run3"
FINETUNED_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{TUNED_MODEL}/final_model"
SAVE_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/data/"
NAME2GENRE_PATH = "/scratch/user/chuanhsin0110/SPRec/eval/Goodreads/name2genre.json"
USE_LORA = True  # 改成 False 就會用原始模型

# === Utils ===r
def prepare_prompt(instruction, input_text):
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text}\n"
    prompt += "### Response:"
    return prompt

def get_user_interest_cluster(input_text, name2genre, topk=3):
    genre_count = {}
    book_names = re.findall(r'"(.*?)"', input_text)
    for name in book_names:
        genres = name2genre.get(name, [])
        for g in genres:
            genre_count[g] = genre_count.get(g, 0) + 1
    top_genres = sorted(genre_count, key=genre_count.get, reverse=True)[:topk]
    return set(top_genres)

def generate_candidates_topk(model, tokenizer, prompt, max_new_tokens=100, num_return_sequences=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True, 
            #num_beams=num_return_sequences,  # Beam Search
            top_k=50,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )
    candidates = []
    for output in outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        response = decoded[len(prompt):].strip()
        match = re.search(r'"([^"]*)', response)
        if match:
            candidates.append(f"\"{match.group(1)}\"\n")
    return candidates

# def generate_candidates_beam(model, tokenizer, prompt, max_new_tokens=100, num_return_sequences=10):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False, 
#             num_beams=num_return_sequences,  # Beam Search
#             # top_k=50,
#             num_return_sequences=num_return_sequences,
#             pad_token_id=tokenizer.eos_token_id
#         )
#     candidates = []
#     for output in outputs:
#         decoded = tokenizer.decode(output, skip_special_tokens=True)
#         response = decoded[len(prompt):].strip()
#         match = re.search(r'"([^"]*)', response)
#         if match:
#             candidates.append(f"\"{match.group(1)}\"\n")
#     return candidates

def build_exposure_count(train_data):
    counter = Counter()
    for d in train_data:
        book_names = re.findall(r'"(.*?)"', d["input"])
        counter.update(book_names)
    return counter

# === Main Process ===
def process_data(train_path, valid_path, model, tokenizer, train_size=1024, valid_size=128):
    # tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto").eval()

    with open(NAME2GENRE_PATH, "r") as f:
        name2genre = json.load(f)

    os.makedirs(SAVE_PATH, exist_ok=True)

    def load_and_sample(path, size):
        with open(path, "r") as f:
            data = json.load(f)
        return random.sample(data, size)

    train_data = load_and_sample(train_path, train_size)
    valid_data = load_and_sample(valid_path, valid_size)
    full_data = train_data + valid_data

    exposure_count = build_exposure_count(train_data)
    balanced_data = []
    dpo_hard, dpo_long, dpo_two = [], [], []

    for d in tqdm(full_data, desc="Generating negatives"):
        prompt = prepare_prompt(d["instruction"], d["input"])
        interest_cluster = get_user_interest_cluster(d["input"], name2genre)

        num_seq = 10
        hard_neg = None
        candidates = []
        rn_count = 0

        while not hard_neg:
            new_candidates = generate_candidates_topk(model, tokenizer, prompt, num_return_sequences=num_seq)
            candidates.extend(new_candidates)

            for c in new_candidates:
                genres = name2genre.get(c.strip("\" \n"), [])
                if any(g in interest_cluster for g in genres):
                    hard_neg = c
                    break

            if not hard_neg:
                # 沒找到 → 再跑一次 (num_seq 不變)
                continue

            if len(candidates) >= 40:  # 防止無限 loop
                rn_count = rn_count + 1
                hard_neg = random.choice(candidates)
                break

        print(f"rn_count: {rn_count}\n")
        # Long-tail negative
        long_tail_candidates = [c for c in candidates if not any(g in interest_cluster for g in name2genre.get(c.strip("\" \n"), []))]
        if long_tail_candidates:
            long_tail_candidates.sort(key=lambda x: exposure_count.get(x.strip("\" \n"), 0))
            long_tail_neg = long_tail_candidates[0]
        else:
            long_tail_neg = random.choice(candidates)

        balanced_data.append({
            "instruction": d["instruction"],
            "input": d["input"],
            "output": d["output"].strip(),
            "hard_negatives": hard_neg,
            "long_tail_negatives": long_tail_neg
        })

        dpo_hard.append({
            "prompt": prompt,
            "chosen": d["output"].strip(),
            "rejected": hard_neg
        })
        dpo_long.append({
            "prompt": prompt,
            "chosen": d["output"].strip(),
            "rejected": long_tail_neg
        })
        dpo_two.append({
            "prompt": prompt,
            "chosen": d["output"].strip(),
            "rejected": [hard_neg, long_tail_neg]
        })

    

    # === Save Files ===
    with open(os.path.join(SAVE_PATH, "balanced_data.json"), "w") as f:
        json.dump(balanced_data, f, indent=2)

    # === Split & Save Balanced Data ===
    balanced_train = balanced_data[:train_size]
    balanced_valid = balanced_data[train_size:]

    with open(os.path.join(SAVE_PATH, "balanced_train.json"), "w", encoding="utf-8") as f:
        json.dump(balanced_train, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SAVE_PATH, "balanced_valid.json"), "w", encoding="utf-8") as f:
        json.dump(balanced_valid, f, indent=2, ensure_ascii=False)

    print(f"\nBalanced Data Train/Valid split saved to {SAVE_PATH}")

    for name, data in zip(["dpo_hard.json", "dpo_long_tail.json", "dpo_two_negatives.json"],
                          [dpo_hard, dpo_long, dpo_two]):
        with open(os.path.join(SAVE_PATH, name), "w") as f:
            for item in data:
                json.dump(item, f)
                f.write("\n")

    print(f"\nBalanced DPO datasets saved to {SAVE_PATH}")


    # === Split & Save DPO Data ===
    dpo_hard_train = dpo_hard[:train_size]
    dpo_hard_valid = dpo_hard[train_size:]

    dpo_long_train = dpo_long[:train_size]
    dpo_long_valid = dpo_long[train_size:]

    dpo_two_train = dpo_two[:train_size]
    dpo_two_valid = dpo_two[train_size:]

    with open(os.path.join(SAVE_PATH, "dpo_hard_train.json"), "w", encoding="utf-8") as f:
        json.dump(dpo_hard_train, f, indent=2, ensure_ascii=False)
    with open(os.path.join(SAVE_PATH, "dpo_hard_valid.json"), "w", encoding="utf-8") as f:
        json.dump(dpo_hard_valid, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SAVE_PATH, "dpo_long_tail_train.json"), "w", encoding="utf-8") as f:
        json.dump(dpo_long_train, f, indent=2, ensure_ascii=False)
    with open(os.path.join(SAVE_PATH, "dpo_long_tail_valid.json"), "w", encoding="utf-8") as f:
        json.dump(dpo_long_valid, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SAVE_PATH, "dpo_two_negatives_train.json"), "w", encoding="utf-8") as f:
        json.dump(dpo_two_train, f, indent=2, ensure_ascii=False)
    with open(os.path.join(SAVE_PATH, "dpo_two_negatives_valid.json"), "w", encoding="utf-8") as f:
        json.dump(dpo_two_valid, f, indent=2, ensure_ascii=False)

    print(f"\nDPO Train/Valid splits saved to {SAVE_PATH}")



if __name__ == "__main__":

     # ============ Output dir check ============
    if os.path.exists(SAVE_PATH):
        print(f"Warning: Output dir '{SAVE_PATH}' already exists. It may overwrite previous models.")
        exit(1)
    else:
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"Created output dir: {SAVE_PATH}")
    # ===========================================
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    if torch.cuda.device_count() > 1:
        print("enable parallel\n")
        model = torch.nn.DataParallel(model) 
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if USE_LORA:
        model = PeftModel.from_pretrained(model, FINETUNED_PATH)

    process_data(
        train_path="/scratch/user/chuanhsin0110/test_0321/sampled_data/train_sample.json",
        valid_path="/scratch/user/chuanhsin0110/test_0321/sampled_data/valid_sample.json",
        model=model,
        tokenizer=tokenizer,
    )
