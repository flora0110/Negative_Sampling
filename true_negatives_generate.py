"""
Clustering + Exposure Balanced Sampling Dataset Generator

1. 使用 topk 生成 Top-K Hard Negative candidates , 找同個cluster的
2. Long-tail Negative 根據曝光度排序
3. 若找不到 Hard Negative，再生一次
4. 最後儲存四種 DPO 資料格式：
    - dpo_hard.json
    - dpo_long_tail.json
    - dpo_two_negatives.json -> 做 S-DPO
    - true_negative.json -> 新增 true negative 資料
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
method_name = "True_Negative_Sampling_run1"
BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TUNED_MODEL = "smolLM2-1.7B-lora-run3"
FINETUNED_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{TUNED_MODEL}/final_model"
SAVE_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/data/"
NAME2GENRE_PATH = "/scratch/user/chuanhsin0110/SPRec/eval/Goodreads/name2genre.json"
USE_LORA = True  # 改成 False 就會用原始模型
BATCH_SIZE = 8

# === Utils ===
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

def generate_candidates_beam_batch(model, tokenizer, prompts, max_new_tokens=100, num_return_sequences=3):
    """
    修改後的 generate_candidates_topk_batch 使用 Beam Search
    取出機率最大的前三個候選答案
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    batch_size = len(prompts)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 關閉隨機采樣，使用 Beam Search
            num_beams=num_return_sequences,  # 設定 beam 數量
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )
    all_candidates = []
    for i in range(batch_size):
        prompt = prompts[i]
        candidates = []
        for j in range(num_return_sequences):
            output = outputs[i * num_return_sequences + j]
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            response = decoded[len(prompt):].strip()
            match = re.search(r'"([^"]*)', response)
            if match:
                candidates.append(f"\"{match.group(1)}\"\n")
        all_candidates.append(candidates)
    return all_candidates

def build_exposure_count(train_data):
    counter = Counter()
    for d in train_data:
        book_names = re.findall(r'"(.*?)"', d["input"])
        counter.update(book_names)
    return counter

# === Main Process ===
def process_data(train_path, valid_path, model, tokenizer, train_size=1024, valid_size=128):
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

    true_negative_data_details = []
    # dpo_hard, dpo_long, dpo_two = [], [], []
    true_negative_data = []  # 用來存放 true negative 的 DPO 資料

    # === Batch Processing ===
    for i in tqdm(range(0, len(full_data), BATCH_SIZE), desc="Generating Negatives"):
        batch = full_data[i:i+BATCH_SIZE]
        prompts = [prepare_prompt(d["instruction"], d["input"]) for d in batch]
        interest_clusters = [get_user_interest_cluster(d["input"], name2genre) for d in batch]

        # === Dynamic Sampling for Hard & Long-tail Negatives ===
        batch_candidates = [[] for _ in range(len(batch))]
        # hard_negs = [None] * len(batch)
        # retry_counts = [0] * len(batch)
        num_seq = 3  # 這裡改成 3 (beam search候選數)

        # # 使用 beam search 生成候選，直到每個樣本都有 hard negative
        # while any(h is None for h in hard_negs):
        #     new_candidates = generate_candidates_beam_batch(model, tokenizer, prompts, num_return_sequences=num_seq)
        #     for idx, candidates in enumerate(new_candidates):
        #         batch_candidates[idx].extend(candidates)
        #         if hard_negs[idx] is None:
        #             for c in candidates:
        #                 genres = name2genre.get(c.strip("\" \n"), [])
        #                 if any(g in interest_clusters[idx] for g in genres):
        #                     hard_negs[idx] = c
        #                     break
        #             if hard_negs[idx] is None and len(batch_candidates[idx]) >= 40:
        #                 hard_negs[idx] = random.choice(batch_candidates[idx])
        #                 retry_counts[idx] += 1

        # === 選擇 Long-tail Negative 以及組建 balanced_data 與 DPO 資料 ===
        # for d, prompt, interest_cluster, candidates, hard_neg, rn_count in zip(batch, prompts, interest_clusters, batch_candidates, hard_negs, retry_counts):
        #     long_tail_candidates = [c for c in candidates if not any(g in interest_cluster for g in name2genre.get(c.strip("\" \n"), []))]
        #     if long_tail_candidates:
        #         long_tail_candidates.sort(key=lambda x: exposure_count.get(x.strip("\" \n"), 0))
        #         long_tail_neg = long_tail_candidates[0]
        #     else:
        #         long_tail_neg = random.choice(candidates)

        #     # 先建立 balanced_data 的基本資料，不含 true_negative
        #     balanced_data.append({
        #         "instruction": d["instruction"],
        #         "input": d["input"],
        #         "output": d["output"].strip(),
        #         "hard_negatives": hard_neg,
        #         "long_tail_negatives": long_tail_neg
        #     })

        #     dpo_hard.append({
        #         "prompt": prompt,
        #         "chosen": d["output"].strip(),
        #         "rejected": hard_neg
        #     })
        #     dpo_long.append({
        #         "prompt": prompt,
        #         "chosen": d["output"].strip(),
        #         "rejected": long_tail_neg
        #     })
        #     dpo_two.append({
        #         "prompt": prompt,
        #         "chosen": d["output"].strip(),
        #         "rejected": [hard_neg, long_tail_neg]
        #     })

        # === 使用 Beam Search 生成 True Negative 候選 ===
        # 對整個 batch 的 prompt 再跑一次 beam search，取出前三個候選
        beam_candidates = generate_candidates_topk_batch(model, tokenizer, prompts, num_return_sequences=3)
        for idx, (d, prompt) in enumerate(zip(batch, prompts)):
            true_negatives = []
            for cand in beam_candidates[idx]:
                # 如果候選答案不等於正確答案，則加入 true_negative
                if cand.strip() != d["output"].strip():
                    true_negatives.append(cand)
            # 更新剛剛加入 balanced_data 中的對應資料 (全局索引為 i + idx)
            # balanced_data[i + idx]["true_negative"] = true_negatives
            true_negative_data_details.append({
                "instruction": d["instruction"],
                "input": d["input"],
                "output": d["output"].strip(),
                "interest_clusters": interest_clusters,
                "true_negative": true_negatives,
                "neg_genra": , # 每個true_negative對應的genra
            })
            # 建立 true_negative_data 的項目
            true_negative_data.append({
                "prompt": prompt,
                "chosen": d["output"].strip(),
                "rejected": true_negatives
            })

    # === Save Files ===
    with open(os.path.join(SAVE_PATH, "true_negative_data_details.json"), "w") as f:
        json.dump(true_negative_data_details, f, indent=2)

    # balanced_train = balanced_data[:train_size]
    # balanced_valid = balanced_data[train_size:]
    # with open(os.path.join(SAVE_PATH, "balanced_train.json"), "w", encoding="utf-8") as f:
    #     json.dump(balanced_train, f, indent=2, ensure_ascii=False)
    # with open(os.path.join(SAVE_PATH, "balanced_valid.json"), "w", encoding="utf-8") as f:
    #     json.dump(balanced_valid, f, indent=2, ensure_ascii=False)

    print(f"\nBalanced Data Train/Valid split saved to {SAVE_PATH}")

    # DPO Data
    # Split DPO Data
    true_negative_data_train = true_negative_data[:train_size]
    true_negative_data_valid = true_negative_data[train_size:]


    with open(os.path.join(SAVE_PATH, "true_negative_data_train.json"), "w", encoding="utf-8") as f:
        json.dump(true_negative_data_train, f, indent=2, ensure_ascii=False)
    with open(os.path.join(SAVE_PATH, "true_negative_data_valid.json"), "w", encoding="utf-8") as f:
        json.dump(true_negative_data_valid, f, indent=2, ensure_ascii=False)

    print(f"\nAll datasets saved to {SAVE_PATH}")

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

    if USE_LORA:
        model = PeftModel.from_pretrained(model, FINETUNED_PATH)

    if torch.cuda.device_count() > 1:
        print("enable parallel\n")
        model = torch.nn.DataParallel(model)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")

    process_data(
        train_path="/scratch/user/chuanhsin0110/test_0321/sampled_data/train_sample.json",
        valid_path="/scratch/user/chuanhsin0110/test_0321/sampled_data/valid_sample.json",
        model=model.module,
        tokenizer=tokenizer,
    )
