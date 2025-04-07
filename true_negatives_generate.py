"""
True Negative Sampling Dataset Generator

1. 計算input對應的interest_clusters
2. 使用 beam search 生成 Top-K Hard Negative candidates
3. 存起不是正確答案（output）的Negative candidates (True Negative)
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
import difflib

device = "cuda" if torch.cuda.is_available() else "cpu"
TEST = ""
GENERATE_MODEL = "_Origin"
# === Config ===
method_name = "True_Negative_Sampling_run2"
BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TUNED_MODEL = "smolLM2-1.7B-lora-run3"
FINETUNED_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{TUNED_MODEL}/final_model"
SAVE_PATH = f"/scratch/user/chuanhsin0110/test_0321/{TEST}output/{method_name}/data/"
NAME2GENRE_PATH = "/scratch/user/chuanhsin0110/SPRec/eval/Goodreads/name2genre.json"
USE_LORA = False  # 改成 False 就會用原始模型
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
    使用 Beam Search 生成候選，取出機率最大的前三個候選答案
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
    from collections import Counter

    # 統計每本書出現次數
    counter = Counter()
    for d in train_data:
        book_names = re.findall(r'"(.*?)"', d["input"])
        counter.update(book_names)

    # 排序：曝光次數從高到低
    sorted_books = counter.most_common()

    # 建立曝光排名字典
    exposure_rank = {book: rank + 1 for rank, (book, _) in enumerate(sorted_books)}

    return counter, exposure_rank


# === Main Process ===
def process_data(train_path, valid_path, model, tokenizer, train_size=1024, valid_size=128):
    with open(NAME2GENRE_PATH, "r") as f:
        name2genre = json.load(f)

    os.makedirs(SAVE_PATH, exist_ok=True)

    def load_and_sample(path, size):
        with open(path, "r") as f:
            data = json.load(f)
        # return random.sample(data, size)
        return data[:size]

    

    # train_data = load_and_sample(train_path, 1024)
    # # valid_data = load_and_sample(valid_path, 128)

    train_data = load_and_sample(train_path, train_size)
    valid_data = load_and_sample(valid_path, valid_size)
    

    exposure_count, exposure_rank = build_exposure_count(train_data)

    

    full_data = train_data + valid_data

    true_negative_data_details = []
    true_negative_data = []  # 用來存放 true negative 的 DPO 資料

    # 用來統計有幾次 beam search 生成的 Top-K 候選中含有正確答案（output）
    correct_in_beam_count = 0

    # === Batch Processing ===
    for i in tqdm(range(0, len(full_data), BATCH_SIZE), desc="Generating Negatives"):
        batch = full_data[i:i+BATCH_SIZE]
        prompts = [prepare_prompt(d["instruction"], d["input"]) for d in batch]
        interest_clusters = [get_user_interest_cluster(d["input"], name2genre) for d in batch]

        # === 使用 Beam Search 生成 True Negative 候選 ===
        beam_candidates = generate_candidates_beam_batch(model, tokenizer, prompts, num_return_sequences=10)
        #print(f"\n\n####  beam_candidates:{beam_candidates}")
        for idx, (d, prompt) in enumerate(zip(batch, prompts)):
            # print(f"\n\nprompt: {prompt}")
            # print(f"\ninterest_clusters: {list(interest_clusters[idx])}")
            true_negatives = []
            # 檢查 beam 候選中是否含有正確答案，若有則統計一次（每個樣本只計算一次）
            if any(cand.strip() == d["output"].strip() for cand in beam_candidates[idx]):
                correct_in_beam_count += 1
            print("\n\nbeam_candidates:")
            for cand in beam_candidates[idx]:
                print(cand)
                # 如果候選答案不等於正確答案，則加入 true_negative
                if cand.strip() != d["output"].strip():
                    true_negatives.append(cand)
            print("\ntrue_negative 候選的 genre:")
            # 計算每個 true_negative 候選的 genre
            neg_genra = []
            for cand in true_negatives:
                candidate_name = cand.strip("\" \n")
                genres = name2genre.get(candidate_name, None)
                exposure = exposure_count.get(candidate_name, 0)
                rank = exposure_rank.get(candidate_name, None)

                if genres is None:
                    # 找最相近的書名
                    closest = difflib.get_close_matches(candidate_name, name2genre.keys(), n=1, cutoff=0.6)
                    if closest:
                        matched_name = closest[0]
                        genres = name2genre[matched_name]
                        exposure = exposure_count.get(matched_name, 0)
                        rank = exposure_rank.get(matched_name, None)
                        print(f"Candidate not found: {candidate_name} → using closest match: {matched_name}")
                        neg_genra.append({
                            "candidate": f"**{cand.strip()} -> {matched_name}",
                            "genre": genres,
                            "exposure_count": exposure,
                            "exposure_rank": rank
                        })
                    else:
                        print(f"Candidate not found and no close match: {candidate_name}")
                        neg_genra.append({
                            "candidate": f"**{cand.strip()} -> NO_MATCH",
                            "genre": [],
                            "exposure_count": 0,
                            "exposure_rank": None
                        })
                else:
                    neg_genra.append({
                        "candidate": cand.strip(),
                        "genre": genres,
                        "exposure_count": exposure,
                        "exposure_rank": rank
                    })



            true_negative_data_details.append({
                "instruction": d["instruction"],
                "input": d["input"],
                "output": d["output"].strip(),
                "interest_clusters": list(interest_clusters[idx]),
                "true_negative": true_negatives,
                "neg_genra": neg_genra,
                "beam_candidates": beam_candidates[idx]
            })
            # 建立 true_negative_data 的項目
            true_negative_data.append({
                "prompt": prompt,
                "chosen": d["output"].strip(),
                "rejected": true_negatives
            })

    # === Save Files ===
    with open(os.path.join(SAVE_PATH, f"true_negative_data_details{GENERATE_MODEL}.json"), "w", encoding="utf-8") as f:
        json.dump(true_negative_data_details, f, indent=2, ensure_ascii=False)

    true_negative_data_train = true_negative_data[:train_size]
    true_negative_data_valid = true_negative_data[train_size:]

    with open(os.path.join(SAVE_PATH, "true_negative_data_train.json"), "w", encoding="utf-8") as f:
        json.dump(true_negative_data_train, f, indent=2, ensure_ascii=False)
    with open(os.path.join(SAVE_PATH, "true_negative_data_valid.json"), "w", encoding="utf-8") as f:
        json.dump(true_negative_data_valid, f, indent=2, ensure_ascii=False)

    print(f"\nAll datasets saved to {SAVE_PATH}")
    print(f"Beam search generated Top-K candidates containing correct answer: {correct_in_beam_count} times")

if __name__ == "__main__":

    # ============ Output dir check ============
    if os.path.exists(SAVE_PATH):
        print(f"Warning: Output dir '{SAVE_PATH}' already exists. It may overwrite previous models.")
        #exit(1)
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
        tokenizer=tokenizer,train_size=10,valid_size=5,
    )
