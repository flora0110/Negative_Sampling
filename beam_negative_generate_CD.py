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

# === Config ===
method_name = "Beam_Search_Negative_Generate_CD"
BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TUNED_MODEL = "smolLM2-1.7B-lora-run3"
Test = "test_data/"
FINETUNED_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{TUNED_MODEL}/final_model"
SAVE_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/{Test}data/"
NAME2GENRE_PATH = "/scratch/user/chuanhsin0110/SPRec/eval/Goodreads/name2genre.json"
USE_LORA = True  # 改成 False 就會用原始模型
Diverse_Beam_Search = True

BATCH_SIZE = 8
train_size = 10
valid_size = 12
num_return_sequences = 10
diversity_penalty = 0.5


# === Utils ===
def format_prompt(instruction, input_text):
    # 將 instruction 與 input 組成標準 prompt 格式
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

def filter_rejected_candidates(candidates, correct_answer, input_text):
    history_books = set(re.findall(r'"(.*?)"', input_text))
    filtered = set()
    for cand in candidates:
        cleaned = cand.strip()
        if cleaned and cleaned != correct_answer.strip() and cleaned != "\"\"":
            cand_name = cleaned.strip('"')
            if cand_name not in history_books:
                filtered.add(cleaned)
    return list(filtered)

def generate_candidates_beam_batch(model, tokenizer, prompts, trie=None, max_new_tokens=100, num_return_sequences=3):
    """
    使用 Beam Search (可選 Diverse Beam Search + Constrained Decoding)
    生成候選答案，並從中提取合法書名。
    
    若提供 trie，則執行 Constrained Decoding。
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    batch_size = len(prompts)

    prompt_end_text = "### Response:"
    prompt_end_ids = tokenizer.encode(prompt_end_text, add_special_tokens=False)

    # === prefix_allowed_tokens_fn ===
    def find_response_start(input_ids, prompt_end_ids):
        for i in range(len(input_ids) - len(prompt_end_ids) + 1):
            if input_ids[i:i + len(prompt_end_ids)] == prompt_end_ids:
                return i + len(prompt_end_ids)
        return None

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        input_ids_list = input_ids.tolist()
        response_start = find_response_start(input_ids_list, prompt_end_ids)
        if response_start is None:
            response_start = (input_ids != tokenizer.pad_token_id).sum().item()

        TRIE_START_OFFSET = 2
        if input_ids.shape[-1] <= response_start + TRIE_START_OFFSET:
            return list(range(tokenizer.vocab_size))

        response_only_prefix = input_ids[response_start + TRIE_START_OFFSET:]
        allowed = trie.get_allowed_tokens(response_only_prefix.tolist()) if trie else None
        return allowed if allowed else [tokenizer.eos_token_id]

    # === generate ===
    with torch.no_grad():
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )
        if Diverse_Beam_Search:
            gen_kwargs.update({
                "num_beams": num_return_sequences * 2,
                "num_beam_groups": num_return_sequences,
                "diversity_penalty": diversity_penalty
            })
        else:
            gen_kwargs.update({
                "num_beams": num_return_sequences
            })
        if trie:
            # print("CD on\n")
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn

        outputs = model.generate(**gen_kwargs)

    # === decode ===
    all_candidates = []
    for i in range(batch_size):
        prompt = prompts[i]
        candidates = []
        for j in range(num_return_sequences):
            output = outputs[i * num_return_sequences + j]
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            response = decoded[len(prompt):].strip()
            match = re.search(r'"([^"\n]*)', response)
            if match:
                candidate = f'"{match.group(1)}"\n'
                candidates.append(candidate)
        all_candidates.append(candidates)
    return all_candidates


def build_exposure_count(train_data):
    # 統計每本書在 train_data 中出現次數
    counter = Counter()
    for d in train_data:
        book_names = re.findall(r'"(.*?)"', d["input"])
        counter.update(book_names)
    # 依曝光次數從高到低排序，並建立排名字典
    sorted_books = counter.most_common()
    exposure_rank = {book: rank + 1 for rank, (book, _) in enumerate(sorted_books)}
    return counter, exposure_rank

# === Simple Trie for Constrained Decoding ===
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert(self, token_ids):
        node = self
        for token in token_ids:
            if token not in node.children:
                node.children[token] = Trie()
            node = node.children[token]
        node.is_end = True

    def get_allowed_tokens(self, prefix):
        node = self
        for token in prefix:
            if token in node.children:
                node = node.children[token]
            else:
                return []
        # 若此處為結束，允許 eos token
        return list(node.children.keys()) + ([tokenizer.eos_token_id] if node.is_end else [])

# === Main Process ===
def process_data(train_path, valid_path, model, tokenizer, train_size=1024, valid_size=128):
    with open(NAME2GENRE_PATH, "r") as f:
        name2genre = json.load(f)

    os.makedirs(SAVE_PATH, exist_ok=True)

    def load_and_sample(path, size):
        with open(path, "r") as f:
            data = json.load(f)
        return data[:size]

    train_data = load_and_sample(train_path, train_size)
    valid_data = load_and_sample(valid_path, valid_size)
    exposure_count, exposure_rank = build_exposure_count(train_data)
    full_data = train_data + valid_data

    # 建立合法書名的 Trie (使用 name2genre 的 key 作為合法書名)
    valid_books = list(name2genre.keys())
    trie = Trie()
    for name in valid_books:
        token_ids = tokenizer.encode(name, add_special_tokens=False)
        trie.insert(token_ids)

    # 用來統計有幾次 beam search 生成的候選中含有正確答案
    correct_in_beam_count = 0

    # 用於儲存最終輸出格式 (要求輸出格式為一個包含字典的列表，並嵌套成一個列表)
    all_results = []

    # === Batch Processing ===
    for i in tqdm(range(0, len(full_data), BATCH_SIZE), desc="Generating Negatives"):
        batch = full_data[i:i+BATCH_SIZE]
        # 記錄全域索引（在資料集中的索引）方便 debug（如有需要可加印）\n        batch_global_indices = list(range(i, min(i + BATCH_SIZE, len(full_data))))  # 可用於 debug
        prompts = [format_prompt(d["instruction"], d["input"]) for d in batch]
        interest_clusters = [get_user_interest_cluster(d["input"], name2genre) for d in batch]

        # 使用 generate_candidates_beam_batch 生成候選序列 (候選書名)
        beam_candidates = generate_candidates_beam_batch(model, tokenizer, prompts, trie = trie, num_return_sequences=num_return_sequences)

        # 對於每筆資料，建立輸出格式
        for idx, (d, prompt) in enumerate(zip(batch, prompts)):
            # 統計正確答案出現次數 (只計算每筆資料一次)
            if any(cand.strip() == d["output"].strip() for cand in beam_candidates[idx]):
                correct_in_beam_count += 1

            rejected_list = filter_rejected_candidates(beam_candidates[idx], d["output"], d["input"])


            # # 將候選中不等於正確答案的，並且去除重複與空字串後，作為 rejected 值
            # rejected_set = set()
            # for cand in beam_candidates[idx]:
            #     candidate_clean = cand.strip()
            #     if candidate_clean and candidate_clean != d["output"].strip() and candidate_clean != "\"\"":
            #         rejected_set.add(candidate_clean)
            # rejected_list = list(rejected_set)

            # 建立這筆資料的輸出結構
            result_dict = {
                "prompt": prompt,
                "chosen": d["output"].strip(),
                "rejected": rejected_list
            }
            all_results.append(result_dict)

    # 將最終結果放入一個嵌套列表 (根據要求)
    final_results = [all_results]

    # === Save Files ===
    # 將全資料拆分成 train 與 valid 兩部分 (按順序拆分)
    train_results = final_results[0][:train_size]
    valid_results = final_results[0][train_size:train_size+valid_size]

    D = ""
    if Diverse_Beam_Search:
        D = "Div"
    with open(os.path.join(SAVE_PATH, f"train_{D}_{num_return_sequences}_{diversity_penalty}.json"), "w", encoding="utf-8") as f:
        json.dump(train_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(SAVE_PATH, f"valid_{D}_{num_return_sequences}_{diversity_penalty}.json"), "w", encoding="utf-8") as f:
        json.dump(valid_results, f, indent=2, ensure_ascii=False)

    print(f"\nAll datasets saved to {SAVE_PATH}")
    print(f"Beam search generated Top-K candidates containing correct answer: {correct_in_beam_count} times")

if __name__ == "__main__":
    


    # === Output dir check ===
    if os.path.exists(SAVE_PATH):
        print(f"Warning: Output dir '{SAVE_PATH}' already exists. It may overwrite previous models.")
    else:
        os.makedirs(SAVE_PATH, exist_ok=True)
        print(f"Created output dir: {SAVE_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    if USE_LORA:
        model = PeftModel.from_pretrained(model, FINETUNED_PATH)
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs for inference...")
        model = torch.nn.DataParallel(model)
    # dataset = None
    # try:
    #     from datasets import load_dataset
    #     dataset = load_dataset("json", data_files=TEST_PATH)["train"]
    # except Exception as e:
    #     print("Error loading dataset:", e)
    #     exit(1)
    # dataset = dataset.select(range(train_size + valid_size))
    process_data(
        train_path="/scratch/user/chuanhsin0110/test_0321/sampled_data/train_sample.json",
        valid_path="/scratch/user/chuanhsin0110/test_0321/sampled_data/valid_sample.json",
        model=model.module,
        tokenizer=tokenizer,
        train_size=train_size,
        valid_size=valid_size,
    )
