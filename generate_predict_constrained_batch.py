import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import json
import os
from tqdm import tqdm

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
        return list(node.children.keys()) + ([tokenizer.eos_token_id] if node.is_end else [])

def format_prompt(instruction, input_text):
    if input_text.strip():
        return f"""### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"""
    else:
        return f"""### Instruction:\n{instruction}\n\n### Response:"""

def find_response_start(input_ids, prompt_end_ids):
    """
    在 input_ids（list形式）中搜尋 prompt_end_ids 子序列，
    回傳該子序列結束後的位置作為 response start。
    """
    n = len(prompt_end_ids)
    # 從頭到尾遍歷，找到第一個匹配位置
    for i in range(len(input_ids) - n + 1):
        if input_ids[i:i+n] == prompt_end_ids:
            return i + n  # 結束位置
    return None

def generate_predictions_batch(model, tokenizer, dataset, trie, batch_size=8, max_new_tokens=50):
    results = []
    model.eval()

    prompt_end_text = "### Response:"
    prompt_end_ids = tokenizer.encode(prompt_end_text, add_special_tokens=False)

    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Predictions"):
        batch_global_indices = list(range(i, min(i + batch_size, len(dataset))))
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        prompts = [format_prompt(sample["instruction"], sample["input"]) for sample in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]

        # response_starts = [tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1] for prompt in prompts]
        # response_starts = torch.sum(inputs["attention_mask"], dim=1).tolist()


        TRIE_START_OFFSET = 2  # 提早幾 token 開始讓模型 warm up

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            # 將 input_ids 轉換為 list 以便搜尋
            input_ids_list = input_ids.tolist()
            response_start = find_response_start(input_ids_list, prompt_end_ids)
            if response_start is None:
                # 如果找不到，就退回到用非填充 token 數量作為回退
                response_start = (input_ids != tokenizer.pad_token_id).sum().item()
                #print(f"⚠️ [Batch {batch_id}] 未找到 prompt 結尾，使用 fallback response_start = {response_start}")
            # else:
            #     print(f"🔍 [Batch {batch_id}] 找到 prompt 結尾，response_start = {response_start}")

            # 如果尚未超過 response_start + TRIE_START_OFFSET，就返回全部 token 集合
            if input_ids.shape[-1] <= response_start + TRIE_START_OFFSET:
                if input_ids.shape[-1] > response_start:
                    decoded_prefix = tokenizer.decode(input_ids[-TRIE_START_OFFSET:])
                    # print(f"okk [Batch {batch_id}] Prefix so far (last tokens): ...{repr(decoded_prefix)}")
                return list(range(tokenizer.vocab_size))
            
            # 從 response_start + TRIE_START_OFFSET 開始取出生成的部分
            response_only_prefix = input_ids[response_start + TRIE_START_OFFSET:]
            allowed = trie.get_allowed_tokens(response_only_prefix.tolist())
    
            decoded_prefix = tokenizer.decode(response_only_prefix)
            # print(f"🔹 [Batch {batch_id}] Prefix so far (last tokens): ...{repr(decoded_prefix)}")
            
            if not allowed:
                last_token_id = input_ids[-1].item()
                # print("⚠️ 空限制！最後一個 token:", last_token_id, "→", repr(tokenizer.decode([last_token_id])))
            
            return allowed if allowed else [tokenizer.eos_token_id]




        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
            )

        for j, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            response = decoded[len(prompts[j]):].strip()
            results.append({
                "prompt": prompts[j],
                "prediction": response,
                "ground_truth": batch[j]["output"].strip('\" \n')
            })

    return results


if __name__ == "__main__":
    BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    TUNED_MODEL = "SPRec_on_SFT-1"
    method_name = "Constrained_Predict_Generate/SPRec"
    FINETUNED_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{TUNED_MODEL}/final_model"
    TEST_PATH = "/scratch/user/chuanhsin0110/test_0321/sampled_data/test_sample.json"
    output_dir = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}"
    USE_LORA = True
    test_sample_size = 10
    batch_size = 8

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    if USE_LORA:
        model = PeftModel.from_pretrained(model, FINETUNED_PATH)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    dataset = load_dataset("json", data_files=TEST_PATH)["train"]
    dataset = dataset.select(range(test_sample_size))


    # === Load product list from name2genre.json and build Trie ===
    with open("/scratch/user/chuanhsin0110/SPRec/eval/Goodreads/name2genre.json", "r") as f:
        name2genre = json.load(f)
    trie = Trie()
    for name in name2genre.keys():
        token_ids = tokenizer.encode(name, add_special_tokens=False)
        trie.insert(token_ids)

    raw_results = generate_predictions_batch(model.module, tokenizer, dataset, trie, batch_size=batch_size)
    print(raw_results[0])

    predict_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predict_dir, exist_ok=True)
    raw_results_filename = f"raw_results_dpo_{test_sample_size}.json" if USE_LORA else "raw_results_baseline.json"
    raw_results_path = os.path.join(predict_dir, raw_results_filename)

    with open(raw_results_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    print(f"\nInference completed. Results saved to {raw_results_path}")
