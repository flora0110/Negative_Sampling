import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import pandas as pd
from collections import Counter
import pandas as pd
import numpy as np
import json



def format_prompt(instruction, input_text):
    if input_text.strip():
        return f"""### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"""
    else:
        return f"""### Instruction:\n{instruction}\n\n### Response:"""

def generate_predictions(model, tokenizer, dataset, top_k=5, max_new_tokens=50):
    results = []
    for i, sample in enumerate(dataset):
        print(f"Generating prediction {i+1}/{len(dataset)}")
        prompt = format_prompt(sample["instruction"], sample["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_return_sequences=1)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded[len(prompt):].strip()
        results.append({
            "prompt": prompt,
            "prediction": response,
            "ground_truth": sample["output"].strip('" \n')
        })
    return results

def calculate_metrics(results, top_k=5):
    hits = 0
    ndcg = 0
    all_predictions = []

    for r in results:
        pred = r["prediction"]
        gt = r["ground_truth"]
        pred_list = [p.strip('" ') for p in pred.split(",")][:top_k]
        all_predictions.extend(pred_list)
        r["top_k_predictions"] = pred_list
        r["hit"] = int(gt in pred_list)

        if gt in pred_list:
            rank = pred_list.index(gt)
            r["ndcg"] = 1 / torch.log2(torch.tensor(rank + 2)).item()
            ndcg += r["ndcg"]
            hits += 1
        else:
            r["ndcg"] = 0

    print("all_predictions: ", all_predictions)
    hr = hits / len(results)
    ndcg_score = ndcg / len(results)

    # Diversity ratio
    unique_items = len(set(all_predictions))
    total_items = len(all_predictions)
    div_ratio = unique_items / total_items if total_items > 0 else 0

    # Over-recommendation ratio (top 10% items)
    item_counts = Counter(all_predictions)
    most_common = item_counts.most_common()
    top_10_percent_count = max(1, int(0.1 * len(most_common)))
    top_items = [item for item, _ in most_common[:top_10_percent_count]]
    or_count = sum(item_counts[item] for item in top_items)
    or_ratio = or_count / total_items if total_items > 0 else 0

    return hr, ndcg_score, div_ratio, or_ratio, results


def save_results_to_excel(results, hr, ndcg, div, orr, path="evaluation_results.xlsx"):
    df = pd.DataFrame(results)
    summary = pd.DataFrame({
        "Metric": ["HR@5", "NDCG@5", "DivRatio", "ORRatio"],
        "Value": [hr, ndcg, div, orr]
    })
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="Details", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
    return path

if __name__ == "__main__":
    print("model set")
    # 模型設定
    BASE_MODEL = "facebook/opt-350m"
    FINETUNED_PATH = "/scratch/user/chuanhsin0110/test_0321/output/test_run/final_model"
    checkpoint_path = "/scratch/user/chuanhsin0110/test_0321/output/test_run/checkpoint-256"
    TEST_PATH = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/test.json"
    USE_LORA = True  # 改成 False 就會用原始模型

    print("load tokenizer")
    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("load model")
    # 載入模型（基礎或微調後）
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    if USE_LORA:
        model = PeftModel.from_pretrained(model, FINETUNED_PATH)
        # model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    # 載入測試集
    dataset = load_dataset("json", data_files=TEST_PATH)["train"].select(range(10))
    
    # 推論
    raw_results = generate_predictions(model, tokenizer, dataset)
    print(raw_results[0])
    raw_results_filename = "eval_lora_raw_results.json" if USE_LORA else "eval_baseline_raw_results.json"

    with open(raw_results_filename, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    # 評估指標
    hr, ndcg, div, orr, final_results = calculate_metrics(raw_results)

    # 顯示結果
    print(f"\n📊 Evaluation Results:")
    print(f"HR@5     = {hr:.4f}")
    print(f"NDCG@5   = {ndcg:.4f}")
    print(f"DivRatio = {div:.4f}")
    print(f"ORRatio  = {orr:.4f}")

    # 儲存結果
    filename = "eval_lora.xlsx" if USE_LORA else "eval_baseline.xlsx"
    save_results_to_excel(final_results, hr, ndcg, div, orr, filename)
    print(f"\n結果已儲存至 {filename}")
