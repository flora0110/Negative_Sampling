import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import pandas as pd
from collections import Counter
import pandas as pd
import numpy as np
import json
import os


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

if __name__ == "__main__":
    print("model set")
    # 模型設定
    # BASE_MODEL = "facebook/opt-350m"
    BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    method_name = "Clustering-Exposure_Balanced_Sampling_run1"
    # model_name = "smolLM2-1.7B-lora-dpo-run6"
    sample_method = "long_tail"

    FINETUNED_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/{sample_method}/final_model"
    # FINETUNED_PATH = "/scratch/user/chuanhsin0110/test_0321/output/test_run/final_model"
    # checkpoint_path = "/scratch/user/chuanhsin0110/test_0321/output/test_run/checkpoint-256"
    TEST_PATH = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/test.json"
    output_dir = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/{sample_method}"
    USE_LORA = True  # 改成 False 就會用原始模型
    test_sample_size = 1000

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
    dataset = load_dataset("json", data_files=TEST_PATH)["train"].select(range(test_sample_size))
    
    # 推論
    raw_results = generate_predictions(model, tokenizer, dataset)
    print(raw_results[0])
    # raw_results_filename = "eval_lora_raw_results.json" if USE_LORA else "eval_baseline_raw_results.json"

    # with open(raw_results_filename, "w", encoding="utf-8") as f:
    #     json.dump(raw_results, f, indent=2, ensure_ascii=False)

    # 建立 predictions/ 資料夾（如果還沒有的話）
    predict_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predict_dir, exist_ok=True)

    # 檔名可讀性提升
    raw_results_filename = f"raw_results_dpo_{test_sample_size}.json" if USE_LORA else "raw_results_baseline.json"
    raw_results_path = os.path.join(predict_dir, raw_results_filename)

    # 存檔
    with open(raw_results_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    print(f"推論結果已儲存到：{raw_results_path}")

    # # 評估指標
    # hr, ndcg, div, orr, final_results = calculate_metrics(raw_results)

    # # 顯示結果
    # print(f"\n📊 Evaluation Results:")
    # print(f"HR@5     = {hr:.4f}")
    # print(f"NDCG@5   = {ndcg:.4f}")
    # print(f"DivRatio = {div:.4f}")
    # print(f"ORRatio  = {orr:.4f}")

    # # 儲存結果
    # filename = "eval_lora.xlsx" if USE_LORA else "eval_baseline.xlsx"
    # save_results_to_excel(final_results, hr, ndcg, div, orr, filename)
    # print(f"\n結果已儲存至 {filename}")
