import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import json
import os
from tqdm import tqdm

def format_prompt(instruction, input_text):
    if input_text.strip():
        return f"""### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"""
    else:
        return f"""### Instruction:\n{instruction}\n\n### Response:"""

def generate_predictions_batch(model, tokenizer, dataset, batch_size=8, max_new_tokens=50):
    results = []
    model.eval()
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Predictions"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        prompts = [format_prompt(sample["instruction"], sample["input"]) for sample in batch]

        # batch = dataset[i:i+batch_size]
        # prompts = [format_prompt(sample["instruction"], sample["input"]) for sample in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        for j, output in enumerate(outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            response = decoded[len(prompts[j]):].strip()
            results.append({
                "prompt": prompts[j],
                "prediction": response,
                "ground_truth": batch[j]["output"].strip('" \n')
            })
    return results

if __name__ == "__main__":
    # === Config ===
    BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    method_name = "SPRec_wo_SFT_DPO_on_SFT-1"
    sample_method = "long_tail-2"
    # FINETUNED_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/{sample_method}/final_model"
    FINETUNED_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/final_model"
    TEST_PATH = "/scratch/user/chuanhsin0110/test_0321/sampled_data/test_sample.json"
    # output_dir = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/{sample_method}"
    output_dir = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}"
    USE_LORA = True
    test_sample_size = 1000
    batch_size = 8

    # === Load Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # === Load Model ===
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
    if USE_LORA:
        model = PeftModel.from_pretrained(model, FINETUNED_PATH)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for inference...")
        model = torch.nn.DataParallel(model)
    # model = model.to("cuda")

    # === Load Dataset ===
    dataset = load_dataset("json", data_files=TEST_PATH)["train"]

    # === Predict ===
    raw_results = generate_predictions_batch(model.module, tokenizer, dataset, batch_size=batch_size)
    print(raw_results[0])

    # === Save ===
    predict_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predict_dir, exist_ok=True)
    raw_results_filename = f"raw_results_dpo_{test_sample_size}.json" if USE_LORA else "raw_results_baseline.json"
    raw_results_path = os.path.join(predict_dir, raw_results_filename)

    with open(raw_results_path, "w", encoding="utf-8") as f:
        json.dump(raw_results, f, indent=2, ensure_ascii=False)

    print(f"\nInference completed. Results saved to {raw_results_path}")
