import re
import json
import os
import random
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "SPRec_wo_STF_run1"
model_name = "SPRec_run1"
USE_LORA = True  # 改成 False 就會用原始模型
BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TUNED_MODEL = "smolLM2-1.7B-lora-run3"
FINETUNED_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{TUNED_MODEL}/final_model"
SAVE_PATH = f"/scratch/user/chuanhsin0110/test_0321/output/{model_name}/data/"

def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded[len(prompt):].strip()
    return response

def prepare_prompt(instruction, input_text):
    prompt = f"### Instruction:\n{instruction}\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text}\n"
    prompt += "### Response:"
    return prompt

def process_data(train_path, valid_path, model, tokenizer, train_size=1024, valid_size=128):
    # tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto").eval()

    # 自動創建資料夾
    os.makedirs(SAVE_PATH, exist_ok=True)

    def load_and_sample(path, size):
        with open(path, "r") as f:
            data = json.load(f)
        return random.sample(data, size)

    train_data = load_and_sample(train_path, train_size)
    valid_data = load_and_sample(valid_path, valid_size)
    full_data = train_data + valid_data

    dpo_data = []

    for d in tqdm(full_data, desc="Generating negatives"):
        prompt = prepare_prompt(d["instruction"], d["input"])
        predict = generate_response(model, tokenizer, prompt)

        # Parse prediction as rejected
        match = re.search(r'"([^"]*)', predict)
        if match:
            rejected = f"\"{match.group(1)}\"\n"
        else:
            rejected = predict.split("\n")[0]

        dpo_data.append({
            "prompt": prompt,
            "chosen": d["output"].strip(),
            "rejected": rejected
        })

    # Split & Save
    dpo_train = dpo_data[:train_size]
    dpo_valid = dpo_data[train_size:]

    with open(os.path.join(SAVE_PATH, "dpo_train.json"), "w", encoding="utf-8") as f:
        json.dump(dpo_train, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SAVE_PATH, "dpo_valid.json"), "w", encoding="utf-8") as f:
        json.dump(dpo_valid, f, indent=2, ensure_ascii=False)

    print(f"\nDPO dataset saved to {SAVE_PATH}")

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
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if USE_LORA:
        model = PeftModel.from_pretrained(model, FINETUNED_PATH)

    process_data(
        train_path="/scratch/user/chuanhsin0110/test_0321/sampled_data/train_sample.json",
        valid_path="/scratch/user/chuanhsin0110/test_0321/sampled_data/valid_sample.json",
        model=model,
        tokenizer=tokenizer,
    )
