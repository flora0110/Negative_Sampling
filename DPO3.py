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

class DPOPreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']

        prompt_ids = self.tokenizer(prompt, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        chosen_ids = self.tokenizer(chosen, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")
        rejected_ids = self.tokenizer(rejected, return_tensors="pt", max_length=self.max_length, truncation=True, padding="max_length")

        return {
            'prompt_ids': prompt_ids['input_ids'].squeeze(0),
            'chosen_ids': chosen_ids['input_ids'].squeeze(0),
            'rejected_ids': rejected_ids['input_ids'].squeeze(0)
        }

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

def dpo_loss(model, prompt_ids, chosen_ids, rejected_ids, beta=0.05):
    with torch.no_grad():
        ref_logits_chosen = model(prompt_ids, labels=chosen_ids).logits
        ref_logits_rejected = model(prompt_ids, labels=rejected_ids).logits
        ref_logps_chosen = torch.sum(torch.log_softmax(ref_logits_chosen, -1).gather(-1, chosen_ids.unsqueeze(-1)).squeeze(-1), dim=-1)
        ref_logps_rejected = torch.sum(torch.log_softmax(ref_logits_rejected, -1).gather(-1, rejected_ids.unsqueeze(-1)).squeeze(-1), dim=-1)

    logits_chosen = model(prompt_ids, labels=chosen_ids).logits
    logits_rejected = model(prompt_ids, labels=rejected_ids).logits
    logps_chosen = torch.sum(torch.log_softmax(logits_chosen, -1).gather(-1, chosen_ids.unsqueeze(-1)).squeeze(-1), dim=-1)
    logps_rejected = torch.sum(torch.log_softmax(logits_rejected, -1).gather(-1, rejected_ids.unsqueeze(-1)).squeeze(-1), dim=-1)

    logits_diff = beta * ((logps_chosen - ref_logps_chosen) - (logps_rejected - ref_logps_rejected))
    loss = nn.functional.binary_cross_entropy_with_logits(logits_diff, torch.ones_like(logits_diff))

    return loss

# ========== Training ==========

def train_dpo(model_name, train_data, valid_data, output_dir, epochs=2, batch_size=2, lr=1e-5, beta=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    train_dataset = DPOPreferenceDataset(train_data, tokenizer)
    valid_dataset = DPOPreferenceDataset(valid_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=50, num_training_steps=epochs*len(train_loader))

    # Create output dir
    final_output_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_output_dir, exist_ok=True)

    model.train()
    best_valid_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            loss = dpo_loss(
                model,
                batch['prompt_ids'].to(device),
                batch['chosen_ids'].to(device),
                batch['rejected_ids'].to(device),
                beta
            )
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

        # ===== Validation =====
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                loss = dpo_loss(
                    model,
                    batch['prompt_ids'].to(device),
                    batch['chosen_ids'].to(device),
                    batch['rejected_ids'].to(device),
                    beta
                )
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        print(f"Epoch {epoch+1} - Validation Loss: {valid_loss:.4f}")

        # Save best
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.save_pretrained(final_output_dir)
            tokenizer.save_pretrained(final_output_dir)
            print(f"Best model saved to {final_output_dir} (Val Loss: {valid_loss:.4f})")

        model.train()
        torch.cuda.empty_cache()

# ========== 執行 ==========

if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True

    seed = 0
    train_sample_size = 1024
    valid_sample_size = 256
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    run = "3"

    output_path = f"/scratch/user/chuanhsin0110/test_0321/output/smolLM2-1.7B-lora-dpo-run{run}"
    train_data_path = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/train.json"
    valid_data_path = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/valid.json"

    # Prepare data
    train_data = prepare_dpo_dataset(train_data_path, sample_size=train_sample_size)
    valid_data = prepare_dpo_dataset(valid_data_path, sample_size=valid_sample_size)

    # Train
    train_dpo(
        model_name,
        train_data,
        valid_data,
        output_dir=output_path,
        epochs=2,
        batch_size=1,
        lr=1e-5,
        beta=0.05
    )
