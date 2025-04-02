import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig, DPODataCollatorWithPadding

def load_model(base_model_name):
    print("\nLoad model:", base_model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def add_lora(model):
    print("\nUsing LoRA")
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def prepare_dpo_dataset(path, sample_size=64):
    print("\nLoad dataset:", path)
    raw_dataset = load_dataset("json", data_files=path)["train"].select(range(sample_size))

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
            negative = texts[torch.randint(0, len(texts), (1,)).item()]
        dpo_samples.append({
            "prompt": prompt,
            "chosen": positive,
            "rejected": negative
        })

    return Dataset.from_list(dpo_samples)

def train_model(model, tokenizer, dpo_data, output_dir="./output"):
    print("\nStart DPO training...")
    dpo_config = DPOConfig(
        beta=0.1,
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dpo_data,
        data_collator=DPODataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    final_output_dir = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(final_output_dir, safe_serialization=False)
    trainer.tokenizer.save_pretrained(final_output_dir)
    print("Training completed, model saved to:", final_output_dir)

if __name__ == "__main__":
    seed = 0
    train_sample_size = 1024
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    run = "1"

    output_path = f"/scratch/user/chuanhsin0110/test_0321/output/smolLM2-1.7B-lora-dpo-run{run}"
    train_data_path = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/train.json"

    # ============ Output dir check ============
    if os.path.exists(output_path + "/final_model"):
        print(f"Warning: Output dir '{output_path}' already exists. It may overwrite previous models.")
        exit(1)
    else:
        os.makedirs(output_path, exist_ok=True)
        print(f"Created output dir: {output_path}")
    # ===========================================

    print("torch version:", torch.version.cuda)
    print("Load base model")
    model, tokenizer = load_model(model_name)

    print("Prepare DPO dataset")
    dpo_data = prepare_dpo_dataset(train_data_path, train_sample_size)

    print("Add LoRA")
    model = add_lora(model)

    print("Train model with DPO")
    train_model(model, tokenizer, dpo_data, output_path)
