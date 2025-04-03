import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset, Dataset
from trl import DPOTrainer, DPOConfig
import json

class LossThresholdCallback(TrainerCallback):
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_loss = metrics.get("eval_loss", None)
        if current_loss is not None and current_loss < self.threshold:
            print(f">>> Eval loss {current_loss:.4f} < threshold {self.threshold}, stopping training.")
            control.should_early_stop = True
            control.should_save = True

# ===== 1. 載入 Base Model =====
def load_model(base_model_name):
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

# ===== 2. 加入 LoRA =====
def add_lora(model):
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,  # 建議加入 dropout 防 overfitting
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

# # ===== 3. DPO 資料集製作 (Random Negative) =====
# def prepare_dpo_dataset(path, sample_size=64):
#     raw_dataset = load_dataset("json", data_files=path)["train"].select(range(sample_size))
#     texts = [d["output"] for d in raw_dataset]
#     dpo_samples = []
#     for d in raw_dataset:
#         prompt = f"### Instruction:\n{d['instruction']}\n"
#         if d["input"].strip():
#             prompt += f"### Input:\n{d['input']}\n"
#         prompt += "### Response:"

#         positive = d["output"].strip()
#         negative = positive
#         while negative == positive:
#             negative = texts[torch.randint(0, len(texts), (1,)).item()]
#         dpo_samples.append({
#             "prompt": prompt,
#             "chosen": positive,
#             "rejected": negative
#         })

#     return Dataset.from_list(dpo_samples)

# ===== 4. 訓練流程（加入 Early Stopping + Generation Check） =====
def train_model(model, ref_model, tokenizer, dpo_data, val_data, output_dir="./output"):
    print("\nStarting DPO training...")

    # dpo_config = DPOConfig(
    #     beta=0.01,  # 保守 β
    #     output_dir=output_dir,
    #     per_device_train_batch_size=2,
    #     per_device_eval_batch_size=2,
    #     gradient_accumulation_steps=2,
    #     learning_rate=1e-5,
    #     num_train_epochs=5,  # 允許較大 epoch
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     greater_is_better=False,
    #     bf16=True,
    #     logging_steps=10,
    # )
    training_args = DPOConfig(
        beta=0.1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        num_train_epochs=1,
        learning_rate=1e-5,
        bf16=True,
        logging_steps=1,
        optim="adamw_torch",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        output_dir=output_dir,
        save_total_limit=1,
        load_best_model_at_end=True,
        max_prompt_length=512,
        max_length=512,
    )

    # trainer = DPOTrainer(
    #     model=model,
    #     ref_model=ref_model,
    #     args=dpo_config,
    #     train_dataset=dpo_data,
    #     eval_dataset=val_data,
    #     processing_class=tokenizer,
    #     callbacks=[
    #         EarlyStoppingCallback(early_stopping_patience=2),
    #         LossThresholdCallback(threshold=0.05),
    #     ]
    # )
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dpo_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
    )

    trainer.train()

    final_output_dir = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(final_output_dir, safe_serialization=False)
    tokenizer.save_pretrained(final_output_dir)
    print("\nTraining completed. Best model saved to:", final_output_dir)

    
# ===== 5. 主程式 =====
if __name__ == "__main__":
    resume_from_checkpoint = False

    method_name = "Clustering-Exposure_Balanced_Sampling_run"
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    run = "1"
    sample_method = "long_tail"

    output_path = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}{run}/{sample_method}"
    train_data_path = f"/scratch/user/chuanhsin0110/test_0321/output/Clustering-Exposure_Balanced_Sampling_run1/data/dpo_long_tail_train.json"
    valid_data_path = f"/scratch/user/chuanhsin0110/test_0321/output/Clustering-Exposure_Balanced_Sampling_run1/data/dpo_long_tail_valid.json"
    # train_data_path = f"/scratch/user/chuanhsin0110/test_0321/output/random_sample/train.json"
    # valid_data_path = f"/scratch/user/chuanhsin0110/test_0321/output/random_sample/valid.json"

    # Prepare data
    with open(train_data_path, "r") as f:
        train_data = json.load(f)
    with open(valid_data_path, "r") as f:
        valid_data = json.load(f) 
    
    # 轉為 HuggingFace Dataset
    train_data = Dataset.from_list(train_data)
    valid_data = Dataset.from_list(valid_data)
    if os.path.exists(output_path + "/final_model"):
        print(f"Warning: Output dir '{output_path}' already exists.")
        exit(1)
    else:
        os.makedirs(output_path, exist_ok=True)

    model, tokenizer = load_model(model_name)
    ref_model, _ = load_model(model_name)



    model = add_lora(model)

   
    
    

    train_model(model, ref_model, tokenizer, train_data, valid_data, output_path)
