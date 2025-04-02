import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


def load_model(base_model_name="facebook/opt-350m"):
    print("\nload model:" ,base_model_name)
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    collator = DataCollatorForCompletionOnlyLM(response_template="### Response:", tokenizer=tokenizer)
    model = prepare_model_for_kbit_training(model)
    return model, collator

def prepare_dataset(path, sample_size=64):
    print("\nload dataset:", path)
    dataset = load_dataset("json", data_files=path)["train"].select(range(sample_size))
    return dataset

def add_lora(model):
    print("\nUsing LoRA")
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def formatting_prompts_func(example):
    
    instruction = example["instruction"]
    input_text = example["input"]
    response = example["output"]

    if input_text.strip():
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{response}"""
    else:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""

def train_model(model, collator, train_data, val_data, output_dir="./output"):
    print("\nstart traning...")
    args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=1,
        bf16=True,
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=args,
        formatting_func=formatting_prompts_func,
        data_collator=collator
    )
    trainer.train()
    # trainer.save_model(output_dir)
    # print("traning completed, save to:", output_dir)

    final_output_dir = os.path.join(output_dir, "final_model")
    trainer.model.save_pretrained(final_output_dir, safe_serialization=False)
    trainer.tokenizer.save_pretrained(final_output_dir)
    print("traning completed, save to:", final_output_dir)

if __name__ == "__main__":
    seed=0
    train_sample_size:int = 1024
    # base_model = "facebook/opt-350m"

    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


    output_path = "/scratch/user/chuanhsin0110/test_0321/output/smolLM2-1.7B-lora-run1"
    train_data_path = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/train.json"
    valid_data_path = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/valid.json"

    print("torch version: ",torch.version.cuda)
    print("Load base model")
    model, collator = load_model(model_name)
    
    print("Load dataset\n")
    #train_data = prepare_dataset(train_data_path, 500)
    #val_data = prepare_dataset(valid_data_path, 100)
    train_dataset = load_dataset("json", data_files=train_data_path)
    train_data = train_dataset["train"].shuffle(seed=seed).select(range(train_sample_size))
    val_dataset = load_dataset("json", data_files=valid_data_path)
    val_data = val_dataset["train"].shuffle(seed=seed).select(range(int(train_sample_size/8)))
    print("Add LoRA")
    model = add_lora(model)
    print("Train Model")
    train_model(model, collator, train_data, val_data, output_path)