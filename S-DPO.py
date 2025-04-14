import os
import torch
import random

from peft import PeftModel, prepare_model_for_kbit_training
from transformers import TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trainer.softmax_dpo_trainer import DPOTrainer
from accelerate import Accelerator
import fire


random.seed(1958)

def train(
    resume_from_checkpoint: str = "/scratch/user/chuanhsin0110/test_0321/output/smolLM2-1.7B-lora-run3/final_model",
    # resume_from_checkpoint: str = "/scratch/user/chuanhsin0110/test_0321/output/Clustering-Exposure_Balanced_Sampling_run1/two_negatives/final_model",
    beta: float = 0.1,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-5,
    cutoff_len: int = 512,
    # wandb config
    wandb_project: str = "",
    wandb_name: str = "",
    MAX_NEG_NUM: int = 5,
):


    D = "Div"
    # base_model_name = "SFT_tuned"

    num_return_sequences = "10"      # 請依實際情況替換
    diversity_penalty = "2.0"   # 請依實際情況替換
    method_name = f"Beam_Search_Negative_Generate_CD"
    #run = "1"
    #sample_method = "two_negatives"
    sample_method = "neg_sampling_clustering_exposure_balanced"
    # output_path = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}{run}/{sample_method}"
    # output_path = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/{Div}_on_{base_model_name}_p_{diversity_penalty}"
    output_path = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/{D}_{num_return_sequences}_{diversity_penalty}/{sample_method}"

    final_output_dir = os.path.join(output_path, "final_model")

    if os.path.exists(output_path + "/final_model"):
        print(f"Warning: Output dir '{output_path}' already exists. It may overwrite previous models.")
        exit(1)
    else:
        os.makedirs(output_path, exist_ok=True)

    if resume_from_checkpoint is None:
        resume_from_checkpoint = output_path

    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    # suffix = f"_{Div}_10_{diversity_penalty}"
    data_files = {
        # "train": f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/data/train{suffix}.json",
        # "validation": f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/data/valid{suffix}.json",
        "train": f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/{D}_{num_return_sequences}_{diversity_penalty}/{sample_method}/data/train.json",
        "validation": f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/{D}_{num_return_sequences}_{diversity_penalty}/{sample_method}/data/valid.json",
    }

    def process_data(examples):
        dic = {"prompt": [], "chosen": []}
        #max_neg_num = max([len(r) for r in examples["rejected"]])
        max_neg_num = MAX_NEG_NUM
        # print(f"\n\n\n\nmax_neg_num: {max_neg_num}")
        for i in range(1, max_neg_num + 1):
            dic[f"rejected{i}"] = []

        for i in range(len(examples["prompt"])):
            dic["prompt"].append(examples["prompt"][i])
            dic["chosen"].append(examples["chosen"][i])

            rejected_list = examples["rejected"][i]
            for j in range(max_neg_num):
                if j < len(rejected_list):
                    value = rejected_list[j]
                    # 如果 value 為 None，則用空字串替代
                    if value is None:
                        value = ""
                    dic[f"rejected{j+1}"].append(value)
                else:
                    dic[f"rejected{j+1}"].append("")  # padding 空字串
        # print(dic)
        return dic

    data = load_dataset("json", data_files=data_files)

    columns = data["train"].column_names
    train_data = data["train"].map(process_data, remove_columns=columns, num_proc=8, batched=True).shuffle(seed=42)
    val_data = data["validation"].map(process_data, remove_columns=columns, num_proc=8, batched=True).shuffle(seed=42)
    print(train_data[0])
    if val_data.num_rows > 2000:
        val_data = val_data.select(range(2000))

    device_index = Accelerator().process_index
    # device_map = {"": device_index}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # ==== Policy Model ====
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )
    base_model.config.use_cache = False  # for training
    base_model = prepare_model_for_kbit_training(base_model)
    base_model = PeftModel.from_pretrained(
        base_model, resume_from_checkpoint, is_trainable=True
    )
    base_model.print_trainable_parameters()

    # ==== Reference Model (Frozen) ====
    model_ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config
    )
    model_ref = prepare_model_for_kbit_training(model_ref)
    reference_model = PeftModel.from_pretrained(model_ref, resume_from_checkpoint)
    reference_model.print_trainable_parameters()

    # ==== Tokenizer ====
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=1,
        output_dir=output_path,
        # report_to="wandb",
        report_to = None,
        run_name=wandb_name,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        ddp_find_unused_parameters=False,
    )

    dpo_trainer = DPOTrainer(
        base_model,
        reference_model,
        args=training_args,
        beta=beta,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        max_prompt_length=cutoff_len,
        max_length=cutoff_len,
    )

    dpo_trainer.train()

    # 會自動 load best model at end
    os.makedirs(final_output_dir, exist_ok=True)
    dpo_trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"已將最佳模型存到：{final_output_dir}")

if __name__ == "__main__":
    fire.Fire(train)
