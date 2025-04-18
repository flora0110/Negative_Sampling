import json, os
def generate_prompt(instruction, input_text=None):
    """
    Generate a prompt from the given instruction and optional input.
    
    Args:
        instruction (str): The instruction describing the task.
        input_text (str, optional): Additional context for the task.
        
    Returns:
        str: A formatted prompt string.
    """
    if input_text:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""
train_data_path = "/scratch/user/chuanhsin0110/test_0321/nlp/dpo_data/train_dpo.json"
valid_data_path = "/scratch/user/chuanhsin0110/test_0321/nlp/dpo_data/valid_dpo.json"
SAVE_PATH = "/scratch/user/chuanhsin0110/test_0321/nlp/dpo_data"

with open(train_data_path, 'r', encoding='utf8') as f:
    train_data = json.load(f)

with open(valid_data_path, 'r', encoding='utf8') as f:
    valid_data = json.load(f)

dpo_train = []
for d in train_data:
    prompt = generate_prompt(d["instruction"], d["input"])
    dpo_train.append({
        "prompt": prompt,
        "chosen": d["chosen"],
        "rejected": d["rejected"],
    })

with open(os.path.join(SAVE_PATH, "train.json"), "w", encoding="utf-8") as f:
    json.dump(dpo_train, f, indent=2, ensure_ascii=False)


dpo_valid = []
for d in valid_data:
    prompt = generate_prompt(d["instruction"], d["input"])
    dpo_valid.append({
        "prompt": prompt,
        "chosen": d["chosen"],
        "rejected": d["rejected"],
    })

with open(os.path.join(SAVE_PATH, "valid.json"), "w", encoding="utf-8") as f:
    json.dump(dpo_valid, f, indent=2, ensure_ascii=False)