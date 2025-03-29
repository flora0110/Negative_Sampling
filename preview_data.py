from datasets import load_dataset

def prepare_dataset(path, sample_size=64):
    print("\n載入資料集:", path)
    raw = load_dataset("json", data_files=path)
    print("dataset key:", raw.keys())

    dataset = raw["train"]
    print(f"總筆數：{len(dataset)}")
    
    print("\n前幾筆樣本：")
    for i in range(min(3, len(dataset))):
        print(f"\n樣本 {i+1}:")
        print(dataset[i])
    
    return dataset.select(range(min(sample_size, len(dataset))))

def formatting_prompts_func(examples):
    output_texts = []
    for example in examples:

        instruction = example["instruction"]
        input_text = example["input"]
        response = example["output"]

        if input_text.strip():
            output_texts.append(f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input_text}

    ### Response:
    {response}""")
        else:
            output_texts.append(f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:
    {response}""")

    return output_texts


if __name__ == "__main__":
    train_data_path = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/train.json"
    dataset = prepare_dataset(train_data_path)
    print("\ndataset 類型：", type(dataset))
    print("dataset 長度：", len(dataset))
    print("\ndataset 全部樣本（限前3筆）：")
    for i in range(min(3, len(dataset))):
        print(f"樣本 {i+1}: {dataset[i]}\n")

    valid_data_path = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/valid.json"
    dataset = prepare_dataset(valid_data_path)
    print("\ndataset 類型：", type(dataset))
    print("dataset 長度：", len(dataset))
    print("\ndataset 全部樣本（限前3筆）：")
    for i in range(min(3, len(dataset))):
        print(f"樣本 {i+1}: {dataset[i]}\n")

    valid_data_path = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/test.json"
    dataset = prepare_dataset(valid_data_path)
    print("\ndataset 類型：", type(dataset))
    print("dataset 長度：", len(dataset))
    print("\ndataset 全部樣本（限前3筆）：")
    for i in range(min(3, len(dataset))):
        print(f"樣本 {i+1}: {dataset[i]}\n")

    # output_texts = formatting_prompts_func(dataset)
    # for output_text in output_texts[:10]:
    #     print(output_text)
