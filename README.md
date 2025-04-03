# Clustering-Exposure Balanced Sampling
## Overview

This pipeline generates a **balanced and diverse DPO dataset** using:
- **User Interest Clustering**
- **Exposure-based Long-tail Negative Sampling**
- **Hard Negative Sampling (Top-K Sampling)**

Supports both **DPO** and **S-DPO** training formats.

---

## Sampling Procedure

### 1. User Interest Cluster

For each user's input history:
- Retrieve corresponding genres from `name2genre.json`.
- Count genre occurrences and select **Top-3 genres** as the user's **Interest Cluster**.

---

### 2. Hard Negative Sampling (Top-K Sampling)

For each training instance:
- Use **Top-K Sampling** with:
  - `top_k=50`
  - `do_sample=True`
- **Dynamic Sampling:**
  - If no candidate belongs to the user's Interest Cluster:
    - Continue sampling (`.generate()`) and append new candidates.
  - Maximum of **40 candidates** to prevent infinite loops.
- **Select the first matching candidate** as **Hard Negative**.
- If no match found after 40 candidates → **Randomly select one candidate**.

---

### 3. Long-tail Negative Sampling

For each training instance:
- From all generated `candidates`:
  - Filter candidates **outside the user's Interest Cluster**.
  - Sort by **Exposure Count** (frequency in training data) in ascending order.
  - Select the **least exposed candidate** as **Long-tail Negative**.
- If no matching candidate → **Randomly select one candidate**.

---

## Output Files

The generated datasets are saved to:
```
/output/{method_name}/data/
├── balanced_data.json            # Full dataset
├── balanced_train.json           # Train split
├── balanced_valid.json           # Validation split
├── dpo_hard.json                 # DPO format with Hard Negative
├── dpo_long_tail.json            # DPO format with Long-tail Negative
├── dpo_two_negatives.json        # S-DPO format (two negatives)
├── dpo_hard_train.json
├── dpo_hard_valid.json
├── dpo_long_tail_train.json
├── dpo_long_tail_valid.json
├── dpo_two_negatives_train.json
├── dpo_two_negatives_valid.json
```
## Future Work

- [ ] Support **S-DPO Training** (multi-negative loss using `dpo_two_negatives.json`)
- [ ] Add **Beam Search Hard Negative Sampling** (very hard negatives with higher model confidence)


## BASELINE
### result
| Model                                   | NDCG@10 | HR@10 | Diversity | DivRatio | DGU   | MGU   | ORRatio |
|----------------------------------------|:------:|:-----:|:--------:|:-------:|:-----:|:-----:|:------:|
| smolLM2-1.7B-Instruct (origin model)   | 0.0038 | 0.007 |   510    | 0.0511  | 0.0837| 0.0179| 0.1254 |
| smolLM2-1.7B-lora-run3 (SFT)           | 0.0043 | 0.010 |   608    | 0.0609  | 0.0615| 0.0163| 0.0707 |
| smolLM2-1.7B-lora-dpo-run3 (DPO)       | 0.0025 | 0.006 |   735    | 0.0736  | 0.0747| 0.0148| 0.0868 |
| smolLM2-1.7B-lora-dpo-run6 (DPO)       | 0.0031 | 0.007 |   542    | 0.0543  | 0.0797| 0.0172| 0.1176 |
| SPRec_wo_STF_run2                      | 0.0028 | 0.006 |   647    | 0.0648  | 0.0721| 0.0165| 0.0738 |
| SPRec_run1 (SFT+DPO)                   | 0.0032 | 0.007 |   608    | 0.0609  | 0.0797| 0.0172| 0.1004 |
| Clustering-Exposure_Balanced_Sampling_run1 | 0.0032 | 0.007 |   618    | 0.0619  | 0.0768| 0.0167| 0.0940 |



### code
- smolLM2-1.7B-lora-run3 (SFT): sft_smol.py
- smolLM2-1.7B-lora-dpo-run3 (DPO): DPO4.py - LossThresholdCallback(threshold=0.1)
- smolLM2-1.7B-lora-dpo-run6 (DPO): DPO5.py
- SPRec_wo_STF_run2  : dataset_generate.py $\rightarrow$ DPO_from_dpoData.py
- SPRec_run1 (SFT+DPO)  : dataset_generate.py (with smolLM2-1.7B-lora-run3 model) $\rightarrow$ DPO_from_dpoData.py

### self play
dataset generate $\rightarrow$ DPO

