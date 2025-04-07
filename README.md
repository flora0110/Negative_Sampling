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

## Procedure
### Data
- sample data: 
  - [location](./sampled_data)
  - train_sample_size = 1024
  - valid_sample_size = 256
  - test_sample_size = 1000
  - from /scratch/user/username/SPRec/data/Goodreads/

### Model
HuggingFaceTB/SmolLM2-1.7B-Instruct

### Code
- ClusterIn-NegSample  ClusterOut-LowExposure-NegSample Generate: [dataset_generate_cluster_batch.py](dataset_generate_cluster_batch.py)
- DPO tuning: [DPO_on_SFT.py](DPO_on_SFT.py)
- Generate Prediction: [generate_predict_batch.py](generate_predict_batch.py)
- Evaluate: [evaluate.py](evaluate.py)

## Future Work

- [ ] Add **Beam Search Hard Negative Sampling** (very hard negatives with higher model confidence)


## result

### Direct Generate Baseline
| Model                     | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓  | MGU ↓  | ORRatio ↓ | NotInRatio ↓ |
|---------------------------|:--------:|:------:|:---------:|:--------:|:-----:|:-----:|:-------:| :-------:|
| origin model              | 0.0038   | 0.007  |   510     | 0.0511   | 0.0837 | 0.0179 | 0.1254  | 0.451 |
| SFT-tuned                 | 0.0043   | 0.010  |   608     | 0.0609   | 0.0615 | 0.0163 | 0.0707  | 0.551 |
| DPO-tuned w RN            | 0.0077   | 0.012  |   670     | 0.0671   | 0.0601 | 0.0161 | 0.0648  | 0.624 |

folders:
- origin model: molLM2-1.7B-Instruct
- SFT-tuned: smolLM2-1.7B-lora-run3
- DPO_RN_on_SFT-1: DPO_RN_on_SFT-1

### Self-Play Baseline

| Model                     | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓  | MGU ↓  | ORRatio ↓ | NotInRatio ↓ |
|---------------------------|:--------:|:------:|:---------:|:--------:|:-----:|:-----:|:-------:|:-------:|
| SPRec                                        | 0.0082   | 0.013  |   694     | 0.0695   | 0.0618 | 0.0153 | 0.0586  | 0.671 |
| SPRec_wo_SFT(but DPO on SFT-tuned model)     | 0.0077   | 0.012  |   713     | 0.0714   | 0.0585 | 0.0148 | 0.0622  | 0.714 |

folders:
- SPRec: output/SPRec_on_SFT-1
- SPRec_wo_SFT: output/SPRec_wo_SFT_DPO_on_SFT-1

### Proposed Method: Clustering-Exposure Balanced Sampling

| Model                                  | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓  | MGU ↓  | ORRatio ↓ | NotInRatio ↓ |
|----------------------------------------|:--------:|:------:|:---------:|:--------:|:-----:|:-----:|:-------:|:-------:|
| ClusterIn-NegSampling                  | 0.0077   | 0.012  |    666    | 0.0667   | 0.0593 | 0.0149 | 0.0631  | 0.687 |
| ClusterOut-LowExposure-NegSampling     | 0.0077   | 0.012  |    694    | 0.0695   | 0.0603 | 0.0155 | 0.0620  | 0.652 |
| Two negative                           | 0.0077   | 0.012  |    668    | 0.0669   | 0.0597 | 0.0148 | 0.0615  | 0.685 |

folders:
- [ClusterIn-NegSampling](./output/Clustering-Exposure_Balanced_Sampling_run1/hard-2) : output/Clustering-Exposure_Balanced_Sampling_run1/hard-2
- [ClusterOut-LowExposure-NegSampling](./output/Clustering-Exposure_Balanced_Sampling_run1/long_tail-2): output/Clustering-Exposure_Balanced_Sampling_run1/long_tail-2
- [Two negative](./output/Clustering-Exposure_Balanced_Sampling_run1/two_negatives): output/Clustering-Exposure_Balanced_Sampling_run1/two_negatives


### Baseline on DPO w/o SFT-tuned

| Model                         | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓  | MGU ↓  | ORRatio ↓ | NotInRatio ↓ |
|-------------------------------|:--------:|:------:|:---------:|:--------:|:-----:|:-----:|:-------:|:-------:|
| SPRec_wo_STF                  | 0.0028   | 0.006  |   647     | 0.0648   | 0.0721 | 0.0165 | 0.0738  |
| SPRec                         | 0.0032   | 0.007  |   608     | 0.0609   | 0.0797 | 0.0172 | 0.1004  | 0.542 |

folders:
- SPRec_wo_STF: SPRec_wo_STF_run2
- SPRec: SPRec_run1

### Clustering-Exposure Balanced Sampling on DPO w/o SFT-tuned
| Model                                            | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓  | MGU ↓  | ORRatio ↓ | NotInRatio ↓ |
|--------------------------------------------------|:--------:|:------:|:---------:|:--------:|:-----:|:-----:|:-------:|:-------:|
| ClusterIn-NegSampling                            | 0.0032   | 0.007  |    618    | 0.0619   | 0.0768 | 0.0167 | 0.0940  | 562 |
| ClusterOut-LowExposure-NegSampling               | 0.0028   | 0.006  |    610    | 0.0611   | 0.0799 | 0.0172 | 0.1166  | 0.47 |

folders:
- ClusterIn-NegSampling: Clustering-Exposure_Balanced_Sampling_run1/hard
- ClusterOut-LowExposure-NegSampling: Clustering-Exposure_Balanced_Sampling_run1/long_tail

## code
- smolLM2-1.7B-lora-run3 (SFT): sft_smol.py
- smolLM2-1.7B-lora-dpo-run3 (DPO): DPO4.py - LossThresholdCallback(threshold=0.1)
- smolLM2-1.7B-lora-dpo-run6 (DPO): DPO5.py
- SPRec_wo_STF_run2  : dataset_generate.py $\rightarrow$ DPO_from_dpoData.py
- SPRec_run1 (SFT+DPO)  : dataset_generate.py (with smolLM2-1.7B-lora-run3 model) $\rightarrow$ DPO_from_dpoData.py
- Clustering-Exposure Balanced Sampling: dataset_generate_cluster_batch.py $\rightarrow$ DPO_from_dpoData.py $\rightarrow$ generate_predict_batch.py

## neg data
- sample 1024 train & valid data: sampled_data/
- predictions from origin model as neg: output/SPRec_wo_STF_run1/data/dpo_train_list.json
- random sample: output/random_sample/


## 小筆記
- 之後要讓evaluate.py的update_csv都更新同一個
- our方法在DGU, MGU小贏，其他都輸
- two negative最好，可能可以改mutiple negative，或再加一層beam search出來neg 做DPO 的應該會貼近SPRec?(接近模型本身輸出->最有信心->最難)
