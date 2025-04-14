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

| Model                              | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |
|------------------------------------|:---------:|:-------:|:-----------:|:----------:|:-------:|:-------:|:---------:|:------------:|
| ClusterIn-NegSampling              | 0.0077    | 0.012   | 666         | 0.0667     | 0.0593  | 0.0149  | 0.0631    | 0.687        |
| ClusterOut-LowExposure-NegSampling| 0.0077    | 0.012   | 694         | 0.0695     | 0.0603  | 0.0155  | 0.0620    | 0.652        |
| Two negative                       | 0.0077    | 0.012   | 668         | 0.0669     | 0.0597  | 0.0148  | 0.0615    | 0.685        |

### Proposed Method: Beam - based
| Model                              | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |
|------------------------------------|:---------:|:-------:|:-----------:|:----------:|:-------:|:-------:|:---------:|:------------:|
| Beam w/o Diversity                          | 0.0086| 0.017| 825     | 0.0827 | 0.0399 | 0.0125 | 0.0742 | 0.754    |
| Beam w Diversity        | 0.0096    | 0.018   | 849         | 0.0851     | 0.0403  | 0.0121  | 0.0732    | 0.787        |
| Beam w/o Diversity - two_stage on two neg      | 0.0101    | 0.017   | 851         | 0.0853     | 0.0398  | 0.0114  | 0.0755    | 0.837        |
| Beam w Diversity - two_stage on two neg | 0.0101 | 0.017 | 891         | 0.0893     | 0.0406  | 0.0115  | 0.0876    | 0.872        |
| Beam_Search_Negative_Generate_CD/Div_on_SFT_tuned_p_2.0 | 0.0157    | 0.026   | 1032        | 0.1034     | 0.0425  | 0.0123  | 0.0887    | 0.863        |
| Beam_Search_Negative_Generate_CD/Div_on_SFT_tuned_p_1.0 | 0.0134    | 0.022   | 974         | 0.0976     | 0.0431  | 0.0126  | 0.0928    | 0.854        |
| Beam_Search_Negative_Generate_CD/Div_on_SFT_tuned_p_0.5 | 0.0128    | 0.022   | 971         | 0.0973     | 0.0429  | 0.0124  | 0.0909    | 0.839        |


folders:
- [ClusterIn-NegSampling](./output/Clustering-Exposure_Balanced_Sampling_run1/hard-2) : output/Clustering-Exposure_Balanced_Sampling_run1/hard-2
- [ClusterOut-LowExposure-NegSampling](./output/Clustering-Exposure_Balanced_Sampling_run1/long_tail-2): output/Clustering-Exposure_Balanced_Sampling_run1/long_tail-2
- [Two negative](./output/Clustering-Exposure_Balanced_Sampling_run1/two_negatives): output/Clustering-Exposure_Balanced_Sampling_run1/two_negatives
- Beam w/o Diversity: output/Beam_Search_Negative_Generate/No_Div
- Beam w Diversity : output/Beam_Search_Negative_Generate/Div2
- Beam w/o Diversity - two_stage on two neg : Beam_Search_Negative_Generate/Div_on_two_neg2
- Beam w Diversity - two_stage on two neg: Beam_Search_Negative_Generate/No_Div_on_two_neg2

### Proposed Method: Beam-based Clustering-Exposure Balanced Sampling
#### p = 2.0
| Model                              | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |
|------------------------------------|:---------:|:-------:|:-----------:|:----------:|:-------:|:-------:|:---------:|:------------:|
| neg_sampling_balanced_popularity   |  0.0129   |  0.023  |     906     |   0.0908   |  0.041  | 0.0119  |  0.0685   |    0.797     |
| neg_sampling_clusterin_high_exposure |  0.0144   |  0.024  |     955     |   0.0957   | 0.0397  |  0.012  |  0.0928   |    0.855     |
| neg_sampling_clusterout_low_exposure |  0.0158   |  0.029  |     977     |   0.0979   |  0.051  | 0.0135  |   0.067   |    0.787     |
| neg_sampling_clustering_exposure_balanced |  0.0124   |  0.022  |     907     |   0.0909   | 0.0424  | 0.0119  |  0.0672   |    0.813     |84285147291395 | 0.011879364026443684 | 0.06723446893787575 |    0.813     |


#### p = 1.0
| Model                              | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |
|------------------------------------|:---------:|:-------:|:-----------:|:----------:|:-------:|:-------:|:---------:|:------------:|
| neg_sampling_balanced_popularity   |  0.0123   |  0.021  |     909     |   0.0911   | 0.0398  |  0.012  |  0.0698   |    0.793     |
| neg_sampling_clusterin_low_exposure |  0.0145   |  0.025  |     968     |   0.097    | 0.0419  | 0.0126  |  0.0696   |     0.79     |
| neg_sampling_clusterin_high_exposure |  0.0133   |  0.022  |     915     |   0.0917   | 0.0407  | 0.0124  |   0.098   |    0.853     |
| neg_sampling_clusterout_low_exposure |  0.0187   |  0.032  |    1005     |   0.1007   | 0.0481  |  0.013  |  0.0709   |    0.796     |
| neg_sampling_clusterout_high_exposure |  0.0116   |  0.02   |     918     |   0.092    | 0.0393  | 0.0118  |  0.0887   |    0.836     |
| neg_sampling_low_exposure          |  0.0145   |  0.025  |     968     |   0.097    | 0.0411  | 0.0123  |  0.0685   |    0.784     |
| neg_sampling_clustering_exposure_balanced |  0.0121   |  0.021  |     914     |   0.0916   |  0.041  | 0.0118  |  0.0692   |    0.809     |

### Baseline on DPO w/o SFT-tuned

| Model                         | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓  | MGU ↓  | ORRatio ↓ | NotInRatio ↓ |
|-------------------------------|:--------:|:------:|:---------:|:--------:|:-----:|:-----:|:-------:|:-------:|
| SPRec_wo_STF                  | 0.0028   | 0.006  |   647     | 0.0648   | 0.0721 | 0.0165 | 0.0738  | 0.639 |
| SPRec                         | 0.0032   | 0.007  |   608     | 0.0609   | 0.0797 | 0.0172 | 0.1004  | 0.542 |

folders:
- SPRec_wo_STF: SPRec_wo_STF_run2
- SPRec: SPRec_run1

### Clustering-Exposure Balanced Sampling on DPO w/o SFT-tuned
| Model                                            | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓  | MGU ↓  | ORRatio ↓ | NotInRatio ↓ |
|--------------------------------------------------|:--------:|:------:|:---------:|:--------:|:-----:|:-----:|:-------:|:-------:|
| ClusterIn-NegSampling                            | 0.0032   | 0.007  |    618    | 0.0619   | 0.0768 | 0.0167 | 0.0940  | 0.562 |
| ClusterOut-LowExposure-NegSampling               | 0.0028   | 0.006  |    610    | 0.0611   | 0.0799 | 0.0172 | 0.1166  | 0.47 |

folders:
- ClusterIn-NegSampling: Clustering-Exposure_Balanced_Sampling_run1/hard
- ClusterOut-LowExposure-NegSampling: Clustering-Exposure_Balanced_Sampling_run1/long_tail

## Constrained Decoding (only for predict output)

### Clustering-Exposure_Balanced_Sampling
| Model                                          | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |
|------------------------------------------------|-----------|---------|-------------|------------|---------|---------|-----------|--------------|
|   two negative  | 0.0086    | 0.015  | 637   | 0.0638     | 0.0701  | 0.0168  | 0.0621    | 0       |
| SPRec             | 0.0086    | 0.015  | 649    | 0.0650     | 0.0715  | 0.0173  | 0.0611    | 0      |


- two negative : Clustering-Exposure_Balanced_Sampling_run2
- SPRec : Constrained_Predict_Generate/SPRec

## code
- smolLM2-1.7B-lora-run3 (SFT): sft_smol.py
- smolLM2-1.7B-lora-dpo-run3 (DPO): DPO4.py - LossThresholdCallback(threshold=0.1)
- smolLM2-1.7B-lora-dpo-run6 (DPO): DPO5.py
- SPRec_wo_STF_run2  : dataset_generate.py $\rightarrow$ DPO_from_dpoData.py
- SPRec_run1 (SFT+DPO)  : dataset_generate.py (with smolLM2-1.7B-lora-run3 model) $\rightarrow$ DPO_from_dpoData.py
- Clustering-Exposure Balanced Sampling: dataset_generate_cluster_batch.py $\rightarrow$ DPO_from_dpoData.py $\rightarrow$ generate_predict_batch.py
- beam-based: 
  - beam_negative_generate_CD.py $\rightarrow$ add_details.py $\rightarrow$ neg_sampling_generator_from_details_data.py $\rightarrow$ S-DPO.py/DPO_on_SFT.py

## neg data
- sample 1024 train & valid data: sampled_data/
- predictions from origin model as neg: output/SPRec_wo_STF_run1/data/dpo_train_list.json
- random sample: output/random_sample/


## 小筆記
- 之後要讓evaluate.py的update_csv都更新同一個
- our方法在DGU, MGU小贏，其他都輸
- two negative最好，可能可以改mutiple negative，或再加一層beam search出來neg 做DPO 的應該會貼近SPRec?(接近模型本身輸出->最有信心->最難)
