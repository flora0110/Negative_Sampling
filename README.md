# Clustering-Exposure Balanced Sampling

## 90 vedio
1. What is the key motivation for your project? As in, why should we care about your problem space?
- 近期很多研究都顯示使用LLM作為推薦系統，仰賴LLM 在預訓練時讀過的大量文字，對商品有內建屬性等隱藏資訊，以及其可以更好的捕捉用戶的完整互動歷史、以及其推理能力，使LLM 可將其與歷史上相似行為的用戶做比對，進行類推，並可更有效地忽略雜訊資料
- 而微調LLM可以進一步讓模型學會「什麼樣的商品序列 → 對應何種推薦是有意義的」，例如SFT、DPO等技術都可以讓LLM偏好對齊推薦行為，但是這經常會造成過度推薦熱門產品、喪失多樣性和失去genra間的公平性等問題，而在進行DPO前，採用什麼方式做negative sampling會對結果影響很大，因此我們想找一個可以兼顧準確性、公平性、多樣性的negative sampling方法


2. What is your key insight? You should convey some "special sauce" that tells the audience what is so neat about your approach.
- 為了提升準確性，我們想要挑選到true negative sample，而因為原本長尾產品就已經很少出現在正樣本中，為了讓模型有機會學習到長尾產品的知識，我們也想要挑選曝光度較低的產品作為negative sample
- 同時因為我們想使用self play的方式，通過自我對弈來抑制模型對熱門或同質性項目的偏好，因此模型對於生成candidates的信心、生成candidates是否真的存在在資料庫中，也會影響結果
- 因此我們會使用Beam search加Constrained Decoding去生成負樣本 candidates，再和使用者的興趣集群做比較，並考慮產品曝光度去選擇負樣本，目前我們的方法，在和我們作為參考的論文方法（SPRec）在同樣使用小型模型(HuggingFaceTB/SmolLM2-1.7B-Instruct)進行微調後的結果，在NDCG@5、HR@5 、Diversity、DGU、MGU指標上都大幅進步至少一倍

| Model         | NDCG@5 ↑ | HR@5 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |
|------------|:---------:|:-------:|:-----------:|:----------:|:-------:|:-------:|:---------:|:------------:|
| SPRec      |  0.0066   |  0.008  |     447     |   0.0896   | 0.0758  | 0.0203  |  0.0912   |    0.671     |
| Our Method |  0.0235   |  0.036   |     837     |   0.1677    |  0.041    | 0.0103  |  0.1076   |    0.902     |


- Our Method:
  1. 使用SFT(只有正樣本)微調LLM(HuggingFaceTB/SmolLM2-1.7B-Instruct) 得到SFT-tuned LLM
  2. 使用SFT-tuned LLM 做 Beam search 加 Constrained Decoding 得到十個negative sample candidates
  3. 由使用者過去購買的物品建立使用者興趣集群（genras）
  4. 找出 不 在使用者興趣集群中的candidates中曝光度最低的作為clusterout_low_exposure neg_sample
  5. 找出在使用者興趣集群中的candidates中曝光度最低的作為clusterin_low_exposure neg_sample
  6. 使用clusterout_low_exposure neg_sample 對 SFT-tuned LLM 做 DPO, 得到 DPO-SFT-tuned LLM
  7. 使用clusterin_low_exposure neg_sample 對 DPO-SFT-tuned LLM 做 DPO, 得到最終模型


3. What did you discover or show through your project? You should tell us something interesting as a final takeaway.
-  因為計算指標的方式是將預測的結果去和資料庫中所有產品名稱計算相似度得到到的前topk個當作推薦結果（排序），所以NotInRatio雖然變高（LLM直接生成的產品不在dataset中），準確率還是可以提高

-  在實驗中我們同時有使用S-DPO(On Softmax Direct Preference Optimization for Recommendation)的方法，引入多個負樣本做DPO，但我們發現如果在使用不同方法取負樣本時，比起一次使用多個負樣本，如果先用簡單的負樣本做DPO，再使用困難的負樣本做一次DPO，效果會比較好

| Model                              | NDCG@5 ↑ | HR@5 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |
|------------|:---------:|:-------:|:-----------:|:----------:|:-------:|:-------:|:---------:|:------------:|
| S-DPO |  0.0108   |  0.017  |     582     |   0.1166   | 0.0569  | 0.0165  |   0.114   |    0.809     |
| curriculum learning |  0.0189   |  0.025  |     782     |   0.1567   |  0.04   | 0.0101  |  0.1461   |    0.943     |

- 比起使用topK生成candidates, 使用beam search生成candidates因為是模型有信心的答案，所以在抑制模型對熱門或同質性項目的偏好效果上更好，同時因為對模型來說也是比較困難的負樣本，因此也學習到比較細緻的分類方式，因此準確率上也有提升

| Model      | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |
|------------|:---------:|:-------:|:-----------:|:----------:|:-------:|:-------:|:---------:|:------------:|
| TopK(RN)   | 0.0077    | 0.012   | 694         |   0.0695   | 0.0603  | 0.0155  | 0.0620    | 0.652        |
| BeamSearch |  0.0187   |  0.032  |    1005     |   0.1007   | 0.0481  |  0.013  |  0.0709   |    0.796     |

- 原本我們是設定“在使用者集群內的高曝光度產品”是true negative sample，但實際上無論是在使用者興趣集群內或在使用者興趣集群外，選用的高曝光度產品度都會降低準確性，因此可以推斷高曝光度產品是高機率會是使用者潛在會選擇的產品，因此應該是歸類在false negative sample, 而在採用低曝光度的產品作為負樣本的前提下，採用使用者興趣集群外的產品會大幅提高準確率(neg_sampling_clusterout_low_exposure)，所以使用使用者興趣集群是可以有效過濾掉false negative sample的

| Model                              | NDCG@5 ↑ | HR@5 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |
|------------------------------------|:---------:|:-------:|:-----------:|:----------:|:-------:|:-------:|:---------:|:------------:|
| neg_sampling_clusterin_low_exposure |  0.0126   |  0.019  |     609     |   0.122    | 0.0654  | 0.0173  |  0.1074   |     0.79     |
| neg_sampling_clusterin_high_exposure |  0.0117   |  0.017  |     575     |   0.1152   | 0.0599  | 0.0171  |  0.1465   |    0.853     |
| neg_sampling_clusterout_low_exposure |  0.0162   |  0.024  |     651     |   0.1305   | 0.0586  | 0.0175  |   0.106   |    0.796     |
| neg_sampling_clusterout_high_exposure |  0.0101   |  0.015  |     584     |   0.117    | 0.0592  | 0.0166  |  0.1361   |    0.836     |
| neg_sampling_lowest_exposure          |  0.0126   |  0.019  |     615     |   0.1232   | 0.0643  | 0.0171  |  0.1058   |    0.784     |
- 同時我們也進行了消融實驗，不去做興趣集群，單純選擇candidates中曝光度最低的產品作為負樣本(neg_sampling_lowest_exposure)，可以看到準確性仍低於neg_sampling_clusterout_low_exposure許多，進一步證明了使用使用者興趣集群是可以有效過濾掉false negative sample




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

| Model                              | NDCG@5 ↑ | HR@5 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |
|------------------------------------|:---------:|:-------:|:-----------:|:----------:|:-------:|:-------:|:---------:|:------------:|
| SPRec          |  0.0066   |  0.008  |     447     |   0.0896   | 0.0758  | 0.0203  |  0.0912   |    0.671     |
| SPRec_wo_SFT(but DPO on SFT-tuned model)           |  0.0061   |  0.007  |     460     |   0.0922   | 0.0725  | 0.0198  |  0.0884   |    0.714     |

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
| neg_sampling_clusterin_low_exposure |  0.0145   |  0.025  |     968     |   0.097    | 0.0419  | 0.0126  |  0.0696   |     0.79     |
| neg_sampling_clusterin_high_exposure |  0.0133   |  0.022  |     915     |   0.0917   | 0.0407  | 0.0124  |   0.098   |    0.853     |
| neg_sampling_clusterout_low_exposure |  0.0187   |  0.032  |    1005     |   0.1007   | 0.0481  |  0.013  |  0.0709   |    0.796     |
| neg_sampling_clusterout_high_exposure |  0.0116   |  0.02   |     918     |   0.092    | 0.0393  | 0.0118  |  0.0887   |    0.836     |
| neg_sampling_clustering_exposure_balanced |  0.0121   |  0.021  |     914     |   0.0916   |  0.041  | 0.0118  |  0.0692   |    0.809     |
| neg_sampling_low_exposure          |  0.0145   |  0.025  |     968     |   0.097    | 0.0411  | 0.0123  |  0.0685   |    0.784     |
| neg_sampling_high_exposure         |  0.0133   |  0.022  |     915     |   0.0917   | 0.0407  | 0.0124  |   0.098   |    0.853     |
| neg_sampling_balanced_popularity   |  0.0123   |  0.021  |     909     |   0.0911   | 0.0398  |  0.012  |  0.0698   |    0.793     |
| clusterin_high_exposure_on_clusterout_low_exposure |  0.0202   |  0.029  |    1146     |   0.1148   | 0.0381  | 0.0104  |  0.1019   |    0.943     |
| clusterin_low_exposure_on_clusterout_low_exposure |  0.0263   |  0.045  |    1225     |   0.1227   | 0.0386  | 0.0107  |  0.0763   |    0.902     |

| Model                              | NDCG@5 ↑ | HR@5 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |
|------------------------------------|:---------:|:-------:|:-----------:|:----------:|:-------:|:-------:|:---------:|:------------:|
| neg_sampling_clusterin_low_exposure |  0.0126   |  0.019  |     609     |   0.122    | 0.0654  | 0.0173  |  0.1074   |     0.79     |
| neg_sampling_clusterin_high_exposure |  0.0117   |  0.017  |     575     |   0.1152   | 0.0599  | 0.0171  |  0.1465   |    0.853     |
| neg_sampling_clusterout_low_exposure |  0.0162   |  0.024  |     651     |   0.1305   | 0.0586  | 0.0175  |   0.106   |    0.796     |
| neg_sampling_clusterout_high_exposure |  0.0101   |  0.015  |     584     |   0.117    | 0.0592  | 0.0166  |  0.1361   |    0.836     |
| neg_sampling_clustering_exposure_balanced |  0.0108   |  0.017  |     582     |   0.1166   | 0.0569  | 0.0165  |   0.114   |    0.809     |
| neg_sampling_low_exposure          |  0.0126   |  0.019  |     615     |   0.1232   | 0.0643  | 0.0171  |  0.1058   |    0.784     |
| neg_sampling_high_exposure         |  0.0117   |  0.017  |     575     |   0.1152   | 0.0599  | 0.0171  |  0.1465   |    0.853     |
| neg_sampling_balanced_popularity   |   0.011   |  0.017  |     570     |   0.1142   | 0.0599  |  0.017  |  0.1104   |    0.793     |
| clusterin_high_exposure_on_clusterout_low_exposure |  0.0189   |  0.025  |     782     |   0.1567   |  0.04   | 0.0101  |  0.1461   |    0.943     |
| clusterin_low_exposure_on_clusterout_low_exposure |  0.0235   |  0.036  |     837     |   0.1677   |  0.041  | 0.0103  |  0.1076   |    0.902     |




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
