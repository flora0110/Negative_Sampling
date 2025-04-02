# BASELINE
## result
| Model                              | NDCG   | HR    | Diversity | DivRatio | DGU   | MGU   | ORRatio |
|-----------------------------------|:------:|:-----:|:--------:|:-------:|:-----:|:-----:|:------:|
| smolLM2-1.7B-Instruct (origin model) | 0.0038 | 0.007 |   510    | 0.0511  | 0.0837| 0.0179| 0.1254 |
| smolLM2-1.7B-lora-run3 (SFT)        | 0.0043 | 0.010 |   608    | 0.0609  | 0.0615| 0.0163| 0.0707 |
| smolLM2-1.7B-lora-dpo-run3 (DPO)    | 0.0025 | 0.006 |   735    | 0.0736  | 0.0747| 0.0148| 0.0868 |
| smolLM2-1.7B-lora-dpo-run6 (DPO)    | 0.0031 | 0.007 |   542    | 0.0543  | 0.0797| 0.0172| 0.1176 |
| SPRec_wo_STF_run2                   | 0.0028 | 0.006 |   647    | 0.0648  | 0.0721| 0.0165| 0.0738 |
| SPRec_run1 (SFT+DPO)                | 0.0032 | 0.007 | 608       | 0.0609   | 0.0797 | 0.0172 | 0.1004  |



## code
### traning
- smolLM2-1.7B-lora-run3 (SFT): sft_smol.py
- smolLM2-1.7B-lora-dpo-run3 (DPO): DPO4.py - LossThresholdCallback(threshold=0.1)
- smolLM2-1.7B-lora-dpo-run6 (DPO): DPO5.py
- SPRec_wo_STF_run2  : dataset_generate.py $\rightarrow$ DPO_from_dpoData.py
- SPRec_run1 (SFT+DPO)  : dataset_generate.py (with smolLM2-1.7B-lora-run3 model) $\rightarrow$ DPO_from_dpoData.py