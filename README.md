# BASELINE
## result
| Model                          | NDCG   | HR    | Diversity | DivRatio | DGU   | MGU   | ORRatio |
|-------------------------------|:------:|:-----:|:--------:|:-------:|:-----:|:-----:|:------:|
| molLM2-1.7B-Instruct (origin model) | 0.0038 | 0.007 |   510    | 0.0511  | 0.0837| 0.0179| 0.1254 |
| smolLM2-1.7B-lora-run3 (SFT) | 0.0043 | 0.010 |   608    | 0.0609  | 0.0615| 0.0163| 0.0707 |
| smolLM2-1.7B-lora-dpo-run3 (DPO) | 0.0025 | 0.006 |   735    | 0.0736  | 0.0747| 0.0148| 0.0868 |

## code
### traning
- smolLM2-1.7B-lora-run3 (SFT): sft_smol.py
- smolLM2-1.7B-lora-dpo-run3 (DPO): DPO3.py