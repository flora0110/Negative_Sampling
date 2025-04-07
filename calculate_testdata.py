import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# 讀取兩個 true_negative_data_details 檔案
with open("/scratch/user/chuanhsin0110/test_0321/test/output/True_Negative_Sampling_run1/data/true_negative_data_details_Origin.json") as f:
    origin_data = json.load(f)

with open("/scratch/user/chuanhsin0110/test_0321/test/output/True_Negative_Sampling_run1/data/true_negative_data_details_SFT.json") as f:
    sft_data = json.load(f)

def get_true_candidate_name(candidate):
    if candidate.startswith("**") and "->" in candidate:
        # 取出箭頭後面的部分
        # return candidate.split("->")[-1].strip()
        return ""
    else:
        return candidate.strip()

def extract_negatives(data):
    candidates = []
    genres = []
    exposures = []
    empty_count = 0  # 新增：計算空字串的數量

    for d in data:
        for item in d["neg_genra"]:
            name = get_true_candidate_name(item["candidate"])
            candidates.append(name)
            if name != "":
                exposures.append(item.get("exposure_count", 0))
                genres.extend(item.get("genre", []))
            else:
                empty_count += 1
                exposures.append(0)

    empty_ratio = empty_count / len(candidates) if candidates else 0
    return candidates, exposures, genres, empty_ratio


def calculate_metrics(candidates, exposures, genres, empty_ratio=0.0, threshold=50):
    n_total = len(candidates)
    unique_candidates = set(candidates)
    unique_genres = set(genres)

    div_ratio = len(unique_candidates) / n_total if n_total > 0 else 0
    dgu = len(genres) / n_total if n_total > 0 else 0
    mgu = len(unique_genres) / len(genres) if len(genres) > 0 else 0
    overexposed_count = sum(1 for e in exposures if e > threshold)
    orratio = overexposed_count / n_total if n_total > 0 else 0

    return {
        "DiversityRatio": div_ratio,
        "DGU": dgu,
        "MGU": mgu,
        "ORRatio": orratio,
        "AvgExposure": sum(exposures) / n_total if n_total > 0 else 0,
        "EmptyCandidateRatio": empty_ratio  # 新增：空書名比例
    }



# 讀取資料集
with open("/scratch/user/chuanhsin0110/test_0321/sampled_data/train_sample.json") as f:
    train_data = json.load(f)

# 計算書名曝光次數
counter = Counter()
for d in train_data:
    book_names = re.findall(r'"(.*?)"', d["input"])
    counter.update(book_names)

# 將曝光次數排序並取出數值
exposure_counts = [count for _, count in counter.most_common()]

# 畫曝光次數的長尾分布圖（log-log）
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(exposure_counts) + 1), exposure_counts)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Book Rank (log)")
plt.ylabel("Exposure Count (log)")
plt.title("Book Exposure Distribution (Log-Log)")
plt.grid(True)
plt.tight_layout()
plt.savefig("test/book_exposure_distribution.png")

# 統計一些常用分位數門檻
thresholds = {
    "Top 1% threshold": np.percentile(exposure_counts, 99),
    "Top 5% threshold": np.percentile(exposure_counts, 95),
    "Top 10% threshold": np.percentile(exposure_counts, 90),
    "Top 25% threshold": np.percentile(exposure_counts, 75),
}

print(thresholds)

# 提取與分析
origin_candidates, origin_exposures, origin_genres, origin_empty_ratio = extract_negatives(origin_data)
sft_candidates, sft_exposures, sft_genres, sft_empty_ratio = extract_negatives(sft_data)

origin_metrics = calculate_metrics(origin_candidates, origin_exposures, origin_genres, origin_empty_ratio)
sft_metrics = calculate_metrics(sft_candidates, sft_exposures, sft_genres, sft_empty_ratio, thresholds["Top 5% threshold"])


# 彙整為 DataFrame 呈現
comparison_df = pd.DataFrame([origin_metrics, sft_metrics], index=["Origin", "SFT"])
print("True Negative Diversity Metrics:")
print(comparison_df.to_string())
print("\n📘 指標說明：")
print("DiversityRatio：不同 negative 書目的數量 / 全部數量，代表候選的多樣性（越高越好）")
print("DGU（Diversity in Genre Utilization）：平均每個候選含有幾個 genre，越高表示每本書的 genre 描述越豐富（中等偏高較好）")
print("MGU（Mean Genre Uniqueness）：候選書目的 genre 中，有多少比例是「獨特」的 genre（越高越多樣）")
print("ORRatio（Overexposure Ratio）：候選中曝光次數超過 50 的比例，越低越好（避免過度推薦熱門書）")
print("AvgExposure：平均曝光次數，反映推薦書的整體熱門程度，過高代表熱門項偏好，越低表示偏冷門")
