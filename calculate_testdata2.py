import json
import argparse
from collections import Counter
import numpy as np

def get_true_candidate_name(candidate: str) -> str:
    """
    若 candidate 字串以 "**" 起頭且含有 "->" 則回傳空字串，
    否則回傳去除前後空白的 candidate 名稱。
    """
    if candidate.startswith("**") and "->" in candidate:
        return ""
    else:
        return candidate.strip()

def extract_record_candidates(record: dict):
    """
    從單筆資料中取出 neg_genra 欄位，依序取得：
      - candidates：候選書目名稱（經過 get_true_candidate_name）
      - exposures：候選的曝光次數（exposure_count）
      - genres_list：每個候選項的 genre 列表
    """
    candidates = []
    exposures = []
    genres_list = []
    for item in record.get("neg_genra", []):
        name = get_true_candidate_name(item.get("candidate", ""))
        candidates.append(name)
        exposures.append(item.get("exposure_count", 0))
        genres = item.get("genre", [])
        genres_list.append(genres)
    return candidates, exposures, genres_list

def process_dataset(data: list) -> dict:
    """
    對資料集（list of records）計算各項指標：
      - EmptyCandidateRatio：所有候選中空字串的比例
      - AvgUniqueCandidatesPerRecord：每筆資料中獨特 candidate 的數量平均值
      - GenreDiversityUtilization：每筆資料中候選項涵蓋的不同 genre 數量（取 union）之平均值
      - CandidateDiversityRatio：全資料中不同 candidate 的數量 / 總候選數
      - HighExposureRatio：候選中曝光次數大於 50 的比例
      - AvgExposure：所有候選的平均曝光次數
      - DGU：每筆資料中候選的 genre 分佈與全資料理想分佈差異中，最大與最小差值的差的平均值
      - MGU：每筆資料中候選的 genre 分佈與理想分佈的平均絕對偏差
    """
    total_empty = 0
    total_candidates = 0
    unique_candidate_counts = []       # 每筆資料中不同 candidate 數
    genre_diversity_counts = []        # 每筆資料中候選項涵蓋的不同 genre 數（取 union）
    exposures_all = []                 # 全資料所有候選的曝光次數
    high_exposure_count = 0            # 曝光次數 > 50 的候選個數
    record_candidates_list = []        # 用於計算全資料不同 candidate 數
    record_genre_counters = []         # 每筆資料中 candidate 的 genre 計數（每個 candidate 有 n 個 genre 則每個 genre 貢獻 1/n）

    # 遍歷每筆資料
    for record in data:
        candidates, exposures, genres_list = extract_record_candidates(record)
        total_candidates += len(candidates)
        empty_count = sum(1 for c in candidates if c == "")
        total_empty += empty_count
        record_candidates_list.append(candidates)
        
        # 計算每筆資料中候選項的獨特性
        unique_count = len(set(candidates))
        unique_candidate_counts.append(unique_count)
        
        # 統計該筆資料中所有候選涵蓋的 genre（取 union）
        record_genres = set()
        # 同時累計該筆資料中每個 genre 的權重
        record_counter = Counter()
        for genres in genres_list:
            if genres:
                for g in genres:
                    record_genres.add(g)
                    record_counter[g] += 1 / len(genres)
        genre_diversity_counts.append(len(record_genres))
        record_genre_counters.append(record_counter)
        
        exposures_all.extend(exposures)
        high_exposure_count += sum(1 for x in exposures if x > 50)
    
    # Empty candidate ratio
    empty_candidate_ratio = total_empty / total_candidates if total_candidates else 0
    # 平均每筆資料中獨特候選的數量
    avg_unique_candidates = np.mean(unique_candidate_counts) if unique_candidate_counts else 0
    # 平均每筆資料涵蓋的 genre 數
    avg_genre_diversity = np.mean(genre_diversity_counts) if genre_diversity_counts else 0
    # 全資料中不同 candidate 的數量與總候選數的比例
    all_candidates = [cand for rec in record_candidates_list for cand in rec]
    candidate_diversity_ratio = len(set(all_candidates)) / total_candidates if total_candidates else 0
    # 曝光次數大於 50 的比例與平均曝光次數
    high_exposure_ratio = high_exposure_count / total_candidates if total_candidates else 0
    avg_exposure = np.mean(exposures_all) if exposures_all else 0

    # 計算全資料「理想」的 genre 分佈（加權累計後再正規化）
    global_genre_counter = Counter()
    for counter in record_genre_counters:
        global_genre_counter.update(counter)
    total_global = sum(global_genre_counter.values())
    if total_global > 0:
        ideal_genre_dist = {g: count/total_global for g, count in global_genre_counter.items()}
    else:
        ideal_genre_dist = {}

    # 針對每筆資料計算該筆資料的 genre 分佈與理想分佈的差異
    record_dgu = []  # 每筆資料 DGU
    record_mgu = []  # 每筆資料 MGU
    for counter in record_genre_counters:
        total_record = sum(counter.values())
        # 若該筆資料沒有 genre，則設定差異為 0
        if total_record == 0:
            record_dist = {g: 0 for g in ideal_genre_dist}
        else:
            record_dist = {g: counter.get(g, 0)/total_record for g in ideal_genre_dist}
        # 差值向量：針對每個 genre，差值 = 該筆資料比例 - 理想比例
        diffs = [record_dist[g] - ideal_genre_dist[g] for g in ideal_genre_dist]
        if diffs:
            dgu = max(diffs) - min(diffs)
            mgu = np.mean(np.abs(diffs))
        else:
            dgu = 0
            mgu = 0
        record_dgu.append(dgu)
        record_mgu.append(mgu)
    avg_dgu = np.mean(record_dgu) if record_dgu else 0
    avg_mgu = np.mean(record_mgu) if record_mgu else 0

    metrics = {
        "EmptyCandidateRatio": empty_candidate_ratio,
        "AvgUniqueCandidatesPerRecord": avg_unique_candidates,
        "GenreDiversityUtilization": avg_genre_diversity,
        "CandidateDiversityRatio": candidate_diversity_ratio,
        "HighExposureRatio": high_exposure_ratio,
        "AvgExposure": avg_exposure,
        "DGU": avg_dgu,
        "MGU": avg_mgu
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="計算兩組 true_negative 資料的各項指標")
    parser.add_argument("--origin", type=str,
                        default="/scratch/user/chuanhsin0110/test_0321/test/output/True_Negative_Sampling_run1/data/true_negative_data_details_Origin.json",
                        help="Origin 資料的 JSON 路徑")
    parser.add_argument("--sft", type=str,
                        default="/scratch/user/chuanhsin0110/test_0321/test/output/True_Negative_Sampling_run1/data/true_negative_data_details_SFT.json",
                        help="SFT 資料的 JSON 路徑")
    args = parser.parse_args()

    # 讀取兩個 JSON 檔案
    with open(args.origin, "r") as f:
        origin_data = json.load(f)
    with open(args.sft, "r") as f:
        sft_data = json.load(f)

    # 分別計算指標
    origin_metrics = process_dataset(origin_data)
    sft_metrics = process_dataset(sft_data)

    # 指標說明字典
    metric_descriptions = {
        "EmptyCandidateRatio": "所有候選中空字串的比例",
        "AvgUniqueCandidatesPerRecord": "平均每筆資料中獨特候選書目的數量",
        "GenreDiversityUtilization": "平均每筆資料中候選項涵蓋的不同 genre 數量",
        "CandidateDiversityRatio": "不同 candidate 書目的數量 / 全部候選數，代表候選的多樣性",
        "HighExposureRatio": "候選中曝光次數超過 50 的比例",
        "AvgExposure": "候選的平均曝光次數",
        "DGU": "候選的 genre 分佈與全資料理想分佈差異中，最大與最小差值的差",
        "MGU": "平均每筆資料中候選項的 genre 分佈與理想分佈的平均絕對偏差"
    }

    # 輸出格式：依指標逐項印出
    for metric in metric_descriptions:
        print(f"指標名稱: {metric}")
        print(f"指標說明: {metric_descriptions[metric]}")
        print(f"Origin 資料result: {origin_metrics.get(metric, 'N/A')}")
        print(f"SFT 資料result: {sft_metrics.get(metric, 'N/A')}")
        print("-" * 40)

if __name__ == "__main__":
    main()
