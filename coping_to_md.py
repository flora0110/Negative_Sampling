import glob
import json
import os

# =========================
# 參數設定，請依實際狀況修改
method_name = "Beam_Search_Negative_Generate_CD"
D = "Div"                           # 範例值，請自行修改
num_return_sequences = "10"         # 範例值，請自行修改
diversity_penalty = "2.0"              # 範例值，請自行修改

# 評估結果檔案所在的根目錄
base_dir = f"/scratch/user/chuanhsin0110/test_0321/output/{method_name}/{D}_{num_return_sequences}_{diversity_penalty}"
# 檔案搜尋 pattern: 各 sample_method 下的 metrics/eval_results@10_2.json
pattern = os.path.join(base_dir, "*/metrics/eval_results@10_2.json")

# =========================
# 取得所有符合 pattern 的檔案清單
file_list = glob.glob(pattern)
if not file_list:
    print("找不到符合 pattern 的檔案！")
    exit(1)

# =========================
# 收集所有結果的列 (rows)
rows = []
for file_path in file_list:
    with open(file_path, "r") as f:
        eval_data = json.load(f)
    # eval_data 是 list，每個 item 為一筆評估結果
    for record in eval_data:
        # 從 record 取得各欄位資訊，注意部分欄位為 list，取第一個元素
        model = record.get("sample_method", "N/A")
        ndcg_list = record.get("NDCG", [])
        ndcg_val = ndcg_list[0] if ndcg_list else ""
        hr_list = record.get("HR", [])
        hr_val = hr_list[0] if hr_list else ""
        diversity_list = record.get("diversity", [])
        diversity_val = diversity_list[0] if diversity_list else ""
        div_ratio = record.get("DivRatio", "")
        dgu = record.get("DGU", "")
        mgu = record.get("MGU", "")
        orratio = record.get("ORRatio", "")
        notin_ratio = record.get("Predict_NotIn_Ratio", "")

        rows.append({
            "Model": model,
            "NDCG@10": ndcg_val,
            "HR@10": hr_val,
            "Diversity": diversity_val,
            "DivRatio": div_ratio,
            "DGU": dgu,
            "MGU": mgu,
            "ORRatio": orratio,
            "NotInRatio": notin_ratio
        })

# =========================
# 產生 Markdown 表格
header = "| Model                              | NDCG@10 ↑ | HR@10 ↑ | Diversity ↑ | DivRatio ↑ | DGU ↓   | MGU ↓   | ORRatio ↓ | NotInRatio ↓ |"
separator = "|------------------------------------|:---------:|:-------:|:-----------:|:----------:|:-------:|:-------:|:---------:|:------------:|"
print(header)
print(separator)
for row in rows:
    # 依據欄位內容長度進行格式化（此處簡單以 f-string 的寬度格式）
    print(f"| {str(row['Model']):<34} | {str(row['NDCG@10']):^9} | {str(row['HR@10']):^7} | {str(row['Diversity']):^11} | {str(row['DivRatio']):^10} | {str(row['DGU']):^7} | {str(row['MGU']):^7} | {str(row['ORRatio']):^9} | {str(row['NotInRatio']):^12} |")
