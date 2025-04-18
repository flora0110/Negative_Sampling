import json
import re
import os
import numpy as np

# 配置
data_dir    = "/scratch/user/chuanhsin0110/test_0321/nlp"
output_dir  = "/scratch/user/chuanhsin0110/test_0321/nlp/dpo_data"
os.makedirs(output_dir, exist_ok=True)

# 预先导出的 item embeddings 和映射
# 假设你已将 LightGCN 的 item embedding 导出为 numpy 文件
item_emb_path  = os.path.join(data_dir, "models", "item_embeddings_full_data2.npy")
book2idx_path  = os.path.join(data_dir, "models", "book2idx_full_data2.json")

item_embeddings = np.load(item_emb_path)  # shape: (num_items, emb_dim)
with open(book2idx_path, "r") as f:
    book2idx = json.load(f)
# 建立反向字典
idx2book = {int(v): k for k, v in book2idx.items()}

def farthest_negative(input_books):
    # 将历史书籍映射到 embedding
    pos_idxs = [book2idx[b] for b in input_books if b in book2idx]
    if not pos_idxs:
        return None
    pos_embs = item_embeddings[pos_idxs]  # (P, D)
    # 候选集合：所有 idx 除去 pos_idxs
    all_idxs = set(range(item_embeddings.shape[0]))
    cand_idxs = np.array(list(all_idxs - set(pos_idxs)))
    # 计算每个候选与历史的最小欧氏距离
    # 距离矩阵: (C, P)
    diffs = item_embeddings[cand_idxs][:, None, :] - pos_embs[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)  # (C, P)
    min_dists = dists.min(axis=1)         # (C,)
    # 选最大
    idx = cand_idxs[min_dists.argmax()]
    return idx2book[idx]

def process_split(split_name):
    in_path  = os.path.join("/scratch/user/chuanhsin0110/test_0321/sampled_data", f"{split_name}_sample.json")
    out_path = os.path.join(output_dir, f"{split_name}_dpo.json")
    with open(in_path, "r", encoding="utf8") as f:
        data = json.load(f)
    results = []
    for entry in data:
        # 提取书名
        books = re.findall(r'"([^"]+)"', entry.get("input", ""))
        neg = farthest_negative(books)
        results.append({
            "instruction": entry["instruction"],
            "input": entry["input"],
            "chosen": entry["output"].strip(),
            "rejected": neg or ""
        })
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {split_name} negatives to {out_path}")

if __name__ == "__main__":
    process_split("train")
    process_split("valid")
