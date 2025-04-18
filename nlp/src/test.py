import json
import numpy as np

# 1. 載入映射和 embeddings
book2idx = json.load(open("/scratch/user/chuanhsin0110/test_0321/nlp/models/book2idx.json", "r", encoding="utf8"))
emb = np.load("/scratch/user/chuanhsin0110/test_0321/nlp/models/item_embeddings_full_data.npy")  # shape = (N, D)
N, D = emb.shape

# 2. 找出缺失的 idx
missing = [title for title, idx in book2idx.items() if idx >= N]
print(f"共有 {len(missing)} 本書缺少 embedding，例如：")
print(missing[:10])
