import torch
import numpy as np
import os

# 1. LightGCN state_dict 路徑
pth = "/scratch/user/chuanhsin0110/test_0321/nlp/models/lightgcn_from_full_data2.pth"

# 2. 讀取 state_dict
state = torch.load(pth, map_location="cpu")

# 3. 列出所有 key 方便確認
print("STATE_DICT KEYS:")
for k in state.keys():
    print("  ", k)

# 4. 正确的 item embedding key
item_key = "i_emb.weight"  # 之前是 "item_emb.weight"，改成這個

# 5. 取出 embedding tensor
if item_key not in state:
    raise KeyError(f"{item_key} not in state_dict!")
item_emb = state[item_key].cpu().numpy()
print(f"Loaded item embeddings with shape {item_emb.shape}")

# 6. 儲存成 .npy
out = "/scratch/user/chuanhsin0110/test_0321/nlp/models/item_embeddings_full_data2.npy"
os.makedirs(os.path.dirname(out), exist_ok=True)
np.save(out, item_emb)
print(f"Saved item embeddings to {out}")

num_items, emb_dim = item_emb.shape
print(f"Number of item embeddings: {num_items}")
print(f"Embedding dimension: {emb_dim}")