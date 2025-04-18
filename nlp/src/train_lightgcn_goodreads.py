import json
import re
import os
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix

# 1. 設定檔案路徑
data_dir      = "/scratch/user/chuanhsin0110/test_0321/nlp"
train_json    = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/train.json"
valid_json    = "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/valid.json"
model_out_dir = os.path.join(data_dir, "models")
os.makedirs(model_out_dir, exist_ok=True)
model_save    = os.path.join(model_out_dir, "lightgcn_from_full_data2.pth")

# 2. 讀 JSON 並抽取 (user_idx, book_title) 互動
def load_interactions(json_path, user_offset=0):
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    inter = []
    for idx, entry in enumerate(data):
        user_id = idx + user_offset
        titles = re.findall(r'"([^"]+)"', entry.get('input', ''))
        for t in titles:
            inter.append((user_id, t.strip()))
    return inter, len(data)

train_inter, n_train = load_interactions(train_json, user_offset=0)
valid_inter, n_valid = load_interactions(valid_json, user_offset=n_train)

interactions = train_inter + valid_inter
num_users   = n_train + n_valid

# 3. 書籍編號映射
# 3.1 把所有書名收集好之後，印一印
all_books = {t for _, t in interactions}
print(f"→ Found {len(all_books)} unique book titles in train+valid interactions")

# 3.2 接着建立 book2idx
book2idx  = {b:i for i,b in enumerate(sorted(all_books))}
num_items = len(book2idx)
print(f"→ book2idx size = {num_items}")

# 4. 轉成 (user_idx, item_idx) 並建立查詢集合
inter_idx = [(u, book2idx[t]) for u, t in interactions]
inter_set = set(inter_idx)

# 5. 建正規化鄰接矩陣
def build_norm_adj(inter_idx, U, I):
    rows, cols, data = [], [], []
    for u, i in inter_idx:
        rows += [u, i+U]
        cols += [i+U, u]
        data += [1.0, 1.0]
    A = coo_matrix((data, (rows, cols)), shape=(U+I, U+I))
    d = np.array(A.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5, where=d>0)
    D = coo_matrix((d_inv_sqrt, (np.arange(U+I), np.arange(U+I))), shape=A.shape)
    A_norm = D.dot(A).dot(D).tocoo()
    idx = torch.LongTensor(np.vstack((A_norm.row, A_norm.col)))
    val = torch.FloatTensor(A_norm.data)
    return torch.sparse.FloatTensor(idx, val, torch.Size(A_norm.shape))

norm_adj = build_norm_adj(inter_idx, num_users, num_items)

# 6. 定義 LightGCN
class LightGCN(nn.Module):
    def __init__(self, U, I, emb_dim, n_layers, adj):
        super().__init__()
        self.U = U; self.I = I; self.layers = n_layers
        self.adj = adj
        self.u_emb = nn.Embedding(U, emb_dim)
        self.i_emb = nn.Embedding(I, emb_dim)
        nn.init.normal_(self.u_emb.weight, std=0.1)
        nn.init.normal_(self.i_emb.weight, std=0.1)

    def forward(self):
        x = torch.cat([self.u_emb.weight, self.i_emb.weight], dim=0)
        embs = [x]
        for _ in range(self.layers):
            x = torch.sparse.mm(self.adj, x)
            embs.append(x)
        final = torch.stack(embs, dim=1).mean(dim=1)
        return final[:self.U], final[self.U:]

# BPR Loss
def bpr(u, pos, neg):
    pos_s = torch.sum(u*pos, dim=1)
    neg_s = torch.sum(u*neg, dim=1)
    return -torch.log(torch.sigmoid(pos_s - neg_s) + 1e-8).mean()

# 7. 訓練
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LightGCN(num_users, num_items, emb_dim=64, n_layers=3, adj=norm_adj).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs, bs = 5, 1024

for ep in range(epochs):
    model.train()
    total, cnt = 0.0, 0
    np.random.shuffle(inter_idx)
    for st in range(0, len(inter_idx), bs):
        batch = inter_idx[st:st+bs]
        u = torch.LongTensor([x[0] for x in batch]).to(device)
        p = torch.LongTensor([x[1] for x in batch]).to(device)
        # sample negative
        n = []
        for ui in u.cpu().tolist():
            ni = np.random.randint(num_items)
            while (ui,ni) in inter_set:
                ni = np.random.randint(num_items)
            n.append(ni)
        n = torch.LongTensor(n).to(device)

        ue, ie = model()
        loss = bpr(ue[u], ie[p], ie[n])
        opt.zero_grad(); loss.backward(); opt.step()

        total += loss.item() * len(batch); cnt += len(batch)
    print(f"Epoch {ep+1}/{epochs}, Loss: {total/cnt:.4f}")

# 8. 儲存模型
torch.save(model.state_dict(), model_save) 
print("Saved LightGCN model to", model_save)
