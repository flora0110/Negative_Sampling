import json
import re

in_files = [
#   "/scratch/user/chuanhsin0110/test_0321/sampled_data/train_sample.json",
#   "/scratch/user/chuanhsin0110/test_0321/sampled_data/valid_sample.json",
    "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/train.json",
    "/scratch/user/chuanhsin0110/SPRec/data/Goodreads/valid.json"
]

book_set = set()
for jf in in_files:
    data = json.load(open(jf, "r", encoding="utf8"))
    for entry in data:
        books = re.findall(r'"([^"]+)"', entry["input"])
        # books += re.findall(r'"([^"]+)"', entry["output"])
        book_set.update(b.strip() for b in books)
book2idx = {b:i for i,b in enumerate(sorted(book_set))}
with open("/scratch/user/chuanhsin0110/test_0321/nlp/models/book2idx_full_data2.json","w", encoding="utf8") as f:
    json.dump(book2idx, f, ensure_ascii=False, indent=2)
print("Saved book2idx.json with", len(book2idx), "entries")
