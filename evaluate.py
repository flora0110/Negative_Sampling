import os
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
import re
import math
import json
from peft import PeftModel
import argparse
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm



# parse = argparse.ArgumentParser()
# parse.add_argument("--input_dir",type=str, default="./", help="result file")
# parse.add_argument("--model",type=str, default="SPRec", help="result file")
# parse.add_argument("--exp_csv",type=str, default=None, help="result file")
# parse.add_argument("--output_dir",type=str, default="./", help="eval_result")
# parse.add_argument("--topk",type=str, default="./", help="topk")
# parse.add_argument("--gamma",type=float,default=0.0,help="gamma")
# parse.add_argument("--category",type=str,default="CDs_and_Vinyl",help="gamma")
# args = parse.parse_args()



def read_json(json_file:str) -> dict:
    f = open(json_file, 'r')
    return json.load(f)
category = "Goodreads"
id2name = read_json(f"./eval/{category}/id2name.json")
name2id = read_json(f"./eval/{category}/name2id.json")
embeddings = torch.load(f"./eval/{category}/embeddings.pt")
name2genre = read_json(f"./eval/{category}/name2genre.json")
genre_dict = read_json(f"./eval/{category}/genre_dict.json")
def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

def sum_of_first_i_keys(sorted_dic, i):
    keys = list(sorted_dic.values())[:i]
    return sum(keys)

def gh(category:str,test_data):
    notin_count = 0
    in_count = 0
    name2genre=read_json(f"./eval/{category}/name2genre.json")
    genre_dict = read_json(f"./eval/{category}/genre_dict.json")
    for data in tqdm(test_data,desc="Processing category data......"):
        input = data['prompt']
        names = re.findall(r'"([^"]+)"', input)
        for name in names:
            if name in name2genre:
                in_count += 1
                genres = name2genre[name]
            else:
                notin_count += 1
                # print(f"Not exist in name2genre:{name}")
                continue
            select_genres = []
            for genre in genres:
                if genre in genre_dict:
                        select_genres.append(genre)
            if(len(select_genres)>0):
                for genre in select_genres:
                    genre_dict[genre] += 1/len(select_genres)
    gh = [genre_dict[x] for x in genre_dict]
    gh_normalize = [x/sum(gh) for x in gh]
    print(f"InCount:{in_count}\nNotinCount:{notin_count}")
    return gh_normalize
    
def update_csv(dataset_name, model_name, metrics_dict, csv_file):
    # 如果檔案不存在 → 建立一個新的 DataFrame
    if not os.path.exists(csv_file):
        # 初始化欄位
        df = pd.DataFrame(columns=["Dataset", "Model"] + list(metrics_dict.keys()))
        # 新增第一行
        new_row = {"Dataset": dataset_name, "Model": model_name}
        new_row.update(metrics_dict)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # 檔案存在 → 讀取舊檔
        df = pd.read_csv(csv_file)
        condition = (df["Dataset"] == dataset_name) & (df["Model"] == model_name)
        # 如果這組沒有出現過 → 新增一行
        if not condition.any():
            new_row = {col: None for col in df.columns}
            new_row["Dataset"] = dataset_name
            new_row["Model"] = model_name
            for metric, value in metrics_dict.items():
                if metric not in df.columns:
                    df[metric] = None  # 新增欄位
                new_row[metric] = value
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # 如果這組已存在 → 更新欄位
            for metric, value in metrics_dict.items():
                if metric not in df.columns:
                    df[metric] = None
                df.loc[condition, metric] = value

    # 最後存檔
    df.to_csv(csv_file, index=False)
    print(f"CSV 已更新：{csv_file}")

if __name__ == "__main__":
    print("start")
    # input_dir = "./results/eval_lora_raw_results.json"
    input_dir = "./results/eval_baseline_raw_results.json"
    
    # model_name = "STF_tuned_1024"
    model_name = "origin_opt-350m"
    # exp_csv = "./STF_eval_metrics.csv"
    # output_dir = "./STF_eval_output.json"
    exp_csv = "./origin_eval_metrics.csv"
    output_dir = "./origin_eval_output.json"
    topk = 5
    category = "Goodreads"

    result_json = input_dir
    f = open(result_json, 'r')
    test_data = json.load(f)
    total = 0
    # Identify your sentence-embedding model
    print("loading model")
    model = SentenceTransformer('./models/paraphrase-MiniLM-L3-v2')
    print("load model successfully")
    
    embeddings = torch.tensor(embeddings).cuda()
    text = []
    for i,_ in tqdm(enumerate(test_data)):
        if(len(_["prediction"])>0):
            if(len(_['prediction'][0])==0):
                text.append("NAN")
                print("Empty prediction!")
            else:
                match = re.search(r'"([^"]*)', _['prediction'][0])
                if match:
                    name = match.group(1)
                    text.append(name)
                else:
                    text.append(_['prediction'][0].split('\n', 1)[0])
        else:
            print("Empty:")

    predict_embeddings = []
    for i, batch_input in tqdm(enumerate(batch(text, 8))):
        predict_embeddings.append(torch.tensor(model.encode(batch_input)))
    predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()
    predict_embeddings.size()
    dist = torch.cdist(predict_embeddings, embeddings, p=2)
    batch_size = 1
    num_batches = (dist.size(0) + batch_size - 1) // batch_size  
    rank_list = []  
    for i in tqdm(range(num_batches), desc="Processing Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, dist.size(0))  
        batch_dist = dist[start_idx:end_idx]

        batch_rank = batch_dist.argsort(dim=-1).argsort(dim=-1)
        torch.cuda.empty_cache ()
        rank_list.append(batch_rank)

    rank_list = torch.cat(rank_list, dim=0)

    NDCG = []
    HR = []
    diversity = []
    diversity_dic = {}
    MGU_genre = []
    DGU_genre = []
    pop_count = {}
    genre_count = {}
    notin = 0
    notin_count = 0
    in_count = 0
    topk_list = [topk]
    diversity_set = set()
    for topk in topk_list:
        S_ndcg = 0
        S_hr = 0
        for i in tqdm(range(len(test_data)),desc="Calculating Metrics......"):
            rank = rank_list[i]
            # Target id
            target_name = test_data[i]['ground_truth']
            predict_name = test_data[i]['prediction'][0]
            target_name = target_name.strip().strip('"')
            if target_name in name2id:
                target_id = name2id[target_name]
                total += 1
            else:
                continue
                    
            rankId = rank[target_id]

            # NDCG & HR
            if(rankId<topk):
                S_ndcg += (1 / math.log(rankId + 2))
                S_hr += 1
            
            # Popularity bias & genre fairness
            for i in range(topk): # calculated only once
                topi_id = torch.argwhere(rank==i).item()
                topi_name = id2name[str(topi_id)]
                #pop = id2group[str(topi_id)]
                #pop_count[str(pop)] = 1 + pop_count[str(pop)]
                if topi_name in name2genre:
                    topi_genre = name2genre[topi_name]
                    select_genres = []
                    for genre in topi_genre:
                        if genre in genre_dict:
                            select_genres.append(genre)
                    if(len(select_genres)>0):
                        for genre in select_genres:
                            genre_dict[genre] += 1/len(select_genres)
                else:
                    notin += 1


            # diversity
            for i in range(topk):
                diversity_set.add(torch.argwhere(rank==i).item())
                if torch.argwhere(rank==i).item() in diversity_dic:
                    diversity_dic[torch.argwhere(rank==i).item()] += 1
                else:
                    diversity_dic[torch.argwhere(rank==i).item()] = 1


        NDCG.append(S_ndcg / len(test_data) / (1 / math.log(2)))
        HR.append(S_hr / len(test_data))
        diversity.append(len(diversity_set))
    genre = category

    gh_genre = gh(category,test_data)
    
    print(len(gh_genre))
    gp_genre = [genre_dict[x] for x in genre_dict]
    gp_genre = [x/sum(gp_genre) for x in gp_genre]
    dis_genre = [gp_genre[i]-gh_genre[i] for i in range(len(gh_genre))]
    DGU_genre = max(dis_genre)-min(dis_genre)
    dis_abs_genre = [abs(x) for x in dis_genre]
    MGU_genre = sum(dis_abs_genre) / len(dis_genre)
    i=0

    gp_dict = {}
    i=0
    for key in genre_dict:
        gp_dict[key] = dis_abs_genre[i]
        i += 1
    print(f"gp_dict:{gp_dict}")
    print(f"NDCG:{NDCG}")
    print(f"HR:{HR}")
    div_ratio = diversity[0] / (total*topk)
    print(f"DGU:{DGU_genre}")
    print(f"MGU:{MGU_genre}")
    print(f"DivRatio:{div_ratio}")

    eval_dic = {}
    # eval_dic["model"] = input_dir
    eval_dic["model"] = model_name
    eval_dic["Dis_genre"] = dis_abs_genre
    eval_dic['NDCG'] = NDCG
    eval_dic["HR"] = HR
    eval_dic["diversity"] = diversity
    eval_dic["DivRatio"] = div_ratio
    eval_dic['DGU'] = DGU_genre
    eval_dic["MGU"] = MGU_genre

    file_path = "./eval/Goodreads/eval_result.json"
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    sorted_dic = dict(sorted(diversity_dic.items(), key=lambda item: item[1],reverse=True))
    count = 0
    i=0
    eval_dic["ORRatio"] = sum_of_first_i_keys(sorted_dic,3) / (topk*total)
    print(f"ORRatio:{sum_of_first_i_keys(sorted_dic,3) / (topk*total)}")
    print(dict(sorted(diversity_dic.items(), key=lambda item: item[1])))
    data.append(eval_dic)
    print(count)
    with open(output_dir, 'w') as file:
        json.dump(data, file,separators=(',', ': '),indent=2)

    

    if exp_csv != None:
        metric_dic = {}
        metric_dic[f"MGU@{topk}"] = eval_dic["MGU"]
        metric_dic[f"DGU@{topk}"] = eval_dic["DGU"]
        metric_dic[f"DivRatio@{topk}"] = eval_dic["DivRatio"]
        metric_dic[f"ORRatio@{topk}"] = sum_of_first_i_keys(sorted_dic,3) / (topk*total)
        if topk == '5':
            metric_dic[f"NDCG@{topk}"] = eval_dic["NDCG"]
        update_csv(category,model_name,metric_dic,exp_csv)