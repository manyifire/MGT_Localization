from pathlib import Path
import json

src_path = Path("hybridCodeData_mixed.jsonl")

records = []
with src_path.open("r", encoding="utf‑8") as fp:
    for line in fp:              # 每行都是一个独立 JSON 对象
        line = line.strip()
        if not line:
            continue             
        records.append(json.loads(line))

def transform(rec: dict) -> dict:
    merge_sentences = rec.get('hybrid_code').split('\n')  # 按行分割代码

    n = len(merge_sentences)            # 总长度

    # 0: human 1: machine
    if rec.get('prompt_pattern') == "H_M":
        zero_len = rec.get("boundary_ix")[0]
        mixed_labels =  [0] * zero_len + [1] * (n - zero_len)
    if rec.get('prompt_pattern') == "M_H":
        zero_len = rec.get("boundary_ix")[0]
        if zero_len == None:
            zero_len = 0
        mixed_labels =  [1] * zero_len + [0] * (n - zero_len)  
    if rec.get('prompt_pattern') == "H_M_H":
        zero_len = rec.get("boundary_ix")[0]
        one_len = rec.get("boundary_ix")[1]
        mixed_labels =  [0] * zero_len + [1] * (one_len - zero_len) + [0] * (n - one_len)
    if rec.get('prompt_pattern') == "M_H_M":
        zero_len = rec.get("boundary_ix")[0]
        one_len = rec.get("boundary_ix")[1]
        if zero_len == None or one_len == None:
            zero_len = 0
            one_len = n
        mixed_labels =  [1] * zero_len + [0] * (one_len - zero_len) + [1] * (n - one_len)

    new_rec = {
        "article_id": rec.get("code_id"),                      
        "original_sentences": rec.get('original_code').split('\n'),  
        "merge_sentences": merge_sentences,
        "config_dict":{
            "mixed_labels":mixed_labels,
            "number_of_chunks": 1,
            "sample_token_length": 182,
            "do_top_p": False,
            "do_top_k": True,
            "top_p": 0.96,
            "top_k": 40,
            "model_name": "codellama:34b"
        }    
    }
    
    return new_rec

dst_path = Path("code_restructured_1boundary.json")

transformed = (transform(rec) for rec in records)  # 惰性生成器
json_data = list(transformed)                      # 也可以分批写出避免内存峰值

with dst_path.open("w", encoding="utf‑8") as fp:
    json.dump(json_data, fp, ensure_ascii=False, indent=2)
print(f"🔄 完成！已写入 {dst_path.resolve()}")

train = json_data[:int(0.7 * len(json_data))]  # 80% 训练集
val   = json_data[int(0.7 * len(json_data)):int(0.85 * len(json_data))]  
test  = json_data[int(0.85 * len(json_data)):]

train_path = Path("code_train_mixed.json")
val_path = Path("code_val_mixed.json")
test_path = Path("code_test_mixed.json")

with train_path.open("w", encoding="utf‑8") as fp:
    json.dump(train, fp, ensure_ascii=False, indent=2)
with val_path.open("w", encoding="utf‑8") as fp:
    json.dump(val, fp, ensure_ascii=False, indent=2)
with test_path.open("w", encoding="utf‑8") as fp:
    json.dump(test, fp, ensure_ascii=False, indent=2)

