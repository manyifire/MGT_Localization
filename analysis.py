
import json

def process_jsonl_v2(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    data = []
    for line in lines:
        data.append(json.loads(line))

    if len(data) % 2 != 0:
        print("Warning: Odd number of entries. The last entry will not be processed for AI code swapping.")

    processed_data = []
    for i in range(0, len(data), 2):
        entry1 = data[i]
        if i + 1 < len(data):
            entry2 = data[i+1]

            # Using split('\n') and explicitly adding '\n' back
            hybrid_code1_lines = entry1['hybrid_code'].split('\n')
            boundary1 = entry1['boundary_ix'][0]
            human_code1 = '\n'.join(entry1['human_part'])
            ai_code1 = '\n'.join(entry1['machine_part'])

            hybrid_code2_lines = entry2['hybrid_code'].split('\n')
            boundary2 = entry2['boundary_ix'][0]
            human_code2 = '\n'.join(entry2['human_part'])
            ai_code2 = '\n'.join(entry2['machine_part'])

            # Reconstruct with explicit newlines
            new_hybrid_code1 = human_code1
            if ai_code2:
                new_hybrid_code1 += '\n' + ai_code2

            new_hybrid_code2 = human_code2
            if ai_code1:
                new_hybrid_code2 += '\n' + ai_code1

            new_entry1 = entry1.copy()
            new_entry1['hybrid_code'] = human_code1+'\n' + ai_code2
            new_entry1['machine_part'] = entry2['machine_part']
            processed_data.append(new_entry1)

            new_entry2 = entry2.copy()
            new_entry2['hybrid_code'] = human_code2+'\n' + ai_code1
            new_entry2['machine_part'] = entry1['machine_part']
            processed_data.append(new_entry2)
        

    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        for entry in processed_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    
    i = 'hybridCodeData_HM.jsonl'
    o = 'hybridCodeData_mixed.jsonl'


    with open(i, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        print(lines[0])
        print('------------------------------------')
    with open(o, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        print(lines[0])
    
    print("Created input_v2.jsonl for testing.")
    process_jsonl_v2(i, o)
    print("Processing complete. Check output_v2.jsonl")

    with open(i, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        print(lines[0])
        print('------------------------------------')
    with open(o, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        print(lines[0])


# import json

# # 输入和输出文件路径
# input_file = 'hybridCodeData_cleaned.jsonl'
# output_file = 'hybridCodeData_HM.jsonl'

# # 打开输入文件进行读取
# with open(input_file, 'r') as infile:
#     # 打开输出文件进行写入
#     with open(output_file, 'w') as outfile:
#         for line in infile:
#             # 读取每一行数据并解析为字典
#             data = json.loads(line.strip())
            
#             # 检查prompt_pattern是否为H_M或M_H
#             if data.get('prompt_pattern') in ['H_M']:
#                 # 如果符合条件，写入输出文件
#                 outfile.write(json.dumps(data) + '\n')

# print(f'Filtered data saved to {output_file}')
