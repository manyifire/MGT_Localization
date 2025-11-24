# from pathlib import Path
# import json

# src_path = Path("hybridCodeData_mixed.jsonl")

# records = []
# with src_path.open("r", encoding="utf‑8") as fp:
#     for line in fp:              # 每行都是一个独立 JSON 对象
#         line = line.strip()
#         if not line:
#             continue             
#         records.append(json.loads(line))

# def transform(rec: dict) -> dict:
#     merge_sentences = rec.get('hybrid_code').split('\n')  # 按行分割代码

#     n = len(merge_sentences)            # 总长度

#     # 0: human 1: machine
#     if rec.get('prompt_pattern') == "H_M":
#         zero_len = rec.get("boundary_ix")[0]
#         mixed_labels =  [0] * zero_len + [1] * (n - zero_len)
#     if rec.get('prompt_pattern') == "M_H":
#         zero_len = rec.get("boundary_ix")[0]
#         if zero_len == None:
#             zero_len = 0
#         mixed_labels =  [1] * zero_len + [0] * (n - zero_len)  
#     if rec.get('prompt_pattern') == "H_M_H":
#         zero_len = rec.get("boundary_ix")[0]
#         one_len = rec.get("boundary_ix")[1]
#         mixed_labels =  [0] * zero_len + [1] * (one_len - zero_len) + [0] * (n - one_len)
#     if rec.get('prompt_pattern') == "M_H_M":
#         zero_len = rec.get("boundary_ix")[0]
#         one_len = rec.get("boundary_ix")[1]
#         if zero_len == None or one_len == None:
#             zero_len = 0
#             one_len = n
#         mixed_labels =  [1] * zero_len + [0] * (one_len - zero_len) + [1] * (n - one_len)

#     new_rec = {
#         "article_id": rec.get("code_id"),                      
#         "original_sentences": rec.get('original_code').split('\n'),  
#         "merge_sentences": merge_sentences,
#         "config_dict":{
#             "mixed_labels":mixed_labels,
#             "number_of_chunks": 1,
#             "sample_token_length": 182,
#             "do_top_p": False,
#             "do_top_k": True,
#             "top_p": 0.96,
#             "top_k": 40,
#             "model_name": "codellama:34b"
#         }    
#     }
    
#     return new_rec

# dst_path = Path("code_restructured_1boundary.json")

# transformed = (transform(rec) for rec in records)  # 惰性生成器
# json_data = list(transformed)                      # 也可以分批写出避免内存峰值

# with dst_path.open("w", encoding="utf‑8") as fp:
#     json.dump(json_data, fp, ensure_ascii=False, indent=2)
# print(f"🔄 完成！已写入 {dst_path.resolve()}")

# train = json_data[:int(0.7 * len(json_data))]  # 80% 训练集
# val   = json_data[int(0.7 * len(json_data)):int(0.85 * len(json_data))]  
# test  = json_data[int(0.85 * len(json_data)):]

# train_path = Path("code_train_mixed.json")
# val_path = Path("code_val_mixed.json")
# test_path = Path("code_test_mixed.json")

# with train_path.open("w", encoding="utf‑8") as fp:
#     json.dump(train, fp, ensure_ascii=False, indent=2)
# with val_path.open("w", encoding="utf‑8") as fp:
#     json.dump(val, fp, ensure_ascii=False, indent=2)
# with test_path.open("w", encoding="utf‑8") as fp:
#     json.dump(test, fp, ensure_ascii=False, indent=2)

# ------------处理数据为code blocks----------------
import ast
import re
from typing import List, Dict, Tuple
from collections import Counter

def parse_code_blocks(source_code: str) -> List[Tuple[str, Tuple[int, int]]]:
    # 清理源代码
    cleaned_code = re.sub(r'^```\w*\s*|\s*```$', '', source_code.strip())
    
    try:
        tree = ast.parse(cleaned_code)
    except SyntaxError as e:
        # 如果解析失败，返回整个代码作为一个块
        lines = cleaned_code.splitlines()
        return [(cleaned_code, (0, len(lines)-1))]
    
    blocks = []
    lines = cleaned_code.splitlines()
    
    # 定义控制流节点类型
    control_flow_nodes = (
        ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AsyncWith,
        ast.Break, ast.Continue, ast.Return, ast.Raise, ast.Assert
    )
    
    # 处理AST节点
    def process_node(node, parent_type=None):
        nonlocal blocks, lines
        
        # 获取节点的起始行和结束行
        start_line = getattr(node, 'lineno', None)
        end_line = getattr(node, 'end_lineno', None)
        
        if start_line is None:
            return
        
        # 计算代码块结束行
        if end_line is None:
            # 对于没有结束行的节点，尝试找到下一个同级节点的起始行
            if hasattr(node, 'parent') and hasattr(node.parent, 'body'):
                index = node.parent.body.index(node)
                if index + 1 < len(node.parent.body):
                    next_node = node.parent.body[index + 1]
                    end_line = getattr(next_node, 'lineno', len(lines)) - 1
                else:
                    end_line = len(lines)
            else:
                end_line = start_line
        
        # 提取对应行的源代码
        code_block_lines = lines[start_line-1:end_line]
        code_block = '\n'.join(code_block_lines)
        
        # 确定代码块类型
        block_type = "other"
        
        if isinstance(node, ast.ClassDef):
            block_type = "class_def"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            block_type = "function_def"
        elif isinstance(node, control_flow_nodes):
            block_type = "control_flow"
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AugAssign, ast.AnnAssign)):
            block_type = "other"  # 这些明确归类为"其他"
        
        # 添加到块列表
        blocks.append((code_block, (start_line-1, end_line-1), block_type))
        
        # 对于函数和类，递归处理其子节点
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in node.body:
                if isinstance(child, ast.AST):
                    child.parent = node
                    process_node(child, block_type)
    
    # 为所有节点添加父节点引用
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    
    # 从顶级节点开始处理
    for node in tree.body:
        process_node(node)
    
    # 如果没有找到任何块，返回整个代码作为一个块
    if not blocks:
        return [(cleaned_code, (0, len(lines)-1))]
    
    for block, line_range, _ in blocks:
        print('block---------')
        if(line_range[0]==0):
            block = block.splitlines()[0]
        print(block)

    # 返回代码块和它们的行号范围
    return [(block, line_range) for block, line_range, _ in blocks]

def get_block_label(block_lines_range: Tuple[int, int], merge_line_labels: List[int]) -> int:
    start, end = block_lines_range
    labels_in_block = merge_line_labels[start:end+1]
    if not labels_in_block:
        return 0
    counter = Counter(labels_in_block)
    return counter.most_common(1)[0][0]

def process_data(original_data: List[Dict]) -> List[Dict]:
    new_data = []
    for item in original_data:
        merge_sentences = item['merge_sentences']
        original_labels = item['config_dict']['mixed_labels']
        
        full_code = '\n'.join(merge_sentences)
        blocks_info = parse_code_blocks(full_code)
        
        new_merge_sentences = []
        new_mixed_labels = []
        
        for block, line_range in blocks_info:
            new_merge_sentences.append(block)
            block_label = get_block_label(line_range, original_labels)
            new_mixed_labels.append(block_label)
        
        item['merge_sentences'] = new_merge_sentences
        item['config_dict']['mixed_labels'] = new_mixed_labels
        new_data.append(item)
    
    return new_data

import json
def main():
    with open('/data/wangmanyi/MGT_Localization/code_val.json', 'r') as f:
        data = json.load(f)
    
    processed_data = process_data(data)
    
    with open('/data/wangmanyi/MGT_Localization/code_val_block.json', 'w') as f:
        json.dump(processed_data, f, indent=2)

if __name__ == '__main__':
#     data = [  {
#     "article_id": 3020,
#     "original_sentences": [
#       "def get_complete_slug(self, language=None, hideroot=True):",
#       "        \"\"\"Return the complete slug of this page by concatenating",
#       "        all parent's slugs.",
#       "",
#       "        :param language: the wanted slug language.\"\"\"",
#       "        if not language:",
#       "            language = settings.PAGE_DEFAULT_LANGUAGE",
#       "",
#       "        if self._complete_slug and language in self._complete_slug:",
#       "            return self._complete_slug[language]",
#       "",
#       "        self._complete_slug = cache.get(self.PAGE_URL_KEY % (self.pk))",
#       "        if self._complete_slug is None:",
#       "            self._complete_slug = {}",
#       "        elif language in self._complete_slug:",
#       "            return self._complete_slug[language]",
#       "",
#       "        if hideroot and settings.PAGE_HIDE_ROOT_SLUG and self.is_first_root():",
#       "            url = ''",
#       "        else:",
#       "            url = '%s' % self.slug(language)",
#       "",
#       "        key = self.ANCESTORS_KEY % self.pk",
#       "        ancestors = cache.get(key, None)",
#       "        if ancestors is None:",
#       "            ancestors = self.get_ancestors(ascending=True)",
#       "            cache.set(key, ancestors)",
#       "",
#       "        for ancestor in ancestors:",
#       "            url = ancestor.slug(language) + '/' + url",
#       "",
#       "        self._complete_slug[language] = url",
#       "        cache.set(self.PAGE_URL_KEY % (self.pk), self._complete_slug)",
#       "        return url"
#     ],
#     "merge_sentences": [
#       "def get_complete_slug(self, language=None, hideroot=True):",
#       "    if not language:",
#       "        language = settings.PAGE_DEFAULT_LANGUAGE",
#       "    if self._complete_slug and language in self._complete_slug:",
#       "        return self._complete_slug[language]",
#       "    self._complete_slug = cache.get(self.PAGE_URL_KEY % (self.pk))",
#       "    if self._complete_slug is None:",
#       "        self._complete_slug = {}",
#       "    elif language in self._complete_slug:",
#       "        return self._complete_slug[language]",
#       "",
#       "    if hideroot and settings.PAGE_HIDE_ROOT_SLUG and self.is_first_root():",
#       "        url = ''",
#       "    else:",
#       "        url = '%s' % self.slug(language)",
#       "",
#       "    slugs = []",
#       "    page = self",
#       "    while page:",
#       "        if not hideroot or not page.is_first_root():",
#       "            slugs.insert(0, page.slug(language))",
#       "        page = page.parent",
#       "",
#       "    return ' '.join(slugs)",
#       "```"
#     ],
#     "config_dict": {
#       "mixed_labels": [
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         0,
#         1,
#         1,
#         1,
#         1,
#         1,
#         1,
#         1
#       ],
#       "number_of_chunks": 1,
#       "sample_token_length": 182,
#       "do_top_p": False,
#       "do_top_k": True,
#       "top_p": 0.96,
#       "top_k": 40,
#       "model_name": "codellama:34b"
#     }
#   },]
    
#     processed_data = process_data(data)
#     print(processed_data)
    main()
