import os
import json
from collections import defaultdict

ChatGLM_6B_bidirectional_mask = '/home.local/dongbenyang/workspace/workspace_43/project/opencompass/outputs/default/outputs/20241019/results/ChatGLM-6B_bidirectional_mask'
ChatGLM_6B_single_mask = '/home.local/dongbenyang/workspace/workspace_43/project/opencompass/outputs/default/outputs/20241019/results/ChatGLM-6B_single_mask'


evaluate_map = defaultdict(list)

min = float('inf')
max = float('-inf')
min_max_save = {'min':[], 'max':[]}

for idx, file in enumerate(os.listdir(ChatGLM_6B_bidirectional_mask)):
    filename = file.split('.')[0]
    
    with open(os.path.join(ChatGLM_6B_bidirectional_mask, file)) as f:
        bidirectional_mask_data = json.load(f)
        
    with open(os.path.join(ChatGLM_6B_single_mask, file)) as f:
        single_mask_data = json.load(f)

    diff = bidirectional_mask_data['accuracy'] - single_mask_data['accuracy']
    print(filename, bidirectional_mask_data['accuracy'], single_mask_data['accuracy'], diff)
        
    evaluate_map.update({filename:[ bidirectional_mask_data['accuracy'], single_mask_data['accuracy'], diff]})
    
    if min > diff:
        min = diff
        min_max_save['min'] = [filename, min]
    if max < diff:
        max = diff
        min_max_save['max'] = [filename, max]
print(min_max_save)
        
def to_md(map_data):
    line_head = '| 数据集 | ChatGLM_6B_bidirectional_mask | ChatGLM_6B_single_mask | |\n'
    line1 = '| -----------| ----------- | ----------- | ----------- |\n'
    line2 = '| mmlu | accuracy | accuracy | diff |\n'
    context = line_head + line1 + line2
    
    for k, v in map_data.items():
        line = f'| {k} | {v[0]} | {v[1]} |{v[2]} |\n'
        context += line
    return context
context = to_md(evaluate_map)

with open('ChatGLM_6B_gms8k_evaluate_diff.md', 'w') as f:
    f.write(context)