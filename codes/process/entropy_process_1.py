import glob
import os
import pandas as pd
import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
from scipy.special import digamma

def means_to_csv(load_path, save_path):
    files = [f for f in os.listdir(load_path) if f.endswith('.npy')]
    data = []
    
    for file in files:
        file_path = os.path.join(load_path, file)
        array = np.load(file_path)
        mean_value = np.mean(array)
        layer_name = os.path.splitext(os.path.basename(file))[0]
        data.append([layer_name, mean_value])
    
    # 创建DataFrame并保存为CSV文件
    df = pd.DataFrame(data, columns=['Layer', 'Mean'])
    df = df.sort_values(by='Layer')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f'Successfully saved means to {save_path}')

# 按task_name, word_id, layer_name运行Shannon entropy估计，并保存至对应目录下
def process_layer(args):
    task_name, word_id = args
    load_path = f"/nfs/xuhan/xyh/results/entropy_new/{task_name}/{word_id}/"
    save_path = f"/nfs/xuhan/xyh/analysis/entropy_new/{task_name}.{word_id}.mean.csv"
    means_to_csv(load_path, save_path)

layer_name_list = [
    'att.key.l0', 'att.key.l1', 'att.key.l2', 'att.key.l3', 'att.key.l4', 'att.key.l5',
    'att.key.l6', 'att.key.l7', 'att.key.l8', 'att.key.l9', 'att.key.l10', 'att.key.l11',
    'att.value.l0', 'att.value.l1', 'att.value.l2', 'att.value.l3', 'att.value.l4', 'att.value.l5',
    'att.value.l6', 'att.value.l7', 'att.value.l8', 'att.value.l9', 'att.value.l10', 'att.value.l11',
    'att.receptance.l0', 'att.receptance.l1', 'att.receptance.l2', 'att.receptance.l3', 'att.receptance.l4', 'att.receptance.l5',
    'att.receptance.l6', 'att.receptance.l7', 'att.receptance.l8', 'att.receptance.l9', 'att.receptance.l10', 'att.receptance.l11',
    'att.sigmoid.l0', 'att.sigmoid.l1', 'att.sigmoid.l2', 'att.sigmoid.l3', 'att.sigmoid.l4', 'att.sigmoid.l5',
    'att.sigmoid.l6', 'att.sigmoid.l7', 'att.sigmoid.l8', 'att.sigmoid.l9', 'att.sigmoid.l10', 'att.sigmoid.l11',
    
    'ffn.key.l0', 'ffn.key.l1', 'ffn.key.l2', 'ffn.key.l3', 'ffn.key.l4', 'ffn.key.l5',
    'ffn.key.l6', 'ffn.key.l7', 'ffn.key.l8', 'ffn.key.l9', 'ffn.key.l10', 'ffn.key.l11',
    'ffn.value.l0', 'ffn.value.l1', 'ffn.value.l2', 'ffn.value.l3', 'ffn.value.l4', 'ffn.value.l5',
    'ffn.value.l6', 'ffn.value.l7', 'ffn.value.l8', 'ffn.value.l9', 'ffn.value.l10', 'ffn.value.l11',
    'ffn.receptance.l0', 'ffn.receptance.l1', 'ffn.receptance.l2', 'ffn.receptance.l3', 'ffn.receptance.l4', 'ffn.receptance.l5',
    'ffn.receptance.l6', 'ffn.receptance.l7', 'ffn.receptance.l8', 'ffn.receptance.l9', 'ffn.receptance.l10', 'ffn.receptance.l11',
    'ffn.sigmoid.l0', 'ffn.sigmoid.l1', 'ffn.sigmoid.l2', 'ffn.sigmoid.l3', 'ffn.sigmoid.l4', 'ffn.sigmoid.l5',
    'ffn.sigmoid.l6', 'ffn.sigmoid.l7', 'ffn.sigmoid.l8', 'ffn.sigmoid.l9', 'ffn.sigmoid.l10', 'ffn.sigmoid.l11',
    
    'block.ln1.l0', 'block.ln1.l1', 'block.ln1.l2', 'block.ln1.l3', 'block.ln1.l4', 'block.ln1.l5',
    'block.ln1.l6', 'block.ln1.l7', 'block.ln1.l8', 'block.ln1.l9', 'block.ln1.l10', 'block.ln1.l11',
    'block.ln2.l0', 'block.ln2.l1', 'block.ln2.l2', 'block.ln2.l3', 'block.ln2.l4', 'block.ln2.l5',
    'block.ln2.l6', 'block.ln2.l7', 'block.ln2.l8', 'block.ln2.l9', 'block.ln2.l10', 'block.ln2.l11',
    
    'quant', 'block.ln0', 'head'
]

task_names = ['winogrande_true', 'winogrande_false', 'arc_easy_true', 'arc_easy_false']

word_ids_dict = {
    'winogrande_true': ['0', '1'],
    'winogrande_false': ['0', '1'],
    'arc_easy_true': ['0', '1', '2', '3'],
    'arc_easy_false': ['0', '1', '2', '3']
}

# 根据单neuron对layer的平均信息熵entropy进行计算的完整代码
# 这个跑起来很快，但现在看每一层的entropy都基本接近在29左右，分析效果不是很好
# 这可能是因为每个神经元在每个时间窗内可以视作等可能的0-15发放
if __name__ == "__main__":
    set_start_method('spawn')  # 兼容不同操作系统，确保在Windows上正常运行
    args_list = []
    for task_name in task_names:
        word_ids = word_ids_dict[task_name]
        for word_id in word_ids:
            args_list.append((task_name, word_id))

    # 使用多进程进行计算
    with Pool() as pool:
        pool.map(process_layer, args_list)
