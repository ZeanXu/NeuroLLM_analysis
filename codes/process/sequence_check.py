import glob
import os
import numpy as np
import torch
import pandas as pd
from math import ceil
from torch.multiprocessing import Pool, set_start_method
from scipy.stats import poisson
from scipy.interpolate import interp1d

def process_neuron(args):
    task_name, word_id, layer_name, sentence_idx, neuron_id, data = args
    data_neuron = data[sentence_idx, :, neuron_id]
    
    # 检查最后一项是否小于等于30
    if np.cumsum(data_neuron)[-1] <= 30:
        return [task_name, word_id, layer_name, sentence_idx, neuron_id, np.cumsum(data_neuron)[-1]]
    return None

def process_layer(task_name, word_id, layer_name):
    paths = glob.glob(os.path.join(f"/nfs/xuhan/xyh/data/data_06_04/int4/{task_name}", f'*doc_id_*_{word_id}.{layer_name}.0.pth'))
    data = np.squeeze(np.array([torch.load(path) for path in paths]))

    tasks = []
    for sentence_idx in range(data.shape[0]):
        for neuron_id in range(data.shape[2]):
            tasks.append((task_name, word_id, layer_name, sentence_idx, neuron_id, data))

    with Pool() as pool:
        results = pool.map(process_neuron, tasks)
    
    return [result for result in results if result is not None]

if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

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

    all_results = []
    
    for task_name in task_names:
        word_ids = word_ids_dict[task_name]
        for word_id in word_ids:
            for layer_name in layer_name_list:
                results = process_layer(task_name, word_id, layer_name)
                all_results.extend(results)

    # 使用 pandas 将数据写入 CSV 文件
    df = pd.DataFrame(all_results, columns=['task_name', 'word_id', 'layer_name', 'sentence_idx', 'neuron_id', 'cumsum'])
    df.to_csv('/nfs/xuhan/xyh/analysis/sparse/sparse_neurons.csv', index=False)
