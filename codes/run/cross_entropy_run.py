import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from torch.multiprocessing import Pool, set_start_method
from sklearn.neighbors import NearestNeighbors

def density_estimate(data, k, offset=1e-10):
    nbrs = NearestNeighbors(n_neighbors=k).fit(data.reshape(-1, 1))
    distances, _ = nbrs.kneighbors(data.reshape(-1, 1))
    # 距离转化为密度估计，并添加偏移量
    density = k / (len(data) * np.pi * (distances[:, -1] + offset) ** 2)
    # 归一化密度为概率分布
    density /= np.sum(density)
    return density

def cross_entropy_knn(x, y, k=3, offset=1e-10):
    x = np.array(x)
    y = np.array(y)
    # 计算x的kNN概率分布
    px_knn = density_estimate(x, k, offset)
    # 计算y的kNN概率分布
    py_knn = density_estimate(y, k, offset)
    # 计算交叉熵 H(P, Q)
    cross_entropy = -np.sum(px_knn * np.log(py_knn))
    return cross_entropy

def process_layer(task_name, word_id, layer_name_1, layer_name_2):
    paths_1 = glob.glob(os.path.join(f"/nfs/xuhan/xyh/data/data_06_04/int4/{task_name}", f'*doc_id_*_{word_id}.{layer_name_1}.0.pth'))
    t_1 = np.squeeze(np.array([torch.load(path) for path in paths_1]))
    t_1 = t_1.reshape(-1, t_1.shape[2])

    paths_2 = glob.glob(os.path.join(f"/nfs/xuhan/xyh/data/data_06_04/int4/{task_name}", f'*doc_id_*_{word_id}.{layer_name_2}.0.pth'))
    t_2 = np.squeeze(np.array([torch.load(path) for path in paths_2]))
    t_2 = t_2.reshape(-1, t_2.shape[2])

    ce_shape = 10000  # 抽样数10000 pair
    ce_matrix = np.zeros(ce_shape)
    for pair in range(ce_shape):
        x = t_1[:, np.random.randint(np.shape(t_1)[1])]
        y = t_2[:, np.random.randint(np.shape(t_2)[1])]
        ce = cross_entropy_knn(x, y)
        ce_matrix[pair] = ce

    print(f"task '{task_name}', word '{word_id}', layer '{layer_name_2}': Cross-Entropy average = {np.mean(ce_matrix)}")
    save_path = f"/nfs/xuhan/xyh/results/cross_entropy/{task_name}/{word_id}/{layer_name_2}.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, ce_matrix)

def process_layer_wrapper(args):
    task_name, word_id, layer_name_1, layer_name_2 = args
    try:
        process_layer(task_name, word_id, layer_name_1, layer_name_2)
    except Exception as e:
        print(f"Error processing task {task_name}, word {word_id}, layer {layer_name_1} to {layer_name_2}: {str(e)}")

if __name__ == "__main__":
    set_start_method("spawn")

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
    
    layer_name_1 = 'quant'  # 初始输入层的layer_name

    task_list = []
    for task_name in task_names:
        word_ids = word_ids_dict[task_name]
        for word_id in word_ids:
            for layer_name_2 in layer_name_list:
                if layer_name_1 != layer_name_2:  # 避免计算同一'quant'层的互信息
                    task_list.append((task_name, word_id, layer_name_1, layer_name_2))

    with Pool() as pool:
        pool.map(process_layer_wrapper, task_list)
