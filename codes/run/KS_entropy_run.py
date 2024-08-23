import glob
import os
import numpy as np
import torch
import time
from math import ceil
from torch.multiprocessing import Pool, set_start_method
from scipy.stats import poisson
from scipy.interpolate import interp1d

# uint i to binary bin(i,q_bit): max(i)=15., q_bit=4
def int_to_binary(numbers):
    binlist = np.array([f'{int(num):04b}' for num in numbers])
    concatenated_string = ''.join(binlist)
    return np.array(list(concatenated_string), dtype=int)

# uint i to binary bin(i), random spikes (1) of total number i in 15 time windows
def int_to_binary_random(numbers):
    binary_arrays = [(np.arange(15) < val).astype(int) for val in numbers]
    for binary_array in binary_arrays:
        np.random.shuffle(binary_array)
    return np.hstack(binary_arrays)

# Smooth the spike array
def smooth_sequence(arr):
    n = len(arr)
    smoothed_arr = arr.copy()
    
    start_indices = np.concatenate(([0], np.where(np.diff(arr) != 0)[0] + 1))
    end_indices = np.concatenate((start_indices[1:], [n]))
    
    for start, end in zip(start_indices, end_indices):
        x = np.arange(start, end)
        y = arr[start:end]
        kind = 'cubic' if end - start > 2 else 'linear'
        try:
            f = interp1d(x, y, kind=kind)
        except ValueError:
            f = interp1d(x, y, kind='linear')
        smoothed_arr[start:end] = f(x)
    
    return smoothed_arr

# Input: a spike train (0/1); Output: the KS Entropy matrix of the spike train
def HKS_calculation(spike_train, smooth = False):
    spike_numlist = np.cumsum(spike_train)
    lambda_mle_list = smooth_sequence(spike_numlist) if smooth else spike_numlist

    t_max = len(spike_train)
    r_max = int(1.2 * np.max(lambda_mle_list))
    tau_max = ceil(0.2 * t_max)
    time_windows = 0.2

    if r_max <= 0:
        raise ValueError("r_max must be greater than 0")
    
    HKS_matrix = np.zeros((t_max - tau_max, tau_max))
    prob_from_initial = np.zeros((t_max - tau_max, r_max))

    for t in range(t_max - tau_max):
        if t == 0:
            prob_from_initial[t, 0] = 1
        else:
            prob_from_initial[t, :] = poisson.pmf(np.arange(r_max), lambda_mle_list[t - 1])

    for t in range(t_max - tau_max):
        prob_from_t = np.zeros((tau_max, r_max))
        for tau in range(tau_max):
            lam = lambda_mle_list[t + tau] if t == 0 else lambda_mle_list[t + tau] - lambda_mle_list[t - 1]
            prob_from_t[tau, :] = poisson.pmf(np.arange(r_max), lam)
            for i in range(r_max):
                for j in range(i + 1, r_max):
                    prob = prob_from_t[tau, j - i] * prob_from_initial[t, i]
                    if prob > 0:
                        HKS_matrix[t, tau] -= prob_from_initial[t, i] * prob * np.log(prob)
            HKS_matrix[t, tau] /= (tau + 1)
    
    return HKS_matrix

def process_neuron(args):
    start_time = time.time()
    task_name, word_id, layer_name, sentence_idx, sample, data = args
    data_neuron = data[sentence_idx, :, sample]
    
    try:
        KS_Entropy = HKS_calculation(data_neuron)
    except ValueError as e:
        print(f"Skipping neuron due to error: {e}")
        return
    
    KS_Entropy = HKS_calculation(data_neuron)
    save_path = f"/nfs/xuhan/xyh/results/KS_entropy/{task_name}/{word_id}/{layer_name}.{sentence_idx}.{sample}.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, KS_Entropy)
    
    total_time = time.time() - start_time
    print(f"For task {task_name}, word_id {word_id}, layer {layer_name}, sentence {sentence_idx}, neuron {sample}: KS entropy calculated and saved. Time elapsed: {total_time:.2f} seconds")
    return

def process_layer(task_name, word_id, layer_name):
    paths = glob.glob(os.path.join(f"/nfs/xuhan/xyh/data/data_06_04/int4/{task_name}", f'*doc_id_*_{word_id}.{layer_name}.0.pth'))
    data = np.squeeze(np.array([torch.load(path) for path in paths]))

    tasks = []
    for sentence_idx in range(data.shape[0]):
        sample_sequence = np.random.choice(np.arange(data.shape[2]), size=40, replace=False)
        for sample in sample_sequence:
            tasks.append((task_name, word_id, layer_name, sentence_idx, sample, data))

    with Pool() as pool:
        pool.map(process_neuron, tasks)

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

    task_names = ['arc_easy_true', 'arc_easy_false']

    word_ids_dict = {
        'arc_easy_true': ['2', '3'],
        'arc_easy_false': ['0', '1', '2', '3']
    }

    # 一组task_name/word_id/layer_name/sentence_id配置下随机抽取40个neuron进行KS熵计算
    # GPU似乎并不能有效加速该程序运行，但并行使得现在的计算显著快于之前，并行后速度大概是1min跑250个（单独跑大概需要10s一个）
    for task_name in task_names:
        word_ids = word_ids_dict[task_name]
        for word_id in word_ids:
            for layer_name in layer_name_list:
                process_layer(task_name, word_id, layer_name)
