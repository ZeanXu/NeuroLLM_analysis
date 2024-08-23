import glob
import os
import numpy as np
import torch
from math import ceil
from sklearn.neighbors import NearestNeighbors
from scipy.special import psi
import numpy.linalg as la
from scipy.signal import argrelextrema
from scipy.linalg import expm
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from sklearn.metrics import mutual_info_score
from torch.multiprocessing import Pool, set_start_method

# uint i to binary bin(i,q_bit): max(i)=15., q_bit=4
# 该转换规则为一种样例，可考虑其他转二值序列方法。得到的序列用于输入下面的HKS_calculation函数中生成KS熵矩阵
def int_to_binary(numbers):
    binlist = np.array([f'{int(num):04b}' for num in numbers])
    concatenated_string = ''.join(binlist)
    sequence = np.array(list(concatenated_string), dtype=int)
    return sequence

# uint i to binary bin(i), random spikes (1) of total number i in 15 time windows
# 另一种转换规则，即将int15映射到具有对应脉冲次数的15时间窗内
def int_to_binary_random(numbers):
    binary_arrays = [(np.arange(15) < val).astype(int) for val in numbers]
    for binary_array in binary_arrays:
        np.random.shuffle(binary_array)
    sequence = np.hstack(binary_arrays)
    return sequence

# smooth the spike array
# 对脉冲数组进行光滑处理
def smooth_sequence(arr):
    n = len(arr)
    smoothed_arr = arr.copy()
    
    start_indices = np.concatenate(([0], np.where(np.diff(arr) != 0)[0] + 1))
    end_indices = np.concatenate((start_indices[1:], [n]))
    
    for start, end in zip(start_indices, end_indices):
        if end - start > 2:
            x = np.arange(start, end)
            y = arr[start:end]
            try:
                f = interp1d(x, y, kind='cubic')
                smoothed_arr[start:end] = f(x)
            except ValueError:
                f = interp1d(x, y, kind='linear')
                smoothed_arr[start:end] = f(x)
        elif end - start == 2:
            x = np.arange(start, end)
            y = arr[start:end]
            f = interp1d(x, y, kind='linear')
            smoothed_arr[start:end] = f(x)
        else:
            smoothed_arr[start:end] = arr[start:end]
    
    smoothed_arr = np.array(smoothed_arr)
    
    return smoothed_arr

# Input: a spike train (0/1); Output: the KS Entropy matrix of the spike train
# 计算脉冲序列的KS熵值
def HKS_calculation(spike_train, smooth = False):
    # MLE of lambda in Poisson distribution equals to the sample average
    spike_numlist = np.cumsum(spike_train)
    lambda_mle_list = spike_numlist
    if smooth == True:
        lambda_mle_list = smooth_sequence(lambda_mle_list)

    t_max = len(spike_train)
    r_max = int(1.2 * np.max(lambda_mle_list))
    tau_max = ceil(0.2 * t_max) # 默认计算时间长度t=0.8*t_max，前向时间长度tau=0.2*t_max，可手动调节
    time_windows = 0.2

    HKS_matrix = np.zeros((t_max-tau_max, tau_max))
    prob_from_initial = np.zeros((t_max-tau_max, r_max))
    for t in range(t_max-tau_max):
        if t == 0:
            prob_from_initial[t, 0] = 1
        else:
            prob_from_initial[t, 0] = np.exp(-lambda_mle_list[t-1])
            for r in range(r_max-1):
                prob_from_initial[t, r+1] = prob_from_initial[t, r]*lambda_mle_list[t-1]/(r+1)

    for t in range(t_max-tau_max):
        prob_from_t = np.zeros((tau_max, r_max))
        for tau in range(tau_max): # 注意应当设置tau>=1，所以后续使用tau计算时，应考虑tau=tau+1的问题
            if t == 0:
                lam = lambda_mle_list[t+tau]
            else:
                lam = lambda_mle_list[t+tau] - lambda_mle_list[t-1]
            prob_from_t[tau, 0] = np.exp(-lam)
            for r in range(r_max-1):
                prob_from_t[tau, r+1] = prob_from_t[tau, r]*lam/(r+1)
            for i in range(r_max):
                for j in range(i+1, r_max):
                    if (prob_from_t[tau, j-i]*prob_from_initial[t, i]) > 0:
                        HKS_matrix[t][tau] += prob_from_initial[t, i]*prob_from_t[tau, j-i]*prob_from_initial[t, i]*np.log(prob_from_t[tau, j-i]*prob_from_initial[t, i])
            HKS_matrix[t][tau] = -1*HKS_matrix[t][tau]/(tau+1)
        # print('HKS matrix ['+str(t+1)+'] done. (end at '+str(t_max-tau_max)+ ')') # 这句话用来提示进度，有点吵可以注释掉
    
    return HKS_matrix

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]

def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)

def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))

def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric="chebyshev")
    return KDTree(points, metric="chebyshev")

def lnc_correction(tree, points, k, alpha):
    e = 0
    n_sample = points.shape[0]
    for point in points:
        # Find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k + 1, return_distance=False)[0]
        knn_points = points[knn]
        # Substract mean of k-nearest neighbor points
        knn_points = knn_points - knn_points[0]
        # Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
        covr = knn_points.T @ knn_points / k
        _, v = la.eig(covr)
        # Calculate PCA-bounding box using eigen vectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        # Calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

        # Perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect) / n_sample
    return e

# 用于估计序列的Shannon Entropy
def entropy(x, k=3, base=2):
    """The classic K-L k-nearest neighbor continuous entropy estimator
    x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * np.log(2)
    return (const + n_features * np.log(nn).mean()) / np.log(base)

# 基于knn的熵估计，可以手动调一下k的数值
def knn_entropy(data, k=3):
    """计算基于 k-nearest neighbor 的熵"""
    n = len(data)
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data.reshape(-1, 1))
    distances, _ = neigh.kneighbors(data.reshape(-1, 1))
    entropy = np.log(n) - np.mean(np.log(distances[:, k-1] + 1e-10))
    return entropy

# 用于估计两个序列之间的MI
def MutualInfoEstimation(x, y, z=None, k=3, base=np.e, alpha=0.001):
    """Mutual information of x and y (conditioned on z if z is not None)
    x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = (
            avgdigamma(x, dvec),
            avgdigamma(y, dvec),
            digamma(k),
            digamma(len(x)),
        )
        if alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = (
            avgdigamma(xz, dvec),
            avgdigamma(yz, dvec),
            avgdigamma(z, dvec),
            digamma(k),
        )
    return (-a - b + c + d) / np.log(base)

# 按task_name, word_id, layer_name运行MI估计，并保存至对应目录下
def process_layer(task_name, word_id, layer_name_1, layer_name_2):
    paths_1 = glob.glob(os.path.join(f"/nfs/xuhan/xyh/data/data_06_04/int4/{task_name}", f'*doc_id_*_{word_id}.{layer_name_1}.0.pth'))
    t_1 = np.squeeze(np.array([torch.load(path) for path in paths_1]))
    t_1 = t_1.reshape(-1, t_1.shape[2])

    paths_2 = glob.glob(os.path.join(f"/nfs/xuhan/xyh/data/data_06_04/int4/{task_name}", f'*doc_id_*_{word_id}.{layer_name_2}.0.pth'))
    t_2 = np.squeeze(np.array([torch.load(path) for path in paths_2]))
    t_2 = t_2.reshape(-1, t_2.shape[2])

    mi_shape = 10000  # 抽样数10000 pair
    mi_matrix = np.zeros(mi_shape)
    for pair in range(mi_shape):
        mi = MutualInfoEstimation(t_1[:,np.random.randint(np.shape(t_1)[1])].reshape(-1,1), t_2[:,np.random.randint(np.shape(t_2)[1])].reshape(-1,1))
        mi_matrix[pair] = mi

    print(f"task '{task_name}', word '{word_id}', layer '{layer_name_2}': MI average = {np.mean(mi_matrix)}")
    save_path = f"/nfs/xuhan/xyh/results/mi/{task_name}/{word_id}/{layer_name_2}.npy"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, mi_matrix)

def process_layer_wrapper(args):
    task_name, word_id, layer_name_1, layer_name_2 = args
    process_layer(task_name, word_id, layer_name_1, layer_name_2)

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

    # 根据quant层和其它层layer之间neuron pair，对quant层与其它层layer的平均互信息MI进行计算的完整代码
    # 按task_name, word_id, layer_name，随机抽选其与quant层的10000个neuron pair进行互信息MI计算，并对原始MI数组进行保存
    # 单独跑一个layer的互信息MI大概需要10min，并行后速度大概是30min跑180个
    task_list = []
    for task_name in task_names:
        word_ids = word_ids_dict[task_name]
        for word_id in word_ids:
            for layer_name_2 in layer_name_list:
                if layer_name_1 != layer_name_2:  # 避免计算同一'quant'层的互信息
                    task_list.append((task_name, word_id, layer_name_1, layer_name_2))

    with Pool() as pool:
        pool.map(process_layer_wrapper, task_list)
