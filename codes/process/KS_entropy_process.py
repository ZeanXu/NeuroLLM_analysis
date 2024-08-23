import os
import numpy as np
import pandas as pd
from multiprocessing import Process, set_start_method

# 定义基本路径、任务名称和对应的word_id字典
base_dir = '/nfs/xuhan/xyh/results/KS_entropy'
output_dir = '/nfs/xuhan/xyh/analysis/KS_entropy'
task_names = ['winogrande_true', 'winogrande_false', 'arc_easy_true', 'arc_easy_false']
word_ids_dict = {
    'winogrande_true': ['0', '1'],
    'winogrande_false': ['0', '1'],
    'arc_easy_true': ['0', '1', '2', '3'],
    'arc_easy_false': ['0', '1', '2', '3']
}

categories = [
    'att.key', 'att.receptance', 'att.sigmoid', 'att.value',
    'block.ln0', 'block.ln1', 'block.ln2',
    'ffn.key', 'ffn.receptance', 'ffn.sigmoid', 'ffn.value',
    'head', 'quant'
]

def process_task(task_name, word_id):
    print(f"Processing {task_name} word_id {word_id}")
    task_dir = os.path.join(base_dir, task_name)
    word_id_dir = os.path.join(task_dir, word_id)
    results = []

    # 获取文件夹中所有文件的列表
    files = os.listdir(word_id_dir)

    # 遍历每个categories
    for category in categories:
        # 找到包含category名的文件
        category_files = [f for f in files if category in f and f.endswith('.npy')]

        for file_name in category_files:
            file_path = os.path.join(word_id_dir, file_name)

            # 读取npy文件
            data = np.load(file_path)

            # 计算均值
            mean_value = np.mean(data)
            results.append([category, mean_value])

    # 转换为DataFrame并按categories排序
    results_df = pd.DataFrame(results, columns=['category', 'mean'])
    results_df['category'] = pd.Categorical(results_df['category'], categories)
    results_df = results_df.sort_values(by='category').reset_index(drop=True)

    # 保存结果为CSV文件
    output_file = os.path.join(output_dir, f"{task_name}.{word_id}.mean.csv")
    results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    set_start_method('spawn')
    processes = []

    # 遍历每个任务名称和对应的word_id
    for task_name in task_names:
        word_ids = word_ids_dict[task_name]

        for word_id in word_ids:
            p = Process(target=process_task, args=(task_name, word_id))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()
