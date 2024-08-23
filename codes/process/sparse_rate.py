import pandas as pd
import itertools

# All possible combinations
task_names = ['winogrande_true', 'winogrande_false', 'arc_easy_true', 'arc_easy_false']
word_ids_dict = {
    'winogrande_true': [0, 1],
    'winogrande_false': [0, 1],
    'arc_easy_true': [0, 1, 2, 3],
    'arc_easy_false': [0, 1, 2, 3]
}
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
sentence_idx_range = range(20)

all_combinations = []
for task_name, word_ids in word_ids_dict.items():
    for word_id in word_ids:
        for layer_name in layer_name_list:
            for sentence_idx in sentence_idx_range:
                all_combinations.append([task_name, word_id, layer_name, sentence_idx])

# Convert to DataFrame
all_combinations_df = pd.DataFrame(all_combinations, columns=['task_name', 'word_id', 'layer_name', 'sentence_idx'])

sparse_df = pd.read_csv('/nfs/xuhan/xyh/analysis/sparse/sparse_neurons.csv')
grouped_df = sparse_df.groupby(['task_name', 'word_id', 'layer_name', 'sentence_idx']).size().reset_index(name='sparse_count')

neuron_counts_df = pd.read_csv('/nfs/xuhan/xyh/analysis/sparse/neuron_counts.csv')
complete_df = pd.merge(all_combinations_df, grouped_df, on=['task_name', 'word_id', 'layer_name', 'sentence_idx'], how='left')
complete_df['sparse_count'] = complete_df['sparse_count'].fillna(0)
complete_df = pd.merge(complete_df, neuron_counts_df, on=['task_name', 'word_id', 'layer_name'])


complete_df['frequency'] = complete_df['sparse_count'] / complete_df['neuron_count']
complete_df.to_csv('/nfs/xuhan/xyh/analysis/sparse/sparse_rate.csv', index=False)
print("Frequency distribution saved to 'sparse_rate.csv'")
