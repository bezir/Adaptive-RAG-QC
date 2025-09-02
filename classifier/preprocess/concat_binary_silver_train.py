import json, os
from preprocess_utils import *
import sys

lst_total = []

lst_dataset_name = ['musique', '2wikimultihopqa', 'hotpotqa', 'nq', 'trivia', 'squad']
lst_set_name = ['train', 'valid']
lst_model_name = ['flan_t5_xl', 'flan_t5_xxl', 'gemini-2.5-flash-lite', 'gemini-1.5-flash-8b', 'qwen']

dataset_dir = "musique_hotpot_wiki2_nq_tqa_sqd"

# train
for model_name in lst_model_name:
    binary_input_file = os.path.join('classifier', "data", dataset_dir, 'binary', 'total_data_train.json')
    silver_input_file = os.path.join('classifier', "data", dataset_dir, model_name, 'silver', 'train.json')
    output_file = os.path.join('classifier', "data", dataset_dir, model_name, 'binary_silver', 'train.json')
    concat_and_save_binary_silver(binary_input_file, silver_input_file, output_file)