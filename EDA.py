from transformers import AutoTokenizer
import torch
from torch import nn

from read_input import read_data

if __name__ == "__main__":
    train_path = "/home/michal/Documents/research/sentiment/DimABSA2026/task-dataset/track_a/subtask_3/eng/eng_restaurant_train_alltasks.jsonl"
    data = read_data(train_path)

    category = {}
    category_1 = {}
    category_2 = {}

    for key, value in data.items():
        print(key)
        for elem in value['Quadruplet']:
            k = elem['Category'].split("#")[0]
            k2 = elem['Category'].split("#")[1]
            
            category[elem['Category']] = category.get(elem['Category'], 0) + 1
            category_1[k] = category_1.get(k, 0) + 1
            category_2[k2] = category_2.get(k2, 0) + 1

    print(category_1)
    print(category_2)
    print(category)

