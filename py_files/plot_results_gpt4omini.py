import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


# cr_metrics_path = 'results/generation_scores_apr.json'

# # load the json file

# with open(cr_metrics_path, 'r', encoding='utf-8') as f:
#     cr_metrics = json.load(f)


# correctness = []


# for dic in cr_metrics:
#     correctness.append(dic['answer_correctness'])


cr_metrics_path = 'meta_codebook.json'

# load the json file

with open(cr_metrics_path, 'r', encoding='utf-8') as f:
    cr_metrics = json.load(f)


print(len(cr_metrics['e_embeddings'][0]))

print(len(cr_metrics['facts_lst'][0]))
print(len(cr_metrics['facts_lst']))












# python plot_results_gpt4omini.py