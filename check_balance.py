""""
This script counts in the number of True and False labels on our dataset.
"""


from DataWrapper import *
import numpy as np
from architectures import *
from evaluation import *

train_data = DataWrapper("data/static/split_train/input/", "data/static/split_train//target/", False)
training_data = create_batches(train_data, batch_size=1)

counts_list = [0, 0]
for n_batch, batch in enumerate(training_data):
    targets = batch['target'].detach().numpy()
    target = targets.flatten()
    unique, counts = np.unique(target, return_counts=True)
    for u, c in zip(unique, counts):
        counts_list[int(u)] += c
    print(n_batch, counts_list)

total = sum(counts_list)
print(counts_list[0]/total, counts_list[1]/total)
