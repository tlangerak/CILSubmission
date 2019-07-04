import os
import random
import shutil

SPLIT = 0.8

input_dir = 'train/input/'
target_dir = 'train/target/'
input_images = [file for file in os.listdir(input_dir) if file.endswith('.png')]
target_images = [file for file in os.listdir(target_dir) if file.endswith('.png')]

indices = range(len(input_images))
train_indices = random.sample(indices, k=int(len(indices) * SPLIT))
val_indices = list(set(indices) - set(train_indices))
print(len(train_indices), len(val_indices), len(indices))

root_train = 'split_train/'
train_input = root_train + 'input/'
train_target = root_train + 'target/'
root_val = 'split_validation/'
val_input = root_val + 'input/'
val_target = root_val + 'target/'
names = [root_train, train_input, train_target, root_val, val_input, val_target]
for name in names:
    if os.path.isdir(name):
        shutil.rmtree(name)
    os.mkdir(name)

for i in train_indices:
    shutil.copy(input_dir + input_images[i], train_input + input_images[i])
    shutil.copy(target_dir + target_images[i], train_target + target_images[i])

for i in val_indices:
    shutil.copy(input_dir + input_images[i], val_input + input_images[i])
    shutil.copy(target_dir + target_images[i], val_target + target_images[i])