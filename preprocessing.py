'''
OLD

some experimentations with preprocessing
'''


from PIL import Image
import random
import glob
import os
import numpy as np
import shutil
from sklearn.utils import shuffle


def sparse_labels(img, label_type = 'road'):
    # Return a 2D label, given a 3D template.
    def color_map(img_array, ch):
        r = img_array[:,:,0]
        #g = img_array[:,:,1]
        b = img_array[:,:,2]
        #rm = np.where(r > 0, 0, 1)
        #gm = np.where(g > 0, 0, 1)
        #bm = np.where(b > 0, 0, 1)
        #print(np.where(rm-gm > 0))

        if ch == 0:
            mask = r # np.where(((rm - gm) <= 0), 0, 1) * 255, I dont understand how the correct way to do it isnt rm - gm...
            return mask
        if ch == 2:
            mask = b
            return mask
    # label_type: road -> only returns roads, 'building' -> only returns buildings
    image_array = np.array(img)
    if label_type is 'road':
        mask = color_map(image_array, 0) #red
        label =  Image.fromarray(mask)
        # determine road_map label
        # label.putdata()
    if label_type is 'building':
        mask = color_map(image_array, 2) #blue
        label =  Image.fromarray(mask, 'L')
        # determine building map
        # label.putdata()
    return label

def save_image(image, dir, counter):
    if image != None:
        image.save(dir + str(counter).zfill(5) + '.png')
    return

def preprocess(images, labels, output_dir_input, output_dir_target):
    # lists of all files
    img_list = []
    label_list = []
    for file in os.listdir(images):
        img_list.append(os.path.join(images, file))
    for file in os.listdir(labels):
        label_list.append(os.path.join(labels, file))

    num_files = len(img_list)
    assert num_files == len(label_list), print('Number of images does not correspond to number of target maps!')
    print('Number of training images:', num_files)

    ids = []
    for id in range(num_files):
        check_img_id = 'o'+str(id)+'_'
        for file_idx in range(num_files):
            if (check_img_id in img_list[file_idx]) and (check_img_id in label_list[file_idx]):
                ids.append(file_idx)
    
    print('Number of valid files:', len(ids))
    # go through the IDs:
    counter = 0
    for i in range(len(ids)):
        #for i in range(len(ids)):
        print('Image ID:', ids[i])
        img = Image.open(img_list[ids[i]])
        label = Image.open(label_list[ids[i]])
        sparse = sparse_labels(label)
        
        save_image(img, output_dir_input + '/', counter)
        save_image(sparse, output_dir_target + '/', counter)
        counter += 1

##################################################################################################################

folder = 'data'
origin = folder + '/chicago/chicago'
output = folder + '/chicago/processed'
input_dir = origin + '/image'
target_dir = origin + '/labels'
output_dir_input = output + '/input'
output_dir_target = output + '/target'
    
preprocess(input_dir, target_dir, output_dir_input, output_dir_target)

##################################################################################################################
