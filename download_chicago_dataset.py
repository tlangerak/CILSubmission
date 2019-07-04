# this file downloads, sorts and puts the images into the right folders. Also crops them to be quadratic (just to
# min(length, height) and then resizes them to be 800x800 so they can be easily cropped to 4 images of 400x400 :)

import requests, zipfile, io
import shutil
import os
import re
from PIL import Image
import numpy as np
from tqdm import tqdm


# Function to download large files
# Taken from https://stackoverflow.com/a/16696317/10613790
def download_file(url):
    local_filename = url.split('/')[-1].split("?")[0]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    return local_filename


print("Starting Download of Chicago Dataset")
file_name = download_file('https://zenodo.org/record/1154821/files/chicago.zip?download=1')

print("Download finished. Starting extraction")
z = zipfile.ZipFile(file_name)
z.extractall(path = 'data/')
print("Extraction finished.")

# Cleanup after downloading
os.remove(file_name)

# Put it in right folders
root = 'data/chicago/'
input_dir = root + 'input/'
target_dir = root + 'target/'
for name in [input_dir, target_dir]:
    if os.path.isdir(name):
        shutil.rmtree(name)
    os.mkdir(name)

images = set(os.listdir(root))

images = tqdm([x for x in images if x.endswith('png')])
for image in images:
    n = int(re.search(r'\d+', image).group())
    # Ignore faulty images
    if n in [97, 201, 203, 253, 297, 303, 352, 402]:
        os.remove(root + image)
        continue
    # Copy input images to input directory
    if image.endswith("image.png"):
        target = Image.open(root + image)
        target_array = np.asarray(target)
        minimum = min(target_array.shape[0],target_array.shape[1])
        target = target.crop((0,0,minimum,minimum))
        target = target.resize((800,800), Image.BICUBIC)
        target.save(input_dir + str(n).zfill(5) + '.png')
        os.remove(root+image)
    # Extract roads and bring to right format
    else:
        target = Image.open(root + image)
        target_array = np.asarray(target)
        black_array = np.zeros((target_array.shape[0], target_array.shape[1]))
        white_array = np.ones((target_array.shape[0], target_array.shape[1])) * 255
        target = np.where(np.logical_and(target_array[:,:,2] == 255,target_array[:,:,0] == 0), white_array, black_array)
        minimum = min(target.shape[0],target.shape[1])
        target = Image.fromarray(target)
        target = target.crop((0, 0, minimum, minimum))
        target = target.resize((800, 800), Image.BICUBIC)
        target.convert("L").save(target_dir + str(n).zfill(5) + '.png')
        os.remove(root+image)
