from PIL import Image
import random
import os
import re
import shutil
from tqdm import tqdm

#rotates image by 45 degrees
def rotate_by_45(image):
    width, height = image.size
    left = (width - 282) / 2
    top = (height - 282) / 2
    right = (width + 282) / 2
    bottom = (height + 282) / 2
    transformed_image = image.crop((left, top, right, bottom)).resize((400, 400),
                                                                              resample=Image.BICUBIC)
    return transformed_image

#does all augmentations on input and target image and saves them
def transform_image_combined(image_1, image_2, dir, counter):
    opened_image_1 = Image.open(image_1)
    opened_image_2 = Image.open(image_2)

    rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    flip_direction = [Image.FLIP_LEFT_RIGHT, None]
    for angle in rotation_angles:
        for flip in flip_direction:
            if flip:
                transformed_image_input = opened_image_1.rotate(angle).transpose(flip)
                transformed_image_target = opened_image_2.rotate(angle).transpose(flip)
            else:
                transformed_image_input = opened_image_1.rotate(angle)
                transformed_image_target = opened_image_2.rotate(angle)
                
            transformed_image_target.convert('1')
            if angle % 90 != 0:
                transformed_image_input = rotate_by_45(transformed_image_input)
                transformed_image_target = rotate_by_45(transformed_image_target)
            transformed_image_input.save(dir + '/input/' + str(counter).zfill(5) + '.png')
            transformed_image_target.save(dir + '/target/' + str(counter).zfill(5) + '.png')
            counter += 1
    return counter


#crop each image from indir into as many nonoverlapping images of "size" as possible and save them to outdir
def crop_and_save_images(indir, outdir, size):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.mkdir(outdir)
    os.mkdir(outdir + '/input')
    os.mkdir(outdir + '/target')
    input_images = [file for file in os.listdir(indir + '/input/') if file.endswith('.png')]
    target_images = [file for file in os.listdir(indir + '/target/') if file.endswith('.png')]
    print("Processing input images")
    for im in tqdm(input_images):
        img = Image.open(indir+ '/input/' +im)
        n = int(re.search(r'\d+', im).group())
        width = img.size[0]
        height = img.size[1]
        i = 0
        j = 0
        while((i+1)*size <= width):
            while((j+1)*size <= height):
                crop = img.crop((i*size, j*size, (i+1)*size, (j+1)*size))
                crop.save(outdir + "/input/" + str(n) + 'crop_' + str(i) + str(j) + '.png')
                j += 1
            j = 0
            i += 1
    print("Processing target images")
    for im in tqdm(target_images):
        img = Image.open(indir + '/target/' + im)
        n = int(re.search(r'\d+', im).group())
        width = img.size[0]
        height = img.size[1]
        i = 0
        j = 0
        while ((i + 1) * size <= width):
            while ((j + 1) * size <= height):
                crop = img.crop((i * size, j * size, (i + 1) * size, (j + 1) * size))
                crop.save(outdir + "/target/" + str(n) + 'crop_' + str(i) + str(j)+ '.png')
                j += 1
            j = 0
            i += 1


#splits a dataset consisting of an input and target folder into 4 folders (split_train/input, split_train/target, split_validate/input, split_validate/target)
#fraction of validation set is according to validation_fraction
def split_dataset(str, validation_fraction):
    os.mkdir(str + '/split_train')
    os.mkdir(str + '/split_train/input')
    os.mkdir(str + '/split_train/target')
    os.mkdir(str + '/split_validate')
    os.mkdir(str + '/split_validate/input')
    os.mkdir(str + '/split_validate/target')

    list_training = os.listdir(str+'/input')
    list_validation = os.listdir(str + '/target')
    nr_total = len(list_validation)
    nr = int(len(list_training)*validation_fraction)
    indices = [x for x in range(0,nr_total)]
    validate = random.sample(indices, k = nr)
    validationset_input = []
    validationset_target = []
    for val in validate:
        validationset_input.append(list_training[val])
        validationset_target.append(list_validation[val])

    train = set(indices) - set(validate)
    trainingset_input = []
    trainingset_target = []
    for tr in train:
        trainingset_input.append(list_training[tr])
        trainingset_target.append(list_validation[tr])
    for pic in validationset_input:
        shutil.copy(str+'/input/'+ pic, str + '/split_validate/input/' + pic)
    for pic in validationset_target:
        shutil.copy(str + '/target/' + pic, str + '/split_validate/target/' + pic)
    for pic in trainingset_input:
        shutil.copy(str + '/input/' + pic, str + '/split_train/input/' + pic)
    for pic in trainingset_target:
        shutil.copy(str + '/target/' + pic, str + '/split_train/target/' + pic)
    return

#check if there already is a train/validatation split, otherwise create one. Then do the augmentations on both splits for input and target
def call_on_all_sets(train_set=[], validation_fraction = 0.2):
    counter_train = 0
    counter_validate = 0
    for str in train_set:
        if(not os.path.isdir(str+'/split_train')):
            split_dataset(str, validation_fraction)
        input_images = [file for file in os.listdir(str + '/split_train/input/') if file.endswith('.png')]
        target_images = [file for file in os.listdir(str + '/split_train/target/') if file.endswith('.png')]
        print("Creating split for train for {}".format(str))
        for im, tar in tqdm(list(zip(input_images, target_images))):
            counter_train = transform_image_combined(str + '/split_train/input/' + im, str + '/split_train/target/' + tar, static_dir +'/split_train', counter_train)
        input_images = [file for file in os.listdir(str + '/split_validate/input/') if file.endswith('.png')]
        target_images = [file for file in os.listdir(str + '/split_validate/target/') if file.endswith('.png')]
        print("Creating split for validation for {}".format(str))
        for im, tar in tqdm(list(zip(input_images, target_images))):
            counter_validate = transform_image_combined(str + '/split_validate/input/' + im,str + '/split_validate/target/'+ tar, static_dir + '/split_validate', counter_validate)

if __name__ == '__main__':
    # Create augmented from original
    static_dir = 'data/augmented'
    if os.path.isdir(static_dir):
        shutil.rmtree(static_dir)
    os.mkdir(static_dir)
    os.mkdir(static_dir + '/split_train')
    os.mkdir(static_dir + '/split_validate')
    os.mkdir(static_dir + '/split_train/input')
    os.mkdir(static_dir + '/split_train/target')
    os.mkdir(static_dir + '/split_validate/input')
    os.mkdir(static_dir + '/split_validate/target')
    call_on_all_sets(['data/original'], 0.2)

    # create static from chicago
    static_dir = 'data/static'
    if os.path.isdir(static_dir):
        shutil.rmtree(static_dir)
    os.mkdir(static_dir)
    os.mkdir(static_dir + '/split_train')
    os.mkdir(static_dir + '/split_validate')
    os.mkdir(static_dir + '/split_train/input')
    os.mkdir(static_dir + '/split_train/target')
    os.mkdir(static_dir + '/split_validate/input')
    os.mkdir(static_dir + '/split_validate/target')
    crop_and_save_images('data/chicago', 'data/chicago/cropped', 400)
    call_on_all_sets(['data/chicago/cropped'], 0.2)