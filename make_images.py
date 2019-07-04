'''
Script to generate images for the report.
It takes a list models you want images for (only U-Net and W-Net intermediate atm)
returns final results and intermediate if applicable

Leonhard command:
bsub -u 1 -W 25:00 -R "rusage[mem=4000, ngpus_excl_p=1]" 'python make_images.py'
'''


import torch
from DataWrapper import *
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import numpy as np
from architectures import *
import torch.nn as nn
import torch.optim
from plotter_helper import *
from evaluation import *
from losses import *
import utils
import os
import constants
from tqdm import trange, tqdm
import shutil


def toTensorRGB(image):
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image_1 = image.transpose((2, 0, 1))
    torch_image = torch.from_numpy(np.asarray([image_1])).type(torch.FloatTensor).cuda()
    return torch_image / 255.

models = [
    'W-Net-Intermediate_opt_Adam_data_constant_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164639',
    'U-Net_opt_Adam_data_constant_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164642',
    'W-Net-Intermediate_opt_Adam_data_TrainAugmented_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164709',
    'U-Net_opt_Adam_data_TrainAugmented_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164709',
    'W-Net-Intermediate_opt_Adam_data_Train_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164708',
    'U-Net_opt_Adam_data_Train_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_162733']

val_dir = os.path.join(constants.data_dir + '/static/', "split_validate")
val_data = DataWrapper(os.path.join(val_dir, "input/"), os.path.join(val_dir, "target/"),
                       torch.cuda.is_available())
static_validation_data = create_batches(val_data, batch_size=1)
nr_validation_samples = val_data.__len__()
savedir = '/images/'
runDir = "runs/"

for m in models:
    inter_dir = runDir + m + savedir
    if os.path.isdir(inter_dir):
        shutil.rmtree(inter_dir)
    os.mkdir(inter_dir)
#
# for m in models:
#     # load the model and create directories.
#     folder = m.split("_")
#     model_name = folder[0]
#     print(model_name)
#     if model_name == "U-Net":
#         model = UNet(3, 2)
#     elif model_name == "W-Net-Intermediate":
#         model = WNet(3, 1, intermediate=True, leaky=False, batch_norm=True, drop_rate=0.1, dilate_first=[1], dilate_second=[1])
#
#     model.load_state_dict(torch.load(runDir + m + "/model/model.pt"))
#     model.cuda()
#     model.eval()
#     inter_dir = runDir + m + savedir
#
#     for i in range(1, 224):
#         filename = "data/original/test/test_" + str(i) + ".png"
#         if not os.path.isfile(filename):
#             continue
#
#         print("Loading image {}".format(filename))
#
#         img = io.imread(filename)
#         input = toTensorRGB(img)
#         outputs = model(input)
#         full_sigmoided_out = torch.sigmoid(outputs)
#
#         if model_name == 'W-Net-Intermediate':
#             outputs = full_sigmoided_out[0][0].unsqueeze(0).detach().cpu().numpy()
#             out_image = Image.fromarray(np.uint8(outputs[0] * 255), 'L')
#             out_image.convert('RGB').save(inter_dir + "/intermediate_" + str(i) + ".png")
#
#         # always save the final result
#         outputs = full_sigmoided_out[0][-1].unsqueeze(0).detach().cpu().numpy()
#         out_image = Image.fromarray(np.uint8(outputs[0] * 255), 'L')
#         out_image.convert('RGB').save(inter_dir + "/final_" + str(i) + ".png")


with tqdm(enumerate(static_validation_data), total=nr_validation_samples) as tb:
    for i_batch, batch in tb:
        for m in models:
            # load the model and create directories.
            folder = m.split("_")
            model_name = folder[0]
            print(model_name)
            if model_name == "U-Net":
                model = UNet(3, 2)
            elif model_name == "W-Net-Intermediate":
                model = WNet(3, 1, intermediate=True, leaky=False, batch_norm=True, drop_rate=0.1,
                             dilate_first=[1], dilate_second=[1])

            model.load_state_dict(torch.load(runDir + m + "/model/model.pt"))
            model.cuda()
            model.eval()
            inter_dir = runDir + m + savedir

            image = batch['input'][0].detach().numpy()
            image = image.transpose((1, 2, 0))
            image = Image.fromarray(np.uint8(image * 255), 'RGB')
            image.convert('RGB').save(inter_dir + "/input_" + str(i_batch) + ".png")

            image = batch['target'][0].detach().numpy()
            image = Image.fromarray(np.uint8(image[0] * 255), 'L')
            image.convert('RGB').save(inter_dir + "/target_" + str(i_batch) + ".png")

            inputs = batch['input'].cuda()
            targets = batch['target'].cuda()
            outputs = model(inputs)
            full_sigmoided_out = torch.sigmoid(outputs)
            
            # get the intermediate output of our model if applicable so we can compare it.
            if model_name == 'W-Net-Intermediate':
                outputs = full_sigmoided_out[0][0].unsqueeze(0).detach().cpu().numpy()
                out_image = Image.fromarray(np.uint8(outputs[0] * 255), 'L')
                out_image.convert('RGB').save(inter_dir + "/intermediate_" + str(i_batch) + ".png")

            # always save the final result
            outputs = full_sigmoided_out[0][-1].unsqueeze(0).detach().cpu().numpy()
            out_image = Image.fromarray(np.uint8(outputs[0] * 255), 'L')
            out_image.convert('RGB').save(inter_dir + "/final_" + str(i_batch) + ".png")

        # we dont need all images, just some
        if i_batch > 100:
            break