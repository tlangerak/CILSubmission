'''
Script that does all the evaluation for us. Gets called in train.py
'''

import os
from PIL import Image
import torch
import numpy as np
from skimage import io
from plotter_helper import *
from mask_to_submission import *
from architectures import *
import argparse
from constants import *
import utils

def evaluate(save_dir, model, threshold=0.5, intermediate_supervision=False):
    def toTensorRGB(image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image_1 = image.transpose((2, 0, 1))
        torch_image = torch.from_numpy(np.asarray([image_1])).type(torch.FloatTensor).cuda()
        return torch_image / 255.

    prediction_test_dir = save_dir + "/results/prediction/"
    for i in range(1, 224):
        filename = "data/original/test/test_" + str(i) + ".png"
        if not os.path.isfile(filename):
            continue
        print("Loading image {}".format(filename))

        # Only prediction
        img = io.imread(filename)
        input = toTensorRGB(img)
        output_model = model(input)
        if intermediate_supervision:
            sigmoided_out = torch.sigmoid(output_model)
            outputs = sigmoided_out[0][0].unsqueeze(0).detach().cpu().numpy()
            out_image = Image.fromarray(np.uint8(outputs[0] * 255), 'L')
            out_image.convert('RGB').save(prediction_test_dir + "intermediate_" + str(i) + ".png")
            outputs = sigmoided_out[0][1].unsqueeze(0).detach().cpu().numpy()
            out_image = Image.fromarray(np.uint8(outputs[0] * 255), 'L')
            out_image.convert('RGB').save(prediction_test_dir + "final_" + str(i) + ".png")

            output_model = output_model[:, -1, :, :].unsqueeze(1)

        outputs = output_model[0].view((img.shape[0], img.shape[1])).cpu().detach().numpy()
        outputs = [[0. if pixel < threshold else 255. for pixel in row] for row in outputs]
        outputs = np.asarray(outputs)
        out_image = Image.fromarray(outputs)
        out_image.convert('RGB').save(prediction_test_dir + "test_prediction_" + str(i) + ".png")
    return prediction_test_dir


def create_overlays(save_dir):
    for i in range(1, 224):
        filename = save_dir+"/results/prediction/" + "test_prediction_" + str(i) + ".png"
        filename_im = "data/original/test/test_" + str(i) + ".png"
        if not os.path.isfile(filename):
            continue

        # Only prediction
        prediction = Image.open(filename)
        prediction = prediction.convert('L')
        prediction = np.asarray(prediction)
        image = io.imread(filename_im)
        overlay_image = make_img_overlay(image, prediction)
        overlay_image.save(save_dir+"/results/overlay/" + str(i) + ".png")
    return


def mask2submission(submission_filename, prediction_directory):
    if not os.path.isdir(prediction_directory):
        print("No directory found. Run the predictions first")

    image_filenames = []
    for i in range(1, 244):
        filename = prediction_directory + "test_prediction_" + str(i) + ".png"
        if not os.path.isfile(filename):
            print(filename + " not found")
            continue
        image_filenames.append(filename)
        print(filename + " found")
    masks_to_submission(submission_filename, *image_filenames)
    return


if __name__ == "__main__":
    args = utils.parse_training_params().parse_args()
    argList = args.log_dir.split("_")
    # New version
    m = argList[0]
    threshold = args.threshold
    inter, model = select_model(m, args)
    runDir = "runs/"
    save_dir = runDir + args.log_dir
    model.load_state_dict(torch.load(runDir + args.log_dir + "/model/model.pt"))
    model.cuda()
    model.eval()

    predictions = evaluate(save_dir, model, args.threshold, inter)
    create_overlays(save_dir)
    mask2submission(save_dir + "/" + args.log_dir + ".csv", predictions)

