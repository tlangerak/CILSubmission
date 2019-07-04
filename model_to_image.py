import torch
from DataWrapper import *
from PIL import Image
from architectures import *
import numpy as np
from skimage import io
import matplotlib.image as mpimg
from torch.utils.data import DataLoader

PREDICT_TEST = True
PREDICT_TRAINING = False

def toTensorRGB(image):
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    image_1 = image.transpose((2, 0, 1))
    torch_image = torch.from_numpy(np.asarray([image_1])).type(torch.FloatTensor)
    return torch_image / 255.


input_dir = 'test/'

model = UNet(3, 2)
model.load_state_dict(torch.load('models/lr_0.0001_bs_1_opt_2_1556294467.pt'))
model.eval()
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
#     model.cuda()
# else:
#     print("CUDA unavailable, using CPU!")

if PREDICT_TEST:
    prediction_test_dir = "predictions_test/scaled/"
    if not os.path.isdir(prediction_test_dir):
        os.mkdir(prediction_test_dir)

    for i in range(1, 224):
        filename = "test/test_" + str(i) + ".png"
        if not os.path.isfile(filename):
            continue
        print("Loading image {}".format(filename))

        # Only prediction
        img = io.imread(filename)

        input = toTensorRGB(img)
        outputs = model(input)

        # outputs = outputs.cpu()
        outputs = outputs[0].view((img.shape[0], img.shape[1])).detach().numpy()
        outputs = [[0. if pixel < 0.5 else 255. for pixel in row] for row in outputs]
        outputs = np.asarray(outputs)
        out_image = Image.fromarray(outputs)
        out_image.convert('RGB').save(prediction_test_dir + 'test_prediction_' + str(i) + ".png")

if PREDICT_TRAINING:
    prediction_training_dir = "predictions_training/"
    if not os.path.isdir(prediction_training_dir):
        os.mkdir(prediction_training_dir)

    for i in range(1, 101):
        filename = "train/input/satImage_" + str(i).zfill(3) + ".png"
        if not os.path.isfile(filename):
            continue
        print("Loading image {}".format(filename))

        img = io.imread(filename)

        input = toTensorRGB(img)
        outputs = model(input)

        # outputs = outputs.cpu()
        outputs = outputs[0].view((img.shape[0], img.shape[1])).detach().numpy()
        outputs = [[0. if pixel < 0.5 else 255. for pixel in row] for row in outputs]
        outputs = np.asarray(outputs)
        out_image = Image.fromarray(outputs)
        out_image.convert('RGB').save(prediction_training_dir + 'test_prediction_' + str(i) + ".png")


