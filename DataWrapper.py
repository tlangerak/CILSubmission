import os
from torch.utils.data import Dataset
from skimage import io
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
from skimage.transform import resize
from skimage.util import img_as_ubyte


class DataWrapper(Dataset):
    def __init__(self, input_dir, target_dir, cuda_available, m=0):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.cuda_available = cuda_available
        self.m = int(m)
        assert len([name for name in os.listdir(self.input_dir)]) == len([name for name in os.listdir(
            self.target_dir)]), "Input and target directory dont have the same number of entries"

    def __len__(self):
        return len([name for name in os.listdir(self.input_dir)])

    def __getitem__(self, idx):
        def toTensorRGB(self, image):
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1))
            if self.cuda_available:
                torch_image = torch.from_numpy(image).type(torch.FloatTensor)
            else:
                torch_image = torch.from_numpy(image).type(torch.FloatTensor)
            return torch_image / 255.

        def toTensorBW(self, image):
            torch_image = torch.from_numpy(image).view(image.shape[0], image.shape[1], 1)
            torch_image = torch_image.permute((2, 0, 1))

            if self.cuda_available:
                torch_image = torch_image.type(torch.FloatTensor)
            else:
                torch_image = torch_image.type(torch.FloatTensor)
            return torch_image / 255.

        input_img_name = os.path.join(self.input_dir, str(idx).zfill(5) + '.png')
        input_image = img_as_ubyte(io.imread(input_img_name))
        target_img_name = os.path.join(self.target_dir, str(idx).zfill(5) + '.png')
        target_image = img_as_ubyte(io.imread(target_img_name))

        #s cale images for specific baseline only.
        if self.m is 18:
            input_image = resize(input_image, (608, 608))
            target_image = resize(target_image, (608, 608))

        sample = {'input': toTensorRGB(self, input_image), 'target': toTensorBW(self, target_image)}
        return sample

    def image_size(self):
        input_img_name = os.path.join(self.input_dir, str(1).zfill(5) + '.png')
        input_image = io.imread(input_img_name)
        if self.m is 18:
            input_image = resize(input_image, (608, 608))
        return (input_image.shape[0], input_image.shape[1])


def create_batches(data, batch_size=10, num_workers=0):

    # create batches, shuffle needs to be false because we use the sampler.
    data = DataLoader(data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    return data
