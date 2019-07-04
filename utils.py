'''
Some helper stuff. Nothing interesting but needed.
'''

import torch
import argparse
import numpy as np
import random
from constants import save_iter_freq, val_count, val_iter_freq
from itertools import repeat


def parse_training_params():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--learning_rate", nargs="?", type=float, dest="learning_rate", default=0.0005,
                        help="Learning rate of the model as float")
    parser.add_argument("--optimizer", nargs="?", type=int, dest="optimizer", default=2,
                        help="Optimizer to use: \n"
                             "1: SGD\n"
                             "2: Adam\n"
                             "3: AdaDelta\n"
                             "4: RMSProp")
    parser.add_argument("--data", nargs="?", type=int, dest="dataset", default=0,
                        help="Dataset to use: \n"
                             "0: Static training set with all data\n"
                             "1: Only our given train data\n"
                             "2: Augmented train data\n"
                             "3: Augmented train data + additional data (Toronto)\n"
                             "4: Augmented train data rescaled to (608,608)")
    parser.add_argument("--batch_size", nargs="?", type=int, dest="batch_size", default=1,
                        help="Batch size")
    parser.add_argument("--log_dir", nargs="?", type=str, dest="log_dir", default="model",
                        help="Log directory")
    parser.add_argument("--nr_epochs", nargs="?", type=int, dest="nr_epochs", default=1,
                        help="Number of epochs")
    parser.add_argument("--model", nargs="?", type=int, dest="model", default=16,
                        help="Model to run:\n"
                             "1:  U-Net\n"
                             "2:  R2U-Net\n"
                             "3:  AttU_Net\n"
                             "4:  R2AttU_Net\n"
                             "5:  U-Net 2 (Flurin)\n"
                             "6:  W2-Net\n"
                             "7:  W16-Net\n"
                             "8:  PW-Net\n"
                             "9:  Leaky-UNet\n"
                             "10: Leaky-R2Net\n"
                             "11: W64-Net"
                             "13: Deeplab V3+")
    parser.add_argument("--pre", nargs="?", type=str, dest="premodel", default="",
                        help="Premodel to use for the PW_Net")
    parser.add_argument("--thres", nargs="?", type=float, dest="threshold", default=0.6,
                        help="Threshold for the validation set and test set")
    parser.add_argument("--stop", nargs="?", type=int, dest="early_stopping", default=10,
                        help="Stops after the specified number of epochs if the validation loss does not decrease")
    parser.add_argument("--save_iter_freq", nargs="?", type=int, dest="save_iter_freq", default=save_iter_freq,
                        help="Every how many epoch we save")
    parser.add_argument("--val_iter_freq", nargs="?", type=int, dest="val_iter_freq", default=val_iter_freq,
                        help="How often to validate while training")
    parser.add_argument("--val_count", nargs="?", type=int, dest="val_count", default=val_count,
                        help="In every evaluation step, how many batches to evaluate")
    parser.add_argument("--jaccard", nargs="?", type=float, dest="jaccard", default=0.2,
                        help="Weight of the jaccard loss")
    parser.add_argument("--leaky", action='store_true', dest="leaky",
                        help="Use leaky ReLU/PreLU instead of ReLU")
    parser.add_argument("--batch_norm", action='store_true', dest="batch_norm",
                        help="Use batch normalization")
    parser.add_argument("--drop_rate", nargs="?", type=float, dest="drop_rate", default=0.0,
                        help="Use drop_rate of network (0.0 means no dropout)")
    parser.add_argument("--dilate_first", nargs="?", type=int, dest="dilate_first", default=1,
                        help="Dilation of convolutions in first U")
    parser.add_argument("--dilate_second", nargs="?", type=int, dest="dilate_second", default=1,
                        help="Dilation of convolutions in second U")
    return parser


def fix_seed():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def infinite_data_loader(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t
