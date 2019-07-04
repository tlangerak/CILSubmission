"""
Baseline for CIL project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss
"""

import constants
import utils
from plotter_helper import evaluation_side_by_side_plot_np 
from augment_data import create_data

import json
import argparse
import time
import os
import sys
import datetime
import tqdm
from tqdm import trange

import matplotlib.image as mpimg
import numpy
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from PIL import Image



parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--data", nargs="?", type=int, dest="dataset", default="1",
                    help="Dataset to use: \n"
                            "0: Static training set with all data\n"
                            "1: Only our given train data\n"
                            "2: Augmented train data\n"
                            "3: Augmented train data + additional data (Toronto)")
parser.add_argument("--batch_size", nargs="?", type=int, dest="batch_size", default="1",
                    help="Batch size")
parser.add_argument("--epochs", nargs="?", type=int, dest="epochs", default="100",
                    help="Number of epochs")
parser.add_argument("--log_dir", nargs="?", type=str, dest="log_dir", default="model",
                    help="Log directory")
args = parser.parse_args()

print(args)

NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TRAINING_SIZE = 80
VALIDATION_SIZE = 20  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = args.batch_size # 64
NUM_EPOCHS = args.epochs
train = True
RESTORE_MODEL = False # If True, restore existing model instead of train a new one
RECORDING_STEP = 1000
PREDICT_TEST = True


# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16

tf.app.flags.DEFINE_string('train_dir', 'models',
                           """Directory where to write event logs """
                           """and checkpoint.""")
FLAGS = tf.app.flags.FLAGS


# TODO: WRITE MY OWN SMALL PARSER
# args = utils.parse_training_params().parse_args()

timestamp = datetime.datetime.now().strftime("%b_%d_%H%M%S")
LOG_NAME = args.log_dir + "_" + timestamp
save_dir = os.path.join(constants.run_dir, LOG_NAME)

# Fix seed for reproducible results
utils.fix_seed()

# Create run_dir the first time
if not os.path.isdir(constants.run_dir):
    os.mkdir(constants.run_dir)

# Here everything is saved
time_before = datetime.datetime.now()
timestamp = time_before.strftime("%b_%d_%H%M%S")

LOG_NAME = args.log_dir + "_" + timestamp

save_dir = os.path.join(constants.run_dir, LOG_NAME)
os.mkdir(save_dir)

model_dir = os.path.join(save_dir, 'model')
os.mkdir(model_dir)


if RESTORE_MODEL:
    # specify model path manually
    model_dir = 'model_Jun_29_113535' + '/model'

results_dir = os.path.join(save_dir, 'results')
os.mkdir(results_dir)

print(f"Saving to {save_dir}")

os.mkdir(os.path.join(results_dir, "prediction"))
os.mkdir(os.path.join(results_dir, "overlay"))

log_dir = os.path.join('logdir', LOG_NAME)
print(f"Writing log files to {log_dir}")

# Initialize JSON Saver
json_saver = {'train_loss': dict(),
              'val_loss': dict(),
              'val_accuracy': dict(),
              'train_accuracy': dict(),
              'train_f1': dict(),
              'val_f1': dict(),
              'train_timestamps': dict(),
              'val_timestamps': dict(),
              'start_time': timestamp,
              'name': args.log_dir,
              'dataset': args.dataset,
              'number_epochs': args.epochs,
              'batch_size': args.batch_size}
with open(save_dir + '/data.json', 'w') as fp:
    json.dump(json_saver, fp, indent=2)


if args.dataset is 0:
    data_save_dir = os.path.join(constants.data_dir, "static")
    train_dir = os.path.join(data_save_dir, "split_train")
    val_dir = os.path.join(constants.data_dir+'/static/', "split_validate")
    if not os.path.isdir(data_save_dir):
        print("no static data directory found. Run create_dataset.py first.")
        exit(1)
else:
    if args.dataset is 1:
        data_save_dir = os.path.join(constants.data_dir, "original")
        train_dir = os.path.join(data_save_dir, "split_train")
        val_dir = os.path.join(constants.data_dir + '/original/', "split_validate")
    elif args.dataset is 2:
        data_save_dir = os.path.join(constants.data_dir, "augmented")
        train_dir = os.path.join(data_save_dir, "train")
        val_dir = os.path.join(constants.data_dir + '/static/', "split_validate")

    elif args.dataset is 3:
        create_data(train_set=['data/chicago', 'data/original/split_train'], train_set_amount=[1000, 1000],
                    eval_set=['data/original/split_validation'], eval_set_amount=[500],
                    augmentations=['rotate', 'flip', 'both'],
                    fraction_augmented_train=0.9, fraction_augmented_validation=0.5,
                    patched=True,
                    train_dir=train_dir, val_dir=val_dir)

    elif args.dataset is 4:
        create_data(train_set=['data/scaled'], train_set_amount=[1000],
                    eval_set=['data/scaled'], eval_set_amount=[200],
                    augmentations=[],
                    fraction_augmented_train=0.5, fraction_augmented_validation=0.5,
                    train_dir=train_dir, val_dir=val_dir)
    else:
        print("Not a training set")
        exit(1)

# Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_data(filename, indices):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print(filename)
    imgs = []
    for i in indices:
        # TODO: change this function to take the filename directly
        image_filename = os.path.join(filename, str(i).zfill(5) + '.png')
        if os.path.isfile(image_filename):
            # print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return numpy.asarray(data)


# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:
        return [0, 1]
    else:
        return [1, 0]


# Extract label images
def extract_labels(filename, indices):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    for i in indices:
        # TODO: change this function to take the filename directly
        image_filename = os.path.join(filename, str(i).zfill(5) + '.png')
        if os.path.isfile(image_filename):
            # print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)]
    data = numpy.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    labels = numpy.asarray([value_to_class(numpy.mean(data[i])) for i in range(len(data))])

    # Convert to dense 1-hot representation.
    return labels.astype(numpy.float32)


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
            100.0 *
            numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
            predictions.shape[0])


# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels[i] + ' ' + max_predictions[i])
    file.close()


# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels, binary=True):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if binary:
                if labels[idx][0] > 0.5:
                    l = 1
                else:
                    l = 0
            else:
                l = labels[idx][0]
            array_labels[j:j + w, i:i + h] = l
            idx = idx + 1
    return array_labels


def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg


def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img * PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def main(argv=None):  # pylint: disable=unused-argument

    if not os.path.isdir("models"):
        os.mkdir("models")

    train_data_filename = os.path.join(train_dir, 'input')
    train_labels_filename = os.path.join(train_dir, 'target')
    val_data_filename = os.path.join(val_dir, 'input')
    val_labels_filename = os.path.join(val_dir, 'target')
    print(val_data_filename)
    print(os.path.isdir(train_data_filename))

    
    ###############################################################################################################

    # TODO: put this somehow into the loop and load per step!!!
    
    print('filenames:', (train_data_filename, train_labels_filename, val_data_filename, val_labels_filename))
    train_indices = list(range(0, sum([len(files) for r, d, files in os.walk(train_data_filename)])))
    validation_indices = list(range(0, sum([len(files) for r, d, files in os.walk(val_data_filename)])))


    print('number of training images:', len(train_indices))
    print('number of validation images:', len(validation_indices))

    # Extract it into numpy arrays.

    train_data = extract_data(train_data_filename, train_indices)
    train_labels = extract_labels(train_labels_filename, train_indices)

    validation_data = extract_data(val_data_filename, validation_indices)
    validation_labels = extract_labels(val_labels_filename, validation_indices)

    validation_size = validation_labels.shape[0]
        
    ###############################################################################################################

    num_epochs = NUM_EPOCHS

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    print('Balancing train data...')
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(train_data.shape)
    train_data = train_data[new_indices, :, :, :]
    train_labels = train_labels[new_indices]

    train_size = train_labels.shape[0]

    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print('Number of data points per class: c0 = ' + str(c0) + ' c1 = ' + str(c1))

    # This is where train samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of train data at each
    # train step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMG_PATCH_SIZE, IMG_PATCH_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.float32,
                                    shape=(BATCH_SIZE, NUM_LABELS))
    train_all_data_node = tf.constant(train_data)

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=SEED))
    conv1_biases = tf.Variable(tf.zeros([32]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, 32, 64],
                            stddev=0.1,
                            seed=SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([int(IMG_PATCH_SIZE / 4 * IMG_PATCH_SIZE / 4 * 64), 512],
                            stddev=0.1,
                            seed=SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([512, NUM_LABELS],
                            stddev=0.1,
                            seed=SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

    # Make an image summary for 4d tensor image with index idx
    def get_image_summary(img, idx=0):
        V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        min_value = tf.reduce_min(V)
        V = V - min_value
        max_value = tf.reduce_max(V)
        V = V / (max_value * PIXEL_DEPTH)
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Make an image summary for 3d tensor image with index idx
    def get_image_summary_3d(img):
        V = tf.slice(img, (0, 0, 0), (1, -1, -1))
        img_w = img.get_shape().as_list()[1]
        img_h = img.get_shape().as_list()[2]
        V = tf.reshape(V, (img_w, img_h, 1))
        V = tf.transpose(V, (2, 0, 1))
        V = tf.reshape(V, (-1, img_w, img_h, 1))
        return V

    # Get prediction for given input image 
    def get_prediction(img, binary=True):
        data = numpy.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
        data_node = tf.constant(data)
        output = tf.nn.softmax(model(data_node))
        output_prediction = s.run(output)
        img_prediction = label_to_img(img.shape[0], img.shape[1],
                                      IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction, binary)

        return img_prediction

    # Get a concatenation of the prediction and target for given input file
    def get_prediction_with_groundtruth(filename, image_idx, binary=True):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img, binary)
        cimg = concatenate_images(img, img_prediction)

        return cimg

    # Get prediction overlaid on the original image for given input file
    def get_prediction_with_overlay(filename, image_idx, binary=True):

        imageid = "satImage_%.3d" % image_idx
        image_filename = filename + imageid + ".png"
        img = mpimg.imread(image_filename)

        img_prediction = get_prediction(img, binary)
        oimg = make_img_overlay(img, img_prediction)

        return oimg

    # We will replicate the model structure for the train subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        conv2 = tf.nn.conv2d(pool,
                             conv2_weights,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        pool2 = tf.nn.max_pool(relu2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

        # Uncomment these lines to check the size of each layer
        print('data ' + str(data.get_shape()))
        print ('conv ' + str(conv.get_shape()))
        print ('relu ' + str(relu.get_shape()))
        print ('pool ' + str(pool.get_shape()))
        print ('pool2 ' + str(pool2.get_shape()))

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool2.get_shape().as_list()
        reshape = tf.reshape(
            pool2,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during train only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        # if train:
        #    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
        out = tf.matmul(hidden, fc2_weights) + fc2_biases

        if train == True:
            summary_id = '_0'
            s_data = get_image_summary(data)
            filter_summary0 = tf.summary.image('summary_data' + summary_id, s_data)
            s_conv = get_image_summary(conv)
            filter_summary2 = tf.summary.image('summary_conv' + summary_id, s_conv)
            s_pool = get_image_summary(pool)
            filter_summary3 = tf.summary.image('summary_pool' + summary_id, s_pool)
            s_conv2 = get_image_summary(conv2)
            filter_summary4 = tf.summary.image('summary_conv2' + summary_id, s_conv2)
            s_pool2 = get_image_summary(pool2)
            filter_summary5 = tf.summary.image('summary_pool2' + summary_id, s_pool2)

        return out

    if train:
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            print(shape)
            print(len(shape))
            variable_parameters = 1
            for dim in shape:
                print(dim)
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
        print("tp:", total_parameters)


        # Training computation: logits + cross-entropy loss.
        logits = model(train_data_node, True)  # BATCH_SIZE*NUM_LABELS
        # print 'logits = ' + str(logits.get_shape()) + ' train_labels_node = ' + str(train_labels_node.get_shape())
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=train_labels_node))

        all_params_node = [conv1_weights, conv1_biases, conv2_weights, conv2_biases, fc1_weights, fc1_biases, fc2_weights,
                        fc2_biases]
        all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'fc1_weights', 'fc1_biases',
                            'fc2_weights', 'fc2_biases']
        all_grads_node = tf.gradients(loss, all_params_node)
        all_grad_norms_node = []
        for i in range(0, len(all_grads_node)):
            norm_grad_i = tf.global_norm([all_grads_node[i]])
            all_grad_norms_node.append(norm_grad_i)
            tf.summary.scalar(all_params_names[i], norm_grad_i)

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                        tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
        # Add the regularization term to the loss.
        loss += 5e-4 * regularizers

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            batch * BATCH_SIZE,  # Current index into the dataset.
            train_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        with tf.name_scope('performance'):
            tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
            tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

            tf_error_ph = tf.placeholder(tf.float32, shape=None, name='error_summary')
            tf_error_summary = tf.summary.scalar('error', tf_error_ph)

        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                            0.0).minimize(loss,
                                                            global_step=batch)

        performance_summaries = tf.summary.merge([tf_loss_summary, tf_error_summary])

        # Predictions for the minibatch, validation set and test set.
        train_prediction = tf.nn.softmax(logits)
        # We'll compute them only once in a while by calling their {eval()} method.
        train_all_prediction = tf.nn.softmax(model(train_all_data_node))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Create a local session to run this computation.
    with tf.Session() as s:

        if RESTORE_MODEL:
            print('Restoring model from ', model_dir)
            # Restore variables from disk.
            tf.saved_model.loader.load(s, [], export_dir = 'runs/' + model_dir)
            print("Model restored.")

        else:
            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()

            # Build the summary operation based on the TF collection of Summaries.
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(model_dir, graph_def=s.graph_def)
            print('Initialized!')
            # Loop through train steps.
            print('Total number of iterations = ' + str(int(num_epochs * train_size / BATCH_SIZE)))
            params = numpy.sum([numpy.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
            print("Number of trainable parameters: {}".format(params))
            json_saver['n_parameters'] = int(params)

            training_indices = range(train_size)
            val_indices = range(validation_size)

            best_error = numpy.Inf
            
            # Timestamp before training start
            time_before = datetime.datetime.now()

            for iepoch in range(num_epochs):

                # Permute train indices
                perm_indices = numpy.random.permutation(training_indices)

                f1_train = []
                train_loss = []
                train_accuracy = []

                with trange(int(train_size / BATCH_SIZE)) as t:
                #for step in range(int(train_size / BATCH_SIZE)):

                    for step in t:
                        
                        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
                        batch_indices = perm_indices[offset:(offset + BATCH_SIZE)]

                        # Compute the offset of the current minibatch in the data.
                        # Note that we could use better randomization across epochs.
                        batch_data = train_data[batch_indices, :, :, :]
                        batch_labels = train_labels[batch_indices]
                        # This dictionary maps the batch data (as a numpy array) to the
                        # node in the graph is should be fed to.
                        feed_dict = {train_data_node: batch_data,
                                    train_labels_node: batch_labels}

                        # Run the graph and fetch some of the nodes.
                        _, l, lr, predictions = s.run(
                            [optimizer, loss, learning_rate, train_prediction],
                            feed_dict=feed_dict)

                        train_loss.append(l)
                        train_accuracy.append(error_rate(predictions, batch_labels))

                        #avg_f1_train = f_tot / BATCH_SIZE
                        avg_train_loss = np.mean(train_loss)
                        avg_train_accuracy = 100 - np.mean(train_accuracy)

                        if step % constants.save_iter_freq == 0 and step != 0:
                            duration = (datetime.datetime.now() - time_before).total_seconds()
                            
                            #f1 = f1_score(y_true=batch_labels, y_pred=predictions)
                            
                            # Add iter values to JSON
                            json_saver['train_loss'][str(step)] = float(avg_train_loss)
                            #json_saver['train_f1'][str(step)] = float(avg_f1_train)
                            json_saver['train_accuracy'][str(step)] = float(avg_train_accuracy)
                            json_saver['train_timestamps'][str(step)] = duration
                        
                        if step % constants.val_iter_freq == 0:
                            # Run validation --> make sure the number of batches in the validation-set is greater than 1/val_iter_freq * train_dataset_size
                            val_accuracy = []
                            val_loss = []
                            
                            for val_step in range(constants.val_count):
                                offset = (val_step * BATCH_SIZE) % (validation_size - BATCH_SIZE)
                                batch_indices = val_indices[offset:(offset + BATCH_SIZE)]

                                batch_data = validation_data[batch_indices, :, :, :]
                                batch_labels = validation_labels[batch_indices]

                                feed_dict = {train_data_node: batch_data,
                                            train_labels_node: batch_labels}

                                l, predictions = s.run([loss, train_prediction], feed_dict=feed_dict)
                                val_accuracy.append(error_rate(predictions, batch_labels))
                                val_loss.append(l)
                                
                            avg_val_loss = numpy.mean(val_loss)
                            avg_val_accuracy = 100 - numpy.mean(val_accuracy)
                            #avg_f1 = numpy.mean(f1)
                            
                            # Add iter values to JSON
                            json_saver['val_loss'][str(step)] = float(avg_val_loss)
                            #json_saver['val_f1'][str(step)] = float(avg_f1)
                            json_saver['val_accuracy'][str(step)] = float(avg_val_accuracy)
                            duration = (datetime.datetime.now() - time_before).total_seconds()
                            json_saver['val_timestamps'][str(step)] = duration

                        '''
                        if step > constants.val_iter_freq:
                        # Run predictions on validation set
                        print("Epoch {} \t Mean train loss: {} \t Mean validation loss: {}".format(iepoch, np.mean(train_loss), avg_val_loss))
                        '''

                        summ = s.run(performance_summaries,
                                        feed_dict={tf_loss_ph: avg_val_loss, tf_error_ph: avg_val_accuracy})

                        summary_writer.add_summary(summ, iepoch)

                        # Save the variables to disk.
                        if avg_val_accuracy < best_error:
                            # TODO: fix this to be the correct path
                            save_path = saver.save(s, model_dir + "/model.ckpt")
                            print("Model saved in file: %s" % save_path)
                            best_error = avg_val_accuracy
                        
                        # Save json to file
                        with open(save_dir + '/data.json', 'w') as fp:
                            json.dump(json_saver, fp, indent=2)
        
        if PREDICT_TEST:
            print("Running prediction on test set")
            # For evaluation you use the provided original data_folder 'test' with 'test_i.png' test images.
            test_dir = 'data/original/test/'

            prediction_test_dir = save_dir + "/results/"
            if not os.path.isdir(prediction_test_dir):
                os.mkdir(prediction_test_dir)

            for i in range(1, 224):
                filename = test_dir +"test_" + str(i) + ".png"
                if not os.path.isfile(filename):
                    continue
                print("Loading image {}".format(filename))

                # Only prediction
                img = mpimg.imread(filename)
                test_prediction = get_prediction(img, binary=False)
                img8 = img_float_to_uint8(test_prediction)
                image = Image.fromarray(img8, 'L')
                image.save(prediction_test_dir + "test_prediction_" + str(i) + ".png")

if __name__ == '__main__':
    tf.app.run()
