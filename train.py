'''
MAIN
everything that is needed to train something.  includes flags for models, datasets, learning rates and more.
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
from tensorboardX import SummaryWriter
import sys
from skimage import io
import json
import argparse
import time
from evaluation import *

from sklearn.metrics import f1_score
from losses import *
import utils
import os
import constants
import datetime
from tqdm import trange, tqdm

# Parse training parameters
args = utils.parse_training_params().parse_args()

# Fix for slow convergence https://github.com/pytorch/pytorch/issues/15054
if args.dilate_first or args.dilate_second:
    torch.backends.cudnn.benchmark = True

# Fix seed for reproducible results
utils.fix_seed()

# Create run_dir the first time
if not os.path.isdir(constants.run_dir):
    os.mkdir(constants.run_dir)

# Here everything is saved
timestamp = datetime.datetime.now().strftime("%b_%d_%H%M%S")
LOG_NAME = args.log_dir + "_" + timestamp
save_dir = os.path.join(constants.run_dir, LOG_NAME)
os.mkdir(save_dir)
model_dir = os.path.join(save_dir, 'model')
os.mkdir(model_dir)
results_dir = os.path.join(save_dir, 'results')
os.mkdir(results_dir)
print(f"Saving to {save_dir}")

os.mkdir(os.path.join(results_dir, "prediction"))
os.mkdir(os.path.join(results_dir, "overlay"))
inter_dir = os.path.join(save_dir, "inter")
os.mkdir(inter_dir)

log_dir = os.path.join('logdir', LOG_NAME)
writer = SummaryWriter(log_dir)
print(f"Writing log files to {log_dir}")

# Initialize JSON Saver
json_saver = {'train_loss': dict(),  # Total batch training loss every save_iter_freq
              'train_f1': dict(),  # Batch F1 score every save_iter_freq
              'train_accuracy': dict(),  # Batch accuracy every save_iter_freq
              'train_timestamps': dict(),  # Timestamps ever save_iter_freq
              'val_loss': dict(),  # Total loss every val_iter_freq averaged over val_count many samples
              'val_f1': dict(),  # F1 score every val_iter_freq averaged over val_count many samples
              'val_accuracy': dict(),  # Accuracy every val_iter_freq averaged over val_count many samples
              'val_timestamps': dict(),  # Timestamps every val_iter_freq
              'start_time': timestamp,
              'name': args.log_dir,
              'dataset': args.dataset,
              'number_epochs': args.nr_epochs,
              'optimizer': args.optimizer,
              'learning_rate': args.learning_rate,
              'train_set': args.dataset,
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
        val_dir = os.path.join(constants.data_dir + '/static/', "split_validate")
    elif args.dataset is 2:
        data_save_dir = os.path.join(constants.data_dir, "augmented")
        train_dir = os.path.join(data_save_dir, "split_train")
        val_dir = os.path.join(constants.data_dir + '/static/', "split_validate")

    # elif args.dataset is 3:
    #     create_data(train_set=['data/chicago', 'data/original/split_train'], train_set_amount=[1000, 1000],
    #                 eval_set=['data/original/split_validation'], eval_set_amount=[500],
    #                 augmentations=['rotate', 'flip', 'both'],
    #                 fraction_augmented_train=0.9, fraction_augmented_validation=0.5,
    #                 patched=True,
    #                 train_dir=train_dir, val_dir=val_dir)
    #
    # elif args.dataset is 4:
    #     create_data(train_set=['data/scaled'], train_set_amount=[1000],
    #                 eval_set=['data/scaled'], eval_set_amount=[200],
    #                 augmentations=[],
    #                 fraction_augmented_train=0.5, fraction_augmented_validation=0.5,
    #                 train_dir=train_dir, val_dir=val_dir)
    else:
        print("Not a training set")
        exit(1)

print(train_dir)
train_data = DataWrapper(os.path.join(train_dir, "input/"), os.path.join(train_dir, "target/"),
                         torch.cuda.is_available(), args.model)
val_data = DataWrapper(os.path.join(val_dir, "input/"), os.path.join(val_dir, "target/"),
                       torch.cuda.is_available(), args.model)

nr_training_samples = train_data.__len__()
nr_validation_samples = val_data.__len__()
nr_training_iterations = nr_training_samples // args.batch_size

# Size of validation images
VAL_IMAGE_SIZE = val_data.image_size()
BATCH_VAL_IMAGE_SIZE = (args.batch_size, 1, 400, 400)
if args.model == 18:
    BATCH_VAL_IMAGE_SIZE = (args.batch_size, 1, 608, 608)

# Select model based on arguments
intermediate_supervision, model = select_model(modelNames[args.model], args)

# Transfer model to GPU if possible
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    model.cuda()
else:
    print("CUDA unavailable, using CPU!")

# Declare loss
criterion = nn.BCEWithLogitsLoss().cuda()

# Select optimizer
if args.optimizer is 1:
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=0.0005)

elif args.optimizer is 2:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

elif args.optimizer is 3:
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0.0005)

elif args.optimizer is 4:
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.99, eps=1e-08,
                                    weight_decay=0.0005,
                                    momentum=0.9,
                                    centered=False)
else:
    print("Not a valid optimizer")
    exit(1)

# Count number of trainable parameters
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Number of trainable paramters in model:", params)
writer.add_text("Trainable Parameters", str(params))
json_saver['n_parameters'] = int(params)

# Variables for model selection
best_f1, best_epoch, best_loss = 0.0, 0, 1.

# Thresholding
threshold1 = nn.Threshold(args.threshold, 0.0).cuda()
threshold2 = nn.Threshold(-args.threshold, -1.0).cuda()

# Infinite validation set to validate while training
static_validation_data = create_batches(val_data, batch_size=1)
infinite_validation_data = utils.infinite_data_loader(static_validation_data)

# Timestamp before training start
time_before = datetime.datetime.now()

# Training loop
with trange(args.nr_epochs) as t:
    for current_epoch in t:

        # Progress bar
        t.set_description("Epochs")

        # Create data batches
        training_data = create_batches(train_data, batch_size=args.batch_size)

        # Enable training mode
        model.train()
        
        epoch_val_f1 = 0.0
        epoch_val_loss =0.0
        epoch_val_count = 0

        with tqdm(enumerate(training_data), total=nr_training_iterations) as tb:
            for i_batch, batch in tb:
                # Training iteration count
                iter_count = current_epoch * (nr_training_samples / args.batch_size) + i_batch

                # For progress bar
                tb.set_description("Training Iterations")

                # Reset gradients
                optimizer.zero_grad()

                # Forward pass
                print(batch['input'].size())
                inputs = batch['input'].cuda()
                targets = batch['target'].cuda()
                outputs = model(inputs)

                # Backward pass
                # Intermediate supervision for W-Net
                if intermediate_supervision:
                    loss = criterion(outputs[:, 0, :, :].unsqueeze(1), targets) + \
                           criterion(outputs[:, 1, :, :].unsqueeze(1), targets)
                    jac = jaccard(utils.flatten(targets),
                                  torch.sigmoid(utils.flatten(outputs[:, -1, :, :].unsqueeze(1))))
                else:
                    loss = criterion(outputs, targets)
                    jac = jaccard(utils.flatten(targets),
                                  torch.sigmoid(utils.flatten(outputs)))

                total_loss = loss + args.jaccard * jac
                total_loss.backward()
                optimizer.step()

                # Save progress
                if i_batch % args.save_iter_freq == 0 and i_batch != 0:

                    # Apply sigmoid to output
                    sigmoided_out = torch.sigmoid(outputs)

                    # Convert to binary labels
                    thresholded_out = -threshold2(-threshold1(sigmoided_out))

                    # Calculate accuracy
                    accuracy = torch.mean((thresholded_out[:, -1, :, :].unsqueeze(1) - targets) ** 2)
                    accuracy = accuracy.cpu().detach().numpy()

                    # Calculate f1 score (on the CPU)
                    ground_truth = batch['target'].cpu().view(BATCH_VAL_IMAGE_SIZE).detach().numpy()
                    predictions = thresholded_out[:, -1, :, :].unsqueeze(1).cpu().view(BATCH_VAL_IMAGE_SIZE).detach().numpy()
                    f1 = f1_score(y_true=ground_truth.flatten().astype(int),
                                  y_pred=predictions.flatten().astype(int))

                    # Get losses on the CPU
                    iter_loss = loss.cpu().detach().numpy()
                    iter_jac_loss = jac.cpu().detach().numpy()
                    iter_total_loss = total_loss.cpu().detach().numpy()

                    # Write losses to tensorboard
                    writer.add_scalar('train/Total Loss', float(iter_total_loss), iter_count)
                    writer.add_scalar('train/BCE Loss', float(iter_loss), iter_count)
                    writer.add_scalar('train/Jaccard Loss', float(iter_jac_loss), iter_count)
                    writer.add_scalar('train/Accuracy', float(accuracy), iter_count)
                    writer.add_scalar('train/F1 Score', float(f1), iter_count)

                    # Add images to tensorboard
                    writer.add_image('train/input', batch['input'][0], iter_count)
                    # Special case for intemediate supervision
                    if intermediate_supervision:
                        writer.add_image('train/intermediate_out', sigmoided_out[0][0].unsqueeze(0), iter_count)
                        writer.add_image('train/intermediate_prediction', thresholded_out[0][0].unsqueeze(0),
                                         iter_count)
                        writer.add_image('train/output', sigmoided_out[0][1].unsqueeze(0), iter_count)
                        writer.add_image('train/prediction', thresholded_out[0][1].unsqueeze(0), iter_count)

                    else:
                        writer.add_image('train/output', sigmoided_out[0], iter_count)
                        writer.add_image('train/prediction', thresholded_out[0], iter_count)
                    writer.add_image('train/target', batch['target'][0], iter_count)

                    # Add iter values to JSON
                    json_saver['train_loss'][str(iter_count)] = float(iter_total_loss)
                    json_saver['train_f1'][str(iter_count)] = float(f1)
                    json_saver['train_accuracy'][str(iter_count)] = float(accuracy)
                    duration = (datetime.datetime.now() - time_before).total_seconds()
                    json_saver['train_timestamps'][str(current_epoch)] = duration

                # Validate
                if i_batch % args.val_iter_freq == 0:

                    time_after = datetime.datetime.now()
                    epoch_val_count += 1

                    # Switch to evaluation mode
                    model.eval()
                    with torch.no_grad():

                        # Store average values
                        val_avg_loss, val_avg_f1, val_avg_acc = 0.0, 0.0, 0.0

                        for val_step in range(args.val_count):
                            val_batch = next(iter(infinite_validation_data))

                            # Forward pass
                            val_inputs = val_batch['input'].cuda()
                            val_targets = val_batch['target'].cuda()
                            val_outputs = model(val_inputs)

                            # Calculate loss
                            if intermediate_supervision:
                                val_loss = criterion(val_outputs[:, 0, :, :].unsqueeze(1), val_targets) + \
                                           criterion(val_outputs[:, 1, :, :].unsqueeze(1), val_targets)
                                val_jac = jaccard(utils.flatten(val_targets),
                                                  torch.sigmoid(utils.flatten(val_outputs[:, -1, :, :].unsqueeze(1))))

                            else:
                                val_loss = criterion(val_outputs, val_targets)
                                val_jac = jaccard(utils.flatten(val_targets),
                                                  torch.sigmoid(utils.flatten(val_outputs)))
                            val_total_loss = val_loss + args.jaccard * val_jac
                            val_avg_loss += val_total_loss

                            # Special case for intermediate supervision
                            if intermediate_supervision:
                                full_outputs = val_outputs
                                full_sigmoided_out = torch.sigmoid(full_outputs)
                                outputs = full_sigmoided_out[0][0].unsqueeze(0).detach().cpu().numpy()
                                out_image = Image.fromarray(np.uint8(outputs[0] * 255), 'L')
                                out_image.convert('RGB').save(inter_dir + "/intermediate_" + str(val_step) + ".png")
                                outputs = full_sigmoided_out[0][1].unsqueeze(0).detach().cpu().numpy()
                                out_image = Image.fromarray(np.uint8(outputs[0] * 255), 'L')
                                out_image.convert('RGB').save(inter_dir + "/final_" + str(val_step) + ".png")

                                val_outputs = val_outputs[:, -1, :, :].unsqueeze(1)

                            # Apply sigmoid
                            val_sigmoided_out = torch.sigmoid(val_outputs)

                            # Threshold predictions
                            val_thresholded_out = -threshold2(-threshold1(val_sigmoided_out))

                            # Calculate accuracy
                            accuracy = torch.mean((val_thresholded_out - val_targets) ** 2).cpu().detach().numpy()
                            val_avg_acc += accuracy

                            # Calculate f1 score (on the CPU)
                            val_ground_truth = val_batch['target'].cpu().view(VAL_IMAGE_SIZE).detach().numpy()
                            val_predictions = val_thresholded_out.cpu().view(VAL_IMAGE_SIZE).detach().numpy()
                            val_f1 = f1_score(y_true=val_ground_truth.flatten().astype(int),
                                              y_pred=val_predictions.flatten().astype(int))
                            val_avg_f1 += val_f1

                            # Save first image during validation
                            if val_step == 0:
                                writer.add_image('valid/input', val_batch['input'][0], iter_count)
                                writer.add_image('valid/output', val_sigmoided_out[0], iter_count)
                                writer.add_image('valid/prediction', val_thresholded_out[0], iter_count)
                                writer.add_image('valid/target', val_batch['target'][0], iter_count)




                        val_avg_loss /= args.val_count
                        val_avg_acc /= args.val_count
                        val_avg_f1 /= args.val_count
                        
                        epoch_val_f1 += val_avg_f1
                        epoch_val_loss += val_avg_loss
                        # Write average validation score to tensorboard
                        writer.add_scalar('valid/Total Loss', val_avg_loss, iter_count)
                        writer.add_scalar('valid/Accuracy', val_avg_acc, iter_count)
                        writer.add_scalar('valid/F1 Score', val_avg_f1, iter_count)

                        # Write average validation score to JSON
                        json_saver['val_loss'][str(iter_count)] = float(val_avg_loss)
                        json_saver['val_accuracy'][str(iter_count)] = float(val_avg_acc)
                        json_saver['val_f1'][str(iter_count)] = float(val_avg_f1)
                        json_saver['val_timestamps'][str(iter_count)] = (time_after - time_before).total_seconds()

                    # Switch back to training mode
                    model.train()

        epoch_val_f1 /= epoch_val_count
        epoch_val_loss /= epoch_val_count
        # Save model if better
        if epoch_val_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
            json_saver['model_save_iter'] = current_epoch
            best_loss = epoch_val_loss
            best_epoch = current_epoch

        # Save json to file
        with open(save_dir + '/data.json', 'w') as fp:
            json.dump(json_saver, fp, indent=2)

        # Early stopping
        if current_epoch - best_epoch > args.early_stopping:
            print("Early stopping. Best iter was {}, now at {}".format(best_epoch, current_epoch))
            break

# Save
time_after_all = datetime.datetime.now()
json_saver['end_time'] = time_after_all.strftime("%b_%d_%H%M%S")
json_saver['run_time'] = (time_after_all - time_before).total_seconds()
json_saver['last_iter'] = iter_count

# Save json file
with open(save_dir + '/data.json', 'w') as fp:
    json.dump(json_saver, fp, indent=2)

print("Starting evaluation")

# Load model, put it on the GPU and set it to evaluation mode
model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt')))
model.cuda()
model.eval()

# Evaluate
predictions = evaluate(save_dir, model, args.threshold, intermediate_supervision)
# Create overlays
create_overlays(save_dir)
# Create submission file
mask2submission(save_dir + "/" + LOG_NAME + ".csv", predictions)
