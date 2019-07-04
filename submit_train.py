'''
OLD

First script to submit train jobs to leonhard.
'''

from constants import *
import os

# Variable parameters
learning_rates = [0.00025]
optimizers = [2]
data = [0]
batch = [1]
model = [16]
jaccard = [0.2]
leaky = [False]
batch_norm = [True]
drop_rate = [0.2]
dilate_first = [1]
dilate_second = [1]

# Fixed parameters
epochs = 10
early_stopping = 20
thres = 0.5
premodel = "model_R2U-Net_optimizer_Adam_dataset_TrainAugmentedAdditional_lr_5e-05_batch_1_1556798918"

# Leonhard Options
cpus = 2 
wall = "24:00"
memory = 4096
gpus = 1
file = "train.py"

for lr in learning_rates:
    for op in optimizers:
        for d in data:
            for b in batch:
                for m in model:
                    for j in jaccard:
                        for l in leaky:
                            for bn in batch_norm:
                                for dr in drop_rate:
                                    for df in dilate_first:
                                        for ds in dilate_second:
                                            log_dir = f"{modelNames[m]}_opt_{opNames[op]}_data_{dataNames[d]}_lr_{lr}" \
                                                + f"_bs_{b}_jac_{j}_" + ("leaky" if l else "reLU") + "_" \
                                                + ("use_bn" if bn else "no_bn") + f"_drop_{dr}_df_{df}_ds_{ds}"
                                            l_str = " --leaky" if l else ""
                                            bn_str = " --batch_norm" if bn else ""
                                            os.system(f"bsub -n {cpus} -W {wall} -R \"rusage[mem={memory},"
                                                      f" ngpus_excl_p={gpus}]\""
                                                      f" -R \"select[gpu_model0==GeForceGTX1080Ti]\""
                                                      f" python {file}"
                                                      f" --learning_rate {lr}"
                                                      f" --batch_size {b}"
                                                      f" --nr_epochs {epochs}"
                                                      f" --optimizer {op}"
                                                      f" --data {d}"
                                                      f" --log_dir {log_dir}"
                                                      f" --model {m}"
                                                      f" --thres {thres}"
                                                      f" --jaccard {j}" +
                                                      f"{l_str}"
                                                      f"{bn_str}"
                                                      f" --drop_rate {dr}"
                                                      f" --dilate_first {df}"
                                                      f" --dilate_second {ds}"
                                                      f" --stop {early_stopping}"
                                                      f" --pre {premodel}")
                                            os.system("sleep 1")
