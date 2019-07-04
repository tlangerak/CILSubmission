'''
Script to submit evaluation jobs to leonhard.
'''

import os

log_dirs = [
        'W-Net-Intermediate_opt_Adam_data_constant_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164639',
        'U-Net_opt_Adam_data_constant_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164642',
        'W-Net-Intermediate_opt_Adam_data_TrainAugmented_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164709',
        'U-Net_opt_Adam_data_TrainAugmented_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164709',
        'W-Net-Intermediate_opt_Adam_data_Train_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_164708',
        'U-Net_opt_Adam_data_Train_lr_0.00025_bs_4_jac_0.2_reLU_use_bn_drop_0.1_df_1_ds_1_Jun_27_162733']

# Leonhard Options
cpus = 4
wall = "20:00"
memory = 4096
gpus = 2
file = "evaluation.py"

runsDir = "runs/"
modelLoc = "/model/model.pt"

for entry in log_dirs:
    assert os.path.isdir(runsDir + entry), "Folder does not exist"
    assert os.path.isfile(runsDir + entry + modelLoc), "No model saved"

for entry in log_dirs:
    submit_string = "bsub -n " + str(cpus) + " -W " + wall + " -R \"rusage[mem=" + str(
        memory) + ", ngpus_excl_p=" + str(gpus) + "]\" python " + file + " --log_dir " + entry + " --batch_norm --drop_rate 0.1 --dilate_first 1 --dilate_second 1 --thres 0.5"
    os.system(submit_string)
    os.system("sleep 2")


