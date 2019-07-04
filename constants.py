'''
Constants used among multiple files. Such as names and directories.
'''

opNames = {2: "Adam", 1: "SGD", 3: "AdaDelta", 4: "RMSProp"}
modelNames = {1: "U-Net", 2: "R2U-Net", 3: "AttU-Net", 4: "R2AttU-Net", 5: "U-Net2",
              6: "W2-Net", 7: "W16-Net", 8: "PW-Net", 9: "Leaky-UNet", 10: "Leaky-R2UNet",
              11: "W64-Net", 12: "PCNet", 13: "DeeplabV3+", 14: "Leaky-R2UNet-NoSigmoid",
              15: "W-Net-Intermediate", 16: "R2-W-Net-Intermediate", 17: "W-Net", 18: "FCN32"}
dataNames = {0: "constant", 1: "Train", 2: "TrainAugmented", 3: "TrainAugmentedAdditional", 4: "TrainAugmentedRescaled"}
data_dir = 'data'
run_dir = 'runs'
save_iter_freq = 40
val_iter_freq = 80
val_count = 20
