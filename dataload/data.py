from os.path import join
from torchvision.transforms import Compose, ToTensor
# from dataload.dataset import DatasetTrainPair, DatasetValLR
from dataload.dataset import DatasetTrainPairCSV, DatasetValLR

def transform():
    return Compose([ToTensor()])

# def get_training_set(data_dir, upscale_factor, patch_size, data_augmentation):
#     # HR is in: data_dir/train
#     # LR is in: data_dir/train_LR_X4/X4  (or X2/X3 if factor changes)
#     hr_dir = join(data_dir, 'train')
#     lr_dir = join(data_dir, 'train_LR_X4', f'X{upscale_factor}')
#     return DatasetTrainPair(hr_dir, lr_dir, patch_size, upscale_factor, data_augmentation, transform=transform())

def get_training_set(data_dir, upscale_factor, patch_size, data_augmentation):
    csv_path = join(data_dir, "train_pairs.csv")
    return DatasetTrainPairCSV(csv_path, patch_size, upscale_factor, data_augmentation, transform=transform())

def get_eval_set(data_dir, upscale_factor):
    # val is LR only: data_dir/val_LR_X4/X4
    lr_dir = join(data_dir, 'testLR_X4', f'X{upscale_factor}')
    return DatasetValLR(lr_dir, transform=transform())

