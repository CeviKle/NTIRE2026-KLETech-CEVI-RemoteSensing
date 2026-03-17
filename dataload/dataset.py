import csv
import torch.utils.data as data
from os import listdir
from os.path import join, basename
from PIL import Image, ImageOps
import random

def is_image_file(filename):
    filename = filename.lower()
    if filename.startswith("._"):  # mac metadata files like ._0.png
        return False
    return any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg"])

def safe_load_img(path):
    # returns PIL image or None (if corrupted / not readable)
    try:
        with Image.open(path) as img:
            return img.convert('RGB')
    except Exception:
        return None

def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    # PIL size is (width, height)
    iw, ih = img_in.size

    ip = patch_size              # LR patch size
    tp = scale * patch_size      # HR patch size

    # if image is smaller than required patch, resize it first
    if iw < ip or ih < ip:
        new_w = max(iw, ip)
        new_h = max(ih, ip)

        img_in = img_in.resize((new_w, new_h), Image.BICUBIC)
        img_tar = img_tar.resize((new_w * scale, new_h * scale), Image.BICUBIC)

        iw, ih = img_in.size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    tx, ty = scale * ix, scale * iy

    # PIL crop expects (left, upper, right, lower)
    img_in = img_in.crop((ix, iy, ix + ip, iy + ip))
    img_tar = img_tar.crop((tx, ty, tx + tp, ty + tp))

    return img_in, img_tar

def augment(img_in, img_tar, flip_h=True, rot=True):
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)

    if random.random() < 0.5:
        img_in = ImageOps.mirror(img_in)
        img_tar = ImageOps.mirror(img_tar)

    if rot:
        r = random.choice([0, 90, 180, 270])
        if r != 0:
            img_in = img_in.rotate(r, expand=False)
            img_tar = img_tar.rotate(r, expand=False)

    return img_in, img_tar

class DatasetTrainPair(data.Dataset):
    """
    Train dataset:
      HR: data_dir/train
      LR: data_dir/train_LR_X4/X4
    Must match by filename (0.png with 0.png, etc.)
    """
    def __init__(self, HR_dir, LR_dir, patch_size, upscale_factor, data_augmentation, transform=None):
        super().__init__()
        self.HR_dir = HR_dir
        self.LR_dir = LR_dir
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation

        hr_files = sorted([x for x in listdir(HR_dir) if is_image_file(x)])
        lr_files = set([x for x in listdir(LR_dir) if is_image_file(x)])

        # keep only files that exist in BOTH hr and lr
        self.pairs = [x for x in hr_files if x in lr_files]

        if len(self.pairs) == 0:
            raise RuntimeError(f"No matching HR/LR pairs found.\nHR: {HR_dir}\nLR: {LR_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        fname = self.pairs[index]
        hr_path = join(self.HR_dir, fname)
        lr_path = join(self.LR_dir, fname)

        target = safe_load_img(hr_path)
        inp = safe_load_img(lr_path)

        # if any file corrupted, skip by moving to next index
        if target is None or inp is None:
            new_index = (index + 1) % len(self.pairs)
            return self.__getitem__(new_index)

        inp, target = get_patch(inp, target, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            inp, target = augment(inp, target)

        if self.transform:
            inp = self.transform(inp)
            target = self.transform(target)

        return inp, target

class DatasetValLR(data.Dataset):
    """
    Validation dataset:
      LR only: data_dir/testLR_X4/X4
    returns (lr_tensor, filename)
    """
    def __init__(self, LR_dir, transform=None):
        super().__init__()
        self.LR_dir = LR_dir
        self.transform = transform

        self.lr_files = sorted([join(LR_dir, x) for x in listdir(LR_dir) if is_image_file(x)])

        if len(self.lr_files) == 0:
            raise RuntimeError(f"No LR images found in: {LR_dir}")

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, index):
        lr_path = self.lr_files[index]
        img = safe_load_img(lr_path)

        if img is None:
            new_index = (index + 1) % len(self.lr_files)
            return self.__getitem__(new_index)

        if self.transform:
            img = self.transform(img)

        return img, basename(lr_path)

class DatasetTrainPairCSV(data.Dataset):
    """
    Train dataset from CSV:
    CSV columns: lr_path, hr_path, name
    """
    def __init__(self, csv_path, patch_size, upscale_factor, data_augmentation, transform=None):
        super().__init__()
        self.csv_path = csv_path
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.data_augmentation = data_augmentation
        self.transform = transform

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.rows = [r for r in reader]

        if len(self.rows) == 0:
            raise RuntimeError(f"CSV has 0 rows: {csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        lr_path = r["lr_path"]
        hr_path = r["hr_path"]

        target = safe_load_img(hr_path)
        inp = safe_load_img(lr_path)

        # if corrupted, skip
        if target is None or inp is None:
            new_index = (index + 1) % len(self.rows)
            return self.__getitem__(new_index)

        inp, target = get_patch(inp, target, self.patch_size, self.upscale_factor)

        if self.data_augmentation:
            inp, target = augment(inp, target)

        if self.transform:
            inp = self.transform(inp)
            target = self.transform(target)

        return inp, target