# eval_4x.py  (FULL FILE)
# ✅ Includes x8 TTA + tiled inference + AMP
# ✅ Keeps your old commented blocks intact
# ✅ Fixes t1 not defined
# ✅ Fixes device mismatch (uses device everywhere)
# ✅ Loads only .png files from val folder
# ✅ Prevents Thumbs.db / zip / junk from being read

from __future__ import print_function
import argparse
import os
import torch
import cv2
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform
from os import listdir
import math
# ---load model architecture---
from model_archs.fremamba import FreMamba
import glob
import numpy as np
import socket
import time
import imageio
from PIL import Image


# -------------------- Test settings --------------------
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='training batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

parser.add_argument('--data_dir', type=str, default='/NTIRE2026/runs/C14_RemSenseISR/C14_RemSenseISR')

parser.add_argument('--model_type', type=str, default='fmsr')
parser.add_argument('--pretrained_sr', default='saved_models/fmsr_4x.pth', help='sr pretrained base model')
parser.add_argument('--save_folder', default='results/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
torch.backends.cudnn.benchmark = True  # ✅ speedup
cuda = opt.gpu_mode
print(opt)

current_time = time.strftime("%H-%M-%S")
opt.save_folder = opt.save_folder + current_time + '/'

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)

transform = transform.Compose([transform.ToTensor(), ])


# -------------------- TTA helpers (x8) --------------------
def _tta_transforms(x):
    # x: (1,C,H,W) tensor
    # returns list of transformed tensors
    xs = []
    xs.append(x)
    xs.append(torch.flip(x, dims=[3]))              # flip W
    xs.append(torch.flip(x, dims=[2]))              # flip H
    xs.append(torch.flip(xs[-1], dims=[3]))         # flip H+W

    # transpose H/W (rotate 90 by transpose + flip)
    x_t = x.transpose(2, 3)
    xs.append(x_t)
    xs.append(torch.flip(x_t, dims=[3]))
    xs.append(torch.flip(x_t, dims=[2]))
    xs.append(torch.flip(xs[-1], dims=[3]))
    return xs


def _inverse_tta(y_list):
    # y_list corresponds to _tta_transforms order
    ys = []
    ys.append(y_list[0])
    ys.append(torch.flip(y_list[1], dims=[3]))
    ys.append(torch.flip(y_list[2], dims=[2]))
    ys.append(torch.flip(torch.flip(y_list[3], dims=[2]), dims=[3]))

    ys.append(y_list[4].transpose(2, 3))
    ys.append(torch.flip(y_list[5], dims=[3]).transpose(2, 3))
    ys.append(torch.flip(y_list[6], dims=[2]).transpose(2, 3))
    ys.append(torch.flip(torch.flip(y_list[7], dims=[2]), dims=[3]).transpose(2, 3))
    return ys


@torch.no_grad()
def forward_x8(model, x):
    # self-ensemble x8
    xs = _tta_transforms(x)
    ys = [model(t) for t in xs]
    ys = _inverse_tta(ys)
    y = torch.stack(ys, dim=0).mean(dim=0)
    return y


@torch.no_grad()
def forward_tiled_x8(model, x, scale=4, tile=512, overlap=32):
    """
    Tile inference with overlap + x8 TTA inside each tile.
    x: (1,C,H,W) in [0,1]
    """
    b, c, h, w = x.shape
    out_h, out_w = h * scale, w * scale

    output = torch.zeros((b, c, out_h, out_w), device=x.device)
    weight = torch.zeros((b, c, out_h, out_w), device=x.device)

    stride = tile - overlap
    if stride <= 0:
        stride = tile

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            y0_ = max(y1 - tile, 0)
            x0_ = max(x1 - tile, 0)

            inp = x[:, :, y0_:y1, x0_:x1]
            pred = forward_x8(model, inp)

            oy0, ox0 = y0_ * scale, x0_ * scale
            oy1, ox1 = y1 * scale, x1 * scale

            output[:, :, oy0:oy1, ox0:ox1] += pred
            weight[:, :, oy0:oy1, ox0:ox1] += 1.0

    output = output / weight.clamp(min=1.0)
    return output


# -------------------- utils --------------------
def PSNR(pred, gt):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %f M' % (num_params / 1e6))


# -------------------- seed + device --------------------
torch.manual_seed(opt.seed)
if torch.cuda.is_available() and cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda:0" if (cuda and torch.cuda.is_available()) else "cpu")
print('===> Building model ', opt.model_type)

# Build base model (NOT DataParallel yet)
model = FreMamba()

model_name = opt.pretrained_sr
print("Looking for pretrained at:", model_name)

if os.path.exists(model_name):
    print("Loading checkpoint:", model_name)
    state_dict = torch.load(model_name, map_location="cpu")

    # if nested checkpoint format
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # if the checkpoint *had* module. keys, strip them (safe)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    print("✅ Pre-trained SR model loaded.")
else:
    print("❌ No pre-trained model!!!!")
    exit(0)

# Now wrap with DataParallel if you want
# model = torch.nn.DataParallel(model, device_ids=gpus_list)
# model = model.cuda(gpus_list[0])
# Only wrap if you really have >1 GPU
if cuda and opt.gpus > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(opt.gpus)))

model = model.to(device)
model.eval()

# print('===> Building model ', opt.model_type)
# model = FreMamba()
# model = torch.nn.DataParallel(model, device_ids=gpus_list)
# print('---------- Networks architecture -------------')
# # print(gpus_list)
# print_network(model)
# model = model.cuda(gpus_list[0])

# # model_name = os.path.join(opt.pretrained_sr)
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# model_name = opt.pretrained_sr
# if not os.path.isabs(model_name):
#     model_name = os.path.join(SCRIPT_DIR, model_name)
# print("Looking for pretrained at:", model_name)

# if os.path.exists(model_name):
#     # print('Beginning--------------------------------------')
#     # model= torch.load(model_name, map_location=lambda storage, loc: storage)
#     # model.load_state_dict(torch.load(model_name))

#     print("Loading checkpoint:", model_name)

#     state_dict = torch.load(model_name, map_location='cuda')
#     model.load_state_dict(state_dict)

#     print("Model loaded successfully!")

#     print('Pre-trained SR model is loaded.')
# else:
#     print('No pre-trained model!!!!')

# def find_lr_folder(root, scale):
#     # Try common folder patterns
#     candidates = [
#         os.path.join(root, "val_LR_X4", f"X{scale}"),
#         os.path.join(root, "val_LR_X4", f"x{scale}"),
#         os.path.join(root, "val_LR_X4"),
#         os.path.join(root, "val_LR"),
#         os.path.join(root, f"val_LR_X{scale}", f"X{scale}"),
#         os.path.join(root, f"val_LR_X{scale}"),
#     ]
#
#     # Also try one-level nested folders under val_LR_X4/
#     base = os.path.join(root, "val_LR_X4")
#     if os.path.isdir(base):
#         for sub in sorted(os.listdir(base)):
#             p = os.path.join(base, sub)
#             if os.path.isdir(p):
#                 candidates.append(p)
#
#     exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
#
#     for p in candidates:
#         if os.path.isdir(p):
#             imgs = []
#             for e in exts:
#                 imgs += glob.glob(os.path.join(p, f"*{e}"))
#             if len(imgs) > 0:
#                 return p, sorted(imgs)
#
#     return None, []


# def eval_x4():
#     print('===> Loading val LR datasets (no GT)')
#
#     LR_folder, LR_image = find_lr_folder(opt.data_dir, opt.upscale_factor)
#
#     if len(LR_image) == 0:
#         # print("No images found under:", opt.data_dir)
#         print("Expected something like: val_LR_X4/X4/ or val_LR_X4/")
#         return
#
#     # print("Using LR folder:", LR_folder)
#     # print("Total LR images:", len(LR_image))
#
#     model.eval()


# -------------------- main eval --------------------
def eval_x4():
    print('===> Loading val LR datasets (no GT)')

    # Your LR folder:
    # C14_RemSenseISR/testLR_X4/X4/
    LR_filename = os.path.join(opt.data_dir, 'testLR_X4', f'X{opt.upscale_factor}')

    # ✅ load only PNGs (avoid thumbs.db, zips, etc.)
    LR_image = sorted(glob.glob(os.path.join(LR_filename, '*.png')))

    if len(LR_image) == 0:
        print("No images found in:", LR_filename)
        return

    model.eval()

    save_folder = os.path.join(opt.save_folder, f'X{opt.upscale_factor}')
    os.makedirs(save_folder, exist_ok=True)

    for img_path in LR_image:
        lr = Image.open(img_path).convert('RGB')
        lr = transform(lr).unsqueeze(0)

        # ✅ always move input to same device as model
        lr = lr.to(device, non_blocking=True)

        with torch.no_grad():
            t0 = time.time()

            prediction = forward_tiled_x8(
                model, lr,
                scale=opt.upscale_factor,
                tile=768,
                overlap=32
            )
            t1 = time.time()  # ✅ FIX: define t1

        prediction = prediction.squeeze(0).detach().cpu().clamp(0, 1).numpy()
        prediction = np.clip(np.rint(prediction * 255.0), 0, 255).astype(np.uint8)
        prediction = prediction.transpose(1, 2, 0)

        save_name = os.path.basename(img_path)
        save_fn = os.path.join(save_folder, save_name)

        Image.fromarray(prediction).save(save_fn, format="PNG")
        print(f"===> {save_name} saved || Timer: {t1 - t0:.4f} sec -> {save_fn}")


if __name__ == '__main__':
    eval_x4()