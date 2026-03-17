from __future__ import print_function
import argparse
import os
import time
import math
import socket

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

import numpy as np
from PIL import Image
from tensorboardX import SummaryWriter

# --- load model arc ---
from model_archs.fremamba import FreMamba as net
from dataload.data import get_training_set, get_eval_set


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.epsilon * self.epsilon))
        return loss


# -------------------- args --------------------
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4)
parser.add_argument('--batchSize', type=int, default=4)
parser.add_argument('--nEpochs', type=int, default=5)
parser.add_argument('--snapshots', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--gpus', default=1, type=int)

parser.add_argument('--data_dir', type=str, default='/NTIRE2026/C14_RemSenseISR')
parser.add_argument('--val_dir', type=str, default='/NTIRE2026/C14_RemSenseISR')

parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='fmsr')
parser.add_argument('--patch_size', type=int, default=64)
parser.add_argument('--residual', type=bool, default=False)

parser.add_argument('--pretrained_sr', default='saved_models/fmsr/xx.pth')
parser.add_argument('--pretrained', type=bool, default=False)

parser.add_argument('--save_folder', default='saved_models/fmsr/')
parser.add_argument('--log_folder', default='tb_logs/fmsr/')

opt = parser.parse_args()
print(opt)

gpus_list = list(range(opt.gpus))
hostname = str(socket.gethostname())
cudnn.benchmark = True

device = torch.device("cuda:0" if (opt.gpu_mode and torch.cuda.is_available()) else "cpu")
torch.manual_seed(opt.seed)
if device.type == "cuda":
    torch.cuda.manual_seed(opt.seed)


# -------------------- folders --------------------
current_time = time.strftime("%H-%M-%S")
opt.save_folder = os.path.join(opt.save_folder, current_time)
opt.log_folder = os.path.join(opt.log_folder, current_time)

os.makedirs(opt.save_folder, exist_ok=True)
os.makedirs(opt.log_folder, exist_ok=True)

writer = SummaryWriter(opt.log_folder)


# -------------------- utils --------------------
def PSNR(pred, gt):
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def print_network(net_):
    num_params = sum(p.numel() for p in net_.parameters())
    print(net_)
    print('Total number of parameters: %f M' % (num_params / 1e6))


def print_log(*args):
    file_path = os.path.join(opt.save_folder, "train_log.txt")
    print(*args)
    with open(file_path, "a", encoding="utf-8") as f:
        print(*args, file=f)


def checkpoint(epoch):
    model_out_path = os.path.join(opt.save_folder, f"{opt.model_type}_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_out_path)
    print_log(f"✅ Checkpoint saved to: {model_out_path}")


# -------------------- data --------------------
print('===> Loading training datasets')
train_set = get_training_set(opt.data_dir, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
training_data_loader = DataLoader(
    dataset=train_set,
    num_workers=opt.threads,
    batch_size=opt.batchSize,
    shuffle=True,
    pin_memory=(device.type == "cuda")
)

print('===> Loading val datasets')
val_set = get_eval_set(opt.val_dir, opt.upscale_factor)
val_data_loader = DataLoader(
    dataset=val_set,
    num_workers=0,
    batch_size=1,
    shuffle=False,
    pin_memory=(device.type == "cuda")
)


# -------------------- model --------------------
print('===> Building model ', opt.model_type)
model = net()

if device.type == "cuda" and len(gpus_list) > 1:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

print('---------- Networks architecture -------------')
print_network(model)

model = model.to(device)

if opt.pretrained:
    model_name = opt.pretrained_sr
    print_log('Looking for pretrained at:', model_name)
    if os.path.exists(model_name):
        state_dict = torch.load(model_name, map_location=device)
        model.load_state_dict(state_dict)
        print_log('✅ Pre-trained SR model is loaded.')
    else:
        print_log('⚠️ Pretrained not found, training from scratch.')


# -------------------- optimizer + loss --------------------
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

mse_criterion = nn.MSELoss().to(device)
charbonnier_criterion = CharbonnierLoss().to(device)

# PSNR-oriented weights
mse_weight = 1.0
charb_weight = 0.05
ssim_weight = 0.02


# -------------------- train --------------------
best_epoch = 0
best_test_psnr = 0.0

for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    model.train()
    epoch_loss = 0.0

    for iteration, batch in enumerate(training_data_loader, 1):
        # train loader expected: (lr, gt) OR (lr, gt, name)
        if len(batch) == 3:
            lr, gt, _ = batch
        else:
            lr, gt = batch

        lr = lr.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        optimizer.zero_grad()

        t0 = time.time()

        prediction = model(lr)

        mse_loss = mse_criterion(prediction, gt)
        charb_loss = charbonnier_criterion(prediction, gt)
        ssim_loss = 1 - ssim(prediction, gt, data_range=1.0, size_average=True)

        loss = (mse_weight * mse_loss) + (charb_weight * charb_loss) + (ssim_weight * ssim_loss)

        loss.backward()
        optimizer.step()

        t1 = time.time()
        epoch_loss += loss.item()

        print_log(
            "===> Epoch[{}]({}/{}): Total: {:.6f} | MSE: {:.6f} | Charb: {:.6f} | SSIM: {:.6f} || Timer: {:.4f}s".format(
                epoch,
                iteration,
                len(training_data_loader),
                loss.item(),
                mse_loss.item(),
                charb_loss.item(),
                ssim_loss.item(),
                (t1 - t0)
            )
        )

    avg_loss = epoch_loss / max(1, len(training_data_loader))
    print_log(f"===> Epoch {epoch} Complete: Avg. Loss: {avg_loss:.6f}")
    writer.add_scalar('Avg_Loss', avg_loss, epoch)

    if epoch % opt.snapshots == 0:
        checkpoint(epoch)

    # -------------------- validation + preview save --------------------
    model.eval()

    preview_dir = os.path.join(opt.save_folder, "val_preview", f"epoch_{epoch}")
    os.makedirs(preview_dir, exist_ok=True)

    saved = 0
    max_save = 5

    avg_psnr_predicted = 0.0
    count_psnr = 0

    with torch.no_grad():
        for batch in val_data_loader:
            # val loader could be (lr, gt, name) OR (lr, name)
            if len(batch) == 3:
                lr, gt, name = batch
                has_gt = True
            elif len(batch) == 2:
                lr, name = batch
                gt = None
                has_gt = False
            else:
                print_log("⚠️ Unexpected val batch format:", type(batch), len(batch))
                continue

            lr = lr.to(device, non_blocking=True)

            t0 = time.time()
            pred = model(lr)
            t1 = time.time()

            # save preview images
            if saved < max_save:
                pred_img = pred.squeeze(0).detach().cpu().clamp(0, 1).numpy()
                pred_img = (pred_img * 255.0).astype(np.uint8).transpose(1, 2, 0)

                out_path = os.path.join(preview_dir, name[0])
                Image.fromarray(pred_img).save(out_path)
                print_log(f"Saved val preview: {out_path}")
                saved += 1

            # compute PSNR if GT exists
            if has_gt:
                gt = gt.to(device, non_blocking=True)

                pred_np = pred.squeeze(0).detach().cpu().clamp(0, 1).numpy() * 255.0
                gt_np = gt.squeeze(0).detach().cpu().clamp(0, 1).numpy() * 255.0

                psnr_val = PSNR(pred_np.astype(np.float32), gt_np.astype(np.float32))
                avg_psnr_predicted += psnr_val
                count_psnr += 1

                print_log(f"===> Val image {count_psnr} || Timer: {t1 - t0:.4f}s || PSNR: {psnr_val:.4f} dB")

    if count_psnr > 0:
        avg_test_psnr = avg_psnr_predicted / count_psnr

        if avg_test_psnr > best_test_psnr:
            best_epoch = epoch
            best_test_psnr = avg_test_psnr

        print_log(
            f"===> Epoch {epoch} Complete: Avg. PSNR: {avg_test_psnr:.4f} | Best Epoch: {best_epoch} | Best PSNR: {best_test_psnr:.4f}"
        )
        writer.add_scalar('Avg_PSNR', avg_test_psnr, epoch)
    else:
        print_log("===> Val done (no GT), previews saved only.")

    # learning rate schedule
    if (epoch + 1) % max(1, int(opt.nEpochs / 2)) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 2.0
        print_log('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

writer.close()

# Example:
# python train_4x.py --nEpochs 100 --batchSize 4 --threads 0