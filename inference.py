import os
import cv2
import lmdb
import math
import argparse
import numpy as np
from io import BytesIO
from PIL import Image
import pandas as pd
from tqdm import tqdm
import skvideo.io
import numpy as np


import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import soundfile as sf

from util.logging import init_logging, make_logging_dir
from util.distributed import init_dist
from util.trainer import get_model_optimizer_and_scheduler, set_random_seed, get_trainer
from util.distributed import master_only_print as print
from data.vox_video_dataset import VoxVideoDataset
from config import Config
import torchvision
from midian_pool import MedianPool2d

import warnings

warnings.filterwarnings("ignore")

median = MedianPool2d()
from u2net import U2NET

seg_net = U2NET(3, 1)
seg_net.load_state_dict(torch.load("u2net_human_seg.pth"))
seg_net.cuda()
seg_net.eval()

parser = argparse.ArgumentParser(description="Training")
parser.add_argument("--config", default="./config/face.yaml")
parser.add_argument("--name", default=None)
parser.add_argument(
    "--checkpoints_dir", default="result", help="Dir for saving logs and models."
)
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--cross_id", action="store_true")
parser.add_argument("--which_iter", type=int, default=None)
parser.add_argument("--no_resume", action="store_true")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--single_gpu", action="store_true")
parser.add_argument("--input", required=True, type=str)
parser.add_argument("--output_dir", type=str)

args = parser.parse_args()


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def seg_inference(net, input):

    input = input.clone()

    input[:, 2] = (input[:, 2] - 0.406) / 0.225
    input[:, 1] = (input[:, 1] - 0.456) / 0.224
    input[:, 0] = (input[:, 0] - 0.485) / 0.229

    input = torch.nn.functional.interpolate(input, size=(320, 320), mode="bicubic")
    input2 = input.flip(3)  # N, C, H, W

    # inference
    d1, d2, d3, d4, d5, d6, d7 = net(input)
    d11, d2, d3, d4, d5, d6, d7 = net(input2)

    d1 = (d1 + d11.flip(3)) / 2
    d1 = torch.nn.functional.interpolate(d1, size=(256, 256), mode="bicubic")

    # normalization
    pred = 1.0 - d1[:, 0, :, :]
    pred = normPRED(pred)

    # convert torch tensor to numpy array
    pred = pred.squeeze()
    pred = pred.cpu().data.numpy()

    del d1, d11, d2, d3, d4, d5, d6, d7

    return pred


def transfera2b(a, b):  # ref1 ref2 output
    with torch.no_grad():
        mb = torch.tensor(seg_inference(seg_net, b)).unsqueeze(0).unsqueeze(0).cuda()
        ma = torch.tensor(seg_inference(seg_net, a)).unsqueeze(0).unsqueeze(0).cuda()
        ma = torchvision.transforms.GaussianBlur(kernel_size=(7, 7), sigma=8.0)(ma)
        ma = (ma > 0.8).float()
        mb = (mb > 0.4).float()
        m = ma * mb
    return m, ma, mb


def write2video(results_dir, *video_list):
    cat_video = None

    for video in video_list:
        video_numpy = video[:, :3, :, :].cpu().float().detach().numpy()
        video_numpy = (np.transpose(video_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
        video_numpy = video_numpy.astype(np.uint8)
        cat_video = (
            np.concatenate([cat_video, video_numpy], 2)
            if cat_video is not None
            else video_numpy
        )
    gen_images = cat_video[:, :, 256:, ::-1]  # N, h, w, c
    gt_images = cat_video[:, :, :256, ::-1]

    out_name = results_dir + ".mp4"
    _, height, width, layers = gen_images.shape
    size = (width, height)
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*"mp4v"), 30, size)

    for i in range(gen_images.shape[0]):
        frame = gen_images[i]
        out.write(frame)
    out.release()


if __name__ == "__main__":
    set_random_seed(args.seed)
    opt = Config(args.config, args, is_train=False)

    opt.data.path = args.input
    if not args.single_gpu:
        opt.local_rank = args.local_rank
        init_dist(opt.local_rank)
        opt.device = torch.cuda.current_device()
    # create a visualizer
    date_uid, logdir = init_logging(opt)
    opt.logdir = logdir
    make_logging_dir(logdir, date_uid)

    # create a model
    net_G, net_G_ema, opt_G, sch_G = get_model_optimizer_and_scheduler(opt)

    trainer = get_trainer(opt, net_G, net_G_ema, opt_G, sch_G, None)

    current_epoch, current_iteration = trainer.load_checkpoint(opt, args.which_iter)
    net_G = trainer.net_G_ema.eval()

    output_dir = os.path.join(
        args.output_dir,
        "epoch_{:05}_iteration_{:09}".format(current_epoch, current_iteration),
    )
    os.makedirs(output_dir, exist_ok=True)
    opt.data.cross_id = args.cross_id
    dataset = VoxVideoDataset(opt.data, is_inference=True)

    with torch.no_grad():
        for video_index in tqdm(range(dataset.__len__())):
            data = dataset.load_next_video()
            input_source1 = data["source_image"][None].cuda()
            name = data["video_name"]

            loop_length = len(data["target_semantics"])

            output_images, gt_images, warp_images = [], [], []
            pred_list = []
            mask_list = []

            for frame_index in tqdm(range(loop_length)):
                target_semantic = data["target_semantics"][frame_index][None].cuda()

                down = input_source1.min()
                up = input_source1.max()
                input_source1 = (input_source1 - down) / (
                    up - down
                ) * 2 - 1  # =>[-1, 1]
                a = (input_source1.cuda() + 1) / 2
                output_dict1 = net_G(input_source1, target_semantic)
                output_dict1["fake_image"] = (
                    output_dict1["fake_image"].clamp_(-1, 1) + 1
                ) / 2 * (up - down) + down
                b = output_dict1["fake_image"]
                b = (b.clamp_(-1, 1) + 1) / 2
                m, ma, mb = transfera2b(a, b)
                if frame_index == 0:
                    m = ma
                    output_dict1["fake_image"] = input_source1.cuda()
                mask_list.append(m)
                m = torch.median(torch.cat(mask_list[-5:]), 0)[0]
                m = torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=8.0)(m)

                output_dict1["fake_image"] = (
                    output_dict1["fake_image"] * (1 - m) + input_source1 * m
                )

                pred_list.append(output_dict1["fake_image"].cpu().clamp_(-1, 1))
                if frame_index == 0:
                    mean = input_source1
                else:
                    mean = torch.mean(torch.cat(pred_list[-3:], 0), 0).cuda()

                output_images.append(output_dict1["fake_image"].cpu().clamp_(-1, 1))
                warp_images.append(output_dict1["warp_image"].cpu().clamp_(-1, 1))
                gt_images.append(data["target_image"][frame_index][None])

            gen_images = torch.cat(output_images, 0)
            gt_images = torch.cat(gt_images, 0)
            warp_images = torch.cat(warp_images, 0)

            write2video(
                "{}/{}".format(output_dir, name), gt_images, warp_images, gen_images
            )
            print("write results to video {}/{}".format(output_dir, name))
