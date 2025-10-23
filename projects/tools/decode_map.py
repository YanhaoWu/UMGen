import math
import os
import pickle
import cv2
import torch


import numpy as np
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from projects.tokenizer.vq_model import (
    get_map_normvq_dim16_res256_f8,
    get_normvq_dim16_res512_f16,
)

np.set_printoptions(suppress=True)


def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def to_rgb(x, seed=0):
    torch.manual_seed(seed)
    weights = torch.randn(3, x.shape[1], 1, 1).to(x)
    x = F.conv2d(x, weight=weights)
    x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
    return x


def postprocess_image(image):
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return image


def add_frame_number(
    image, frame_idx, pose_value=None, font_scale=0.2, cond_num=20
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    if pose_value is not None:
        pose_value = trunc(pose_value, 2)
        if frame_idx < cond_num:
            # text = f"F: {frame_idx} {pose_value}"
            text = f"F: {frame_idx}   [dx, dy, dh]: {pose_value}"
            text_color = (0, 255, 0)
        else:
            # text = f"F: {frame_idx} {pose_value}"
            text = f"F: {frame_idx}   [dx, dy, dh]: {pose_value}"
            text_color = (0, 0, 255)
    else:
        if frame_idx < cond_num:
            text = f"Frame: {frame_idx}"
            text_color = (0, 255, 0)
        else:
            text = f"Frame: {frame_idx}"
            text_color = (0, 0, 255)

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = 10
    text_y = 10 + text_size[1]
    image = cv2.putText(
        image.copy(),
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
    )
    return image


def write_video_single(
    images_0,
    pose_values=None,
    save_path=None,
    cond_num=20,
    font_scale=0.5,
    fps=10,
    h=256,
    w=256 * 1,
):
    # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    images_0 = rearrange(images_0, "b c h w -> b h w c")
    for i in range(len(images_0)):
        img_0 = images_0[i]
        img_0 = postprocess_image(img_0)
        pose_value = pose_values[i] if pose_values is not None else None
        img_0 = add_frame_number(
            img_0,
            i,
            cond_num=cond_num,
            pose_value=pose_value,
            font_scale=font_scale,
        )
        out.write(img_0)
    out.release()


class Mapdecoder(nn.Module):
    def __init__(self, ckpt, device="cuda"):
        super(Mapdecoder, self).__init__()

        map_autoencoder = get_map_normvq_dim16_res256_f8(
            ckpt=ckpt, device=device
        )
        map_autoencoder.eval()
        self.map_autoencoder = map_autoencoder

    def decode_maps(self, map, H=32, W=32):

        with torch.no_grad():
            if isinstance(map, np.ndarray):
                map_tokens = torch.from_numpy(map).cuda()
            else:
                map_tokens = map.cuda()
            map_tokens = map_tokens.squeeze(0)
            T, S = map_tokens.shape
            map_tokens = rearrange(
                map_tokens,
                "b (h w) -> b h w",
                b=T,
                h=H,
                w=W,
            )
            rec_maps = []
            for i in range(math.ceil(T / 20)):
                curr_token = map_tokens[i * 20 : (i + 1) * 20]
                recons = self.map_autoencoder.decode_code(curr_token)
                rec_map = to_rgb(recons)
                mid_img_h, mid_img_w = int(rec_map.shape[2] / 2), int(
                    rec_map.shape[3] / 2,
                )
                # rec_map[:, :, mid_img_h-4:mid_img_h+4, mid_img_w-4:mid_img_w+4] = 1.0           # 绘制ego
                rec_maps.append(rec_map)
            rec_maps = torch.cat(rec_maps, dim=0)
        return rec_maps


class Imagedecoder(nn.Module):
    def __init__(self, ckpt, device="cuda"):
        super(Imagedecoder, self).__init__()

        img_autoencoder = get_normvq_dim16_res512_f16(device=device, ckpt=ckpt).eval()
        self.img_autoencoder = img_autoencoder

    def decode_images(self, image, H=16, W=32):

        if isinstance(image, np.ndarray):
            img_tokens = torch.from_numpy(image).cuda()
        else:
            img_tokens = image.cuda()
        img_tokens = img_tokens.squeeze(0)
        if len(img_tokens.shape) == 1:
            img_tokens = img_tokens.unsqueeze(0)
        T, S = img_tokens.shape
        H, W = 16, 32
        img_tokens = rearrange(
            img_tokens,
            "b (h w) -> b h w",
            b=T,
            h=H,
            w=W,
        )
        rec_images = []
        for i in range(math.ceil(T / 20)):
            curr_token = img_tokens[i * 20 : (i + 1) * 20]
            quant = self.img_autoencoder.indices_to_quant(curr_token)
            rec_img = self.img_autoencoder.decode(quant)
            rec_images.append(rec_img)
        rec_images = torch.cat(rec_images, dim=0)

        return rec_images


def decode_tokens(
    data_path, save_name, fps=5, cond_num=19, ckpt=None, generate_video=True
):
    map_autoencoder = get_map_normvq_dim16_res256_f8(ckpt=ckpt)
    map_autoencoder.eval()

    with torch.no_grad():
        if isinstance(data_path, str):
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        else:
            data = data_path
        if "map" in data:
            # decode map
            map_tokens = torch.from_numpy(data["map"]).cuda()
            map_tokens = map_tokens.squeeze(0)
            T, S = map_tokens.shape
            H, W = 32, 32
            map_tokens = rearrange(
                map_tokens,
                "b (h w) -> b h w",
                b=T,
                h=H,
                w=W,
            )
            rec_maps = []
            for i in range(math.ceil(T / 20)):
                curr_token = map_tokens[i * 20 : (i + 1) * 20]
                recons = map_autoencoder.decode_code(curr_token)
                rec_map = to_rgb(recons)
                mid_img_h, mid_img_w = int(rec_map.shape[2] / 2), int(
                    rec_map.shape[3] / 2,
                )
                rec_map[
                    :,
                    :,
                    mid_img_h - 4 : mid_img_h + 4,
                    mid_img_w - 4 : mid_img_w + 4,
                ] = 1.0  # 绘制ego
                rec_maps.append(rec_map)
            rec_maps = torch.cat(rec_maps, dim=0)

        if "image" in data:
            # decode image
            img_autoencoder = get_normvq_dim16_res512_f16().eval()

            img_tokens = torch.from_numpy(data["image"]).cuda()
            img_tokens = img_tokens.squeeze(0)
            T, S = img_tokens.shape
            H, W = 16, 32
            img_tokens = rearrange(
                img_tokens,
                "b (h w) -> b h w",
                b=T,
                h=H,
                w=W,
            )
            rec_images = []
            for i in range(math.ceil(T / 20)):
                curr_token = img_tokens[i * 20 : (i + 1) * 20]
                quant = img_autoencoder.indices_to_quant(curr_token)
                rec_img = img_autoencoder.decode(quant)
                rec_images.append(rec_img)
            rec_images = torch.cat(rec_images, dim=0)
            rec_maps = torch.cat([rec_images, rec_maps], dim=3)

        # # decode pose
        pose_tokens = torch.from_numpy(data["pose"])
        pose_tokens = pose_tokens.squeeze(0)
        pose_values = ego_pose_tokenizer.decode(pose_tokens)
        pose_values = ego_norm.unnormalize_ego(pose_values)  # changed
        if not isinstance(pose_values, np.ndarray):  # 会以list的形式出现
            pose_values = np.array(pose_values)

        pose_values[:, 2] = pose_values[:, 2] * 180 / np.pi

        video_save_path = save_name
        H, W = rec_maps.shape[2], rec_maps.shape[3]

        if generate_video:
            write_video_single(
                rec_maps,
                pose_values=pose_values,
                save_path=video_save_path,
                cond_num=cond_num,
                font_scale=0.5,
                fps=fps,
                h=H,
                w=W,
            )
