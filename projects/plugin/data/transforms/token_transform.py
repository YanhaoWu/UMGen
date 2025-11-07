# Copyright (c) Horizon Robotics. All rights reserved.
import copy
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

__all__ = [
    "ego_transform",
    "generate_radius_mask",
    "fill_radius_mask",
    "pose_transform",
]


def encode_tokens(trans_box):

    print("encode_tokens")


def fill_radius_mask(radius_mask, num_boxes=60):
    for i in range(len(radius_mask)):
        mask_i = radius_mask[i]
        mask_i = np.concatenate(
            (
                mask_i,
                np.zeros(
                    (num_boxes - mask_i.shape[0], num_boxes - mask_i.shape[0]),
                    dtype=bool,
                ),
            ),
            axis=0,
        )
        radius_mask[i] = mask_i

    return radius_mask


def generate_radius_mask(bbox3d, radius=3):

    if isinstance(bbox3d, list):
        radius_mask = []
        for bbox3d_i in bbox3d:
            box_posi = bbox3d_i[:, 0:3]
            # 计算两两之间的距离
            dist_matrix = torch.cdist(
                torch.tensor(box_posi), torch.tensor(box_posi)
            )
            radius_mask_i = dist_matrix <= radius
            radius_mask.append(radius_mask_i.numpy())

    elif isinstance(bbox3d, torch.Tensor):

        box_posi = bbox3d[:, 0:3]
        # 计算两两之间的距离
        dist_matrix = torch.cdist(box_posi, box_posi)
        radius_mask = dist_matrix <= radius
        radius_mask = radius_mask

    return radius_mask


def pose_transform(box3d, pose):
    trans_box = []
    if pose[0].reshape(1, -1).shape[1] == 3:
        xyz_dim = 2  # 包括z
    else:
        xyz_dim = 3  # 不包括z

    if isinstance(box3d, list):
        yaw = np.array(pose)[:, -1]
        for i in range(len(box3d)):
            if i != (len(box3d) - 1):
                box3d_i = np.array(box3d[i])
                if box3d_i.shape[0] != 0:
                    ones = np.ones((box3d_i.shape[0], 1))
                    box3d_i_posi = np.concatenate(
                        (box3d_i[:, 0:xyz_dim], ones), axis=-1
                    )

                    translation = -pose[i + 1, :2]
                    theta = -yaw[i + 1]
                    rotation_matrix = np.array(
                        [
                            [np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1],
                        ]
                    )
                    rotated_point = (
                        rotation_matrix @ box3d_i_posi.transpose(1, 0)
                    ).transpose(1, 0)
                    trans_box_posi = rotated_point[:, :xyz_dim] + translation

                    # trans_box_posi = box3d_i_posi + pose[i + 1, :xyz_dim]

                    trans_box_yaw = box3d_i[:, 6] + theta
                    trans_box_lwh = box3d_i[:, 3:6]

                    box3d_i[:, 0:xyz_dim] = trans_box_posi
                    box3d_i[:, 3:6] = trans_box_lwh
                    box3d_i[:, 6] = trans_box_yaw
                trans_box.append(box3d_i)
            else:
                trans_box.append(np.array(box3d[i]))

    else:
        box3d_i = box3d
        yaw = pose[-1]
        if box3d_i.shape[0] != 0:

            ones = np.ones((box3d_i.shape[0], 1))
            box3d_i_posi = np.concatenate((box3d_i[:, 0:2], ones), axis=-1)

            translation = -pose[:2]
            theta = -yaw
            rotation_matrix = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1],
                ]
            )
            rotated_point = (
                rotation_matrix @ box3d_i_posi.transpose(1, 0)
            ).transpose(1, 0)
            trans_box_posi = rotated_point[:, :2] + translation

            # trans_box_posi = box3d_i_posi + pose[i + 1, :xyz_dim]

            trans_box_yaw = box3d_i[:, 6] + theta
            trans_box_lwh = box3d_i[:, 3:6]

            box3d_i[:, 0:2] = trans_box_posi  #
            box3d_i[:, 3:6] = trans_box_lwh
            box3d_i[:, 6] = trans_box_yaw

            trans_box = box3d_i
        else:
            trans_box = box3d_i

    return trans_box


def ego_transform(box3d, mat, ego):
    trans_box = []
    if mat[0].reshape(1, -1).shape[1] == 3:
        xyz_dim = 2  # 包括z
    else:
        xyz_dim = 3  # 不包括z

    if isinstance(box3d, list):
        yaw = np.array(ego)[:, -1]
        for i in range(len(box3d)):
            if i != (len(box3d) - 1):
                box3d_i = np.array(box3d[i])
                if box3d_i.shape[0] != 0:
                    ones = np.ones((box3d_i.shape[0], 1))
                    box3d_i_posi = np.concatenate(
                        (box3d_i[:, 0:xyz_dim], ones), axis=-1
                    )
                    trans_box_posi = (
                        mat[i] @ box3d_i_posi.transpose(1, 0)
                    ).transpose(
                        1, 0
                    )  # 将当前帧的坐标转移到下一帧 0->1
                    trans_box_yaw = box3d_i[:, 6] - yaw[i]
                    trans_box_lwh = box3d_i[:, 3:6]

                    box3d_i[:, 0:xyz_dim] = trans_box_posi[:, :-1]
                    box3d_i[:, 3:6] = trans_box_lwh
                    box3d_i[:, 6] = trans_box_yaw
                trans_box.append(box3d_i)
            else:
                trans_box.append(np.array(box3d[i]))

    else:
        box3d_i = box3d
        yaw = ego[-1]
        if box3d_i.shape[0] != 0:
            ones = np.ones((box3d_i.shape[0], 1))
            box3d_i_posi = np.concatenate(
                (box3d_i[:, 0:xyz_dim], ones), axis=-1
            )
            trans_box_posi = (mat @ box3d_i_posi.transpose(1, 0)).transpose(
                1, 0
            )  # 将当前帧的坐标转移到下一帧 0->1
            trans_box_yaw = box3d_i[:, 6] - yaw
            trans_box_lwh = box3d_i[:, 3:6]
            box3d_i[:, 0:xyz_dim] = trans_box_posi[:, :-1]
            box3d_i[:, 3:6] = trans_box_lwh
            box3d_i[:, 6] = trans_box_yaw
            trans_box.append(box3d_i)
        else:
            trans_box.append(np.array(box3d_i))

    return trans_box
    # pass


class FourierEncoding(object):
    def __init__(self, N_freqs, in_channels=1, logscale=True, sparate=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(FourierEncoding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)
        self.sparate = sparate

    def __call__(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        out = []
        shape_out = x.shape[1]
        for i in range(shape_out):
            out.append([])

        if self.sparate:
            for freq in self.freq_bands:
                for func in self.funcs:
                    freq_out = func(freq * x)
                    for i in range(shape_out):
                        out[i] += [freq_out[:, i]]

        out = [torch.stack(sublist, dim=1).unsqueeze(1) for sublist in out]
        encoded_data = torch.cat(out, dim=1)

        return encoded_data
