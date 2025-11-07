# Copyright (c) Horizon Robotics. All rights reserved.
import copy
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

__all__ = [
    # "DeleteKeys",
    # "RenameKeys",
    # "AddKeys",
    # "CopyKeys",
    "SplitAttriute",
    "MergeAttribute",
]


# @OBJECT_REGISTRY.register
# class DeleteKeys(object):
#     """Delete keys in input dict.

#     Args:
#         keys: key list to detele

#     """

#     def __init__(self, keys: List[str]):
#         self.keys = keys

#     def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         for key in self.keys:
#             if key in data:
#                 data.pop(key)
#         return data


# @OBJECT_REGISTRY.register
# class RenameKeys(object):
#     """Rename keys in input dict.

#     Args:
#         keys: key list to rename, in "old_name | new_name" format.

#     """

#     def __init__(self, keys: List[str], split: str = "|"):
#         self.split = split
#         self.keys = keys

#     def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         for key in self.keys:
#             assert self.split in key
#             old_key, new_key = key.split(self.split)
#             old_key = old_key.strip()
#             new_key = new_key.strip()
#             if old_key in data:
#                 data[new_key] = data.pop(old_key)
#         return data


# @OBJECT_REGISTRY.register
# class AddKeys(object):
#     """Add new key-value in input dict.

#     Frequently used when you want to add dummy keys to data dict
#     but don't want to change code.

#     Args:
#         kv: key-value data dict.

#     """

#     def __init__(self, kv: Dict[str, Any]):
#         assert isinstance(kv, Mapping)
#         self._kv = kv

#     def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         for key in self._kv:
#             assert key not in data, f"{key} already exists in data."
#             data[key] = self._kv[key]
#         return data


# @OBJECT_REGISTRY.register
# class CopyKeys(object):
#     """Copy new key in input dict.

#     Frequently used when you want to cache keys to data dict
#     but don't want to change code.

#     Args:
#         kv: key-value data dict.

#     """

#     def __init__(self, keys: List[str], split: str = "|"):
#         self.split = split
#         self.keys = keys

#     def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         for key in self.keys:
#             assert self.split in key
#             old_key, new_key = key.split(self.split)
#             old_key = old_key.strip()
#             new_key = new_key.strip()
#             if old_key in data:
#                 data[new_key] = copy.deepcopy(data[old_key])
#         return data


# target_key =   ("bbox_posi_x", "bbox_posi_y", "bbox_posi_z",
#                 "bbox_wlh_l", "bbox_wlh_w", "bbox_wlh_h",
#                 "bbox_yaw",
#                 "bbox_speed_x", "bbox_speed_y", "bbox_speed_z")


class SplitAttriute:
    """Convert data to torch.Tensor."""

    def __init__(self, input_key, target_key):
        self.input_key = input_key
        self.target_key = target_key

    def __call__(self, data):
        return self.split_attribute(data)

    def split_attribute(self, data):

        for i in range(len(self.input_key)):
            target_key = self.target_key[i]

            bbox3d = data[self.input_key[i]]
            for index in range(
                len(target_key)
            ):  # bbox3d里面属性的放置顺序就和target_key一致
                key = target_key[index]
                for i in range(len(bbox3d)):
                    if key not in data:
                        data[key] = []
                    if len(bbox3d[i].shape) == 1:
                        # bbox3d[i]在0维度上增加一个维度
                        # 有可能是bbox3d[i]只有一个object,也有可能是在处理pose data
                        if bbox3d[i].shape[0] == 0:  # 如果是空的scene
                            data[key].append(np.array([]))
                        else:
                            data[key].append(
                                bbox3d[i][np.newaxis, :][:, index]
                            )
                            # save_path = 'output/pose_data.npy'
                            # np.save(save_path, bbox3d)

                    else:
                        data[key].append(bbox3d[i][:, index])

                # data[key] = torch.stack(data[key], dim=0)

        return data


class MergeAttribute:
    """Convert data to torch.Tensor."""

    def __init__(self, input_key, target_key, merage_name):
        self.input_key = input_key
        self.target_key = target_key
        self.merge_name = merage_name

    def __call__(self, data):
        return self.merge_attribute(data)

    def merge_attribute(self, data):

        for i in range(len(self.input_key)):
            target_key = self.target_key[i]
            merge_name = self.merge_name[i]

            bbox3d = []
            for index in range(len(data[self.input_key[0]])):  # Get batch_size
                bbox3d_index = []
                for key in target_key:
                    bbox3d_index.append(
                        data[key][index]
                    )  # data[key]: list, len(data[ley])=Block size, data[key][index]: np.array, shape=(N), N is the number of bbox3d
                bbox3d_index = np.stack(bbox3d_index, axis=1)
                bbox3d.append(bbox3d_index)

            for key in target_key:
                del data[key]

            data[merge_name] = bbox3d

        # bbox3d = torch.stack(bbox3d, dim=1)

        return data


def ego_transform(box3d, mat, ego):
    # pose_xyz = np.array(ego)[:, 0:3]
    trans_box = []

    if isinstance(box3d, list):
        yaw = np.array(ego)[:, 3]
        for i in range(len(box3d)):
            if i != (len(box3d) - 1):
                box3d_i = np.array(box3d[i])
                if box3d_i.shape[0] != 0:
                    ones = np.ones((box3d_i.shape[0], 1))
                    box3d_i_posi = np.concatenate(
                        (box3d_i[:, 0:3], ones), axis=-1
                    )
                    trans_box_posi = (
                        mat[i] @ box3d_i_posi.transpose(1, 0)
                    ).transpose(
                        1, 0
                    )  # 将当前帧的坐标转移到下一帧 0->1
                    trans_box_yaw = box3d_i[:, 6] - yaw[i + 1]
                    trans_box_lwh = box3d_i[:, 3:6]

                    box3d_i[:, 0:3] = trans_box_posi[:, :-1]
                    box3d_i[:, 3:6] = trans_box_lwh
                    box3d_i[:, 6] = trans_box_yaw
                trans_box.append(box3d_i)
            else:
                trans_box.append(np.array(box3d[i]))

    else:
        box3d_i = box3d
        yaw = ego[3]
        if box3d_i.shape[0] != 0:
            ones = np.ones((box3d_i.shape[0], 1))
            box3d_i_posi = np.concatenate((box3d_i[:, 0:3], ones), axis=-1)
            trans_box_posi = (mat @ box3d_i_posi.transpose(1, 0)).transpose(
                1, 0
            )  # 将当前帧的坐标转移到下一帧 0->1
            trans_box_yaw = box3d_i[:, 6] - yaw
            trans_box_lwh = box3d_i[:, 3:6]
            box3d_i[:, 0:3] = trans_box_posi[:, :-1]
            box3d_i[:, 3:6] = trans_box_lwh
            box3d_i[:, 6] = trans_box_yaw
            trans_box.append(box3d_i)
        else:
            trans_box.append(np.array(box3d[i]))
    return trans_box
    # pass
