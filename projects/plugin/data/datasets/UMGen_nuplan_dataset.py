import logging
import os
import pickle
import random
from typing import List, Optional, Union
import cv2
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils import data
from projects.registry import MODELS, DATASETS
logger = logging.getLogger(__name__)
ego_whl = {"w": 2.297, "l": 5.176, "h": 1.777}
#
@DATASETS.register_module()
class NuPlanTokenDataset(data.Dataset):
    """NuPlan dataset for token-based GPT training.

    Args:
        data_root: Path or a list of paths that contain the pickle files,
            or the pickle file itself.
        training: Whether to use the training mode.
        block_size: The num frames in a sequence.
        views: The views to use.
        sampling_gap: The sampling gap between frames.
        transforms: The transforms to apply to the data.
        img_transform: The transforms to load and augment the image.
        inference_flag: Whether to be in the inference mode.
        start_index: The start index of frame when sampling clip
        sp_list: The special list for training. sample a part of training dataset
        sample_img: Whether to sample the image.
        return_ori_image: Whether to return the original image. in diffusion training, when there is no vae latent, we need to return the original image
        sample_others: Whether to sample the other modalities experpt image.
        return_scene_name: Whether to return the scene name.
        return_lidar_box: Whether to return the 2D lidar box which is projected from 3D lidar box.
        return_lidar_box_id: Whether to return the 2D lidar box id.
        control_test: Whether to use the control test mode. If true, will load from the control test dataset.
        vae_token_path: The path of vae token.
        long_sceniors: Since we split the sceniors into several parts, setting this to True will concatenate the sceniors.
    """

    def __init__(
        self,
        data_root: Union[str, List[str]],
        training: bool,
        block_size: int,
        views: List[str],
        categories_file: str = "projects/configs/category.txt",
        sampling_gap: Optional[int] = 1,
        transforms: Optional[List] = None,
        img_transform=None,
        inference_flag=False,
        start_index=10,
        sp_list=None,
        sample_img=False,
        return_ori_image=False,
        sample_others=True,
        return_scene_name=False,
        return_lidar_box=False,
        return_lidar_box_id=False,
        control_test=False,
        vae_token_path=None,
        **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.training = training
        self.block_size = block_size
        self.sampling_gap = sampling_gap
        self.inference_flag = inference_flag
        self.views = views
        self.start_index = start_index
        self.files = []
        self.sample_img = sample_img
        self.sample_others = sample_others
        self.return_ori_image = return_ori_image
        self.return_lidar_box = return_lidar_box
        self.return_lidar_box_id = return_lidar_box_id
        self.sp_list = None
        self.return_scene_name = return_scene_name
        self.control_test = control_test
        for path in data_root:
            if os.path.isfile(path) and path.endswith(".pkl"):
                self.files.append(path)
                continue
            for filename in os.listdir(path):
                if filename.endswith(".pkl"):
                    self.files.append(os.path.join(path, filename))
        self.files = sorted(self.files)

        if isinstance(transforms, (list, tuple)):
            transforms = torchvision.transforms.Compose(transforms)
        self.transforms = transforms
        self.img_transform = img_transform

        # 读取类别的定义文件
        self._vocab = []
        with open(categories_file, "r") as fr:
            for line in fr:
                text = line.strip()
                if text:
                    self._vocab.append(text)

        # 设定VAE token的路径
        self.vae_token_path = vae_token_path
        if self.vae_token_path is not None:
            self.return_vae_token = True
        else:
            self.return_vae_token = False

        # 记录失败读取的场景
        self.error_scene = []

    def get_scene_order_path(self, base_order_path, pkl_path):
        """获取场景顺序路径.

        Args:
            base_order_path (str): 基础顺序路径。
            pkl_path (list of str): 包含场景路径的列表。

        Returns:
            list of str: 包含完整场景顺序路径的列表。
        """
        order_path = []
        for path in pkl_path:
            city_root = (
                path.split("/")[-3] + "/" + path.split("/")[-2] + "/"
            )  # 因为最后单挂了一个 '/', 所以从-3开始
            order_path.append(os.path.join(base_order_path, city_root))
        return order_path





    @property
    def mode(self):
        return "train" if self.training else "test"

    def __len__(self):
        return len(self.files)

    def get_frame_indices(self, seq_len):

        start_index = 4
        if self.inference_flag:
            start_index = (
                self.start_index
            ) 
        max_start_index = (
            seq_len
            - self.block_size * self.sampling_gap
            - self.sampling_gap
        )
        if max_start_index < self.sampling_gap:
            max_start_index = self.sampling_gap
            block_size = (
                seq_len - self.sampling_gap - 1
            ) // self.sampling_gap
            start_index = min(start_index, max_start_index)
            frame_indices = [
                start_index + i * self.sampling_gap
                for i in range(block_size)
            ]

        else:
            start_index = min(start_index, max_start_index)
            frame_indices = [
                start_index + i * self.sampling_gap
                for i in range(self.block_size)
            ]

        return frame_indices

    def get_sceneior_data(self, idx, curr_file=None):
        # load the scene data from the pickle file
        # return: Dict
        if curr_file is None:
            curr_file = self.files[idx]
        scene_name = curr_file.split("/")[-1]
        try:
            scene_name = (
                scene_name.split("_")[0]
                + "_"
                + scene_name.split("_")[1]
                + "_"
                + scene_name.split("_")[2]
                + "_"
                + scene_name.split("_")[3]
            )
        except:
            scene_name = (
                scene_name.split("_")[0]
                + "_"
                + scene_name.split("_")[1]
                + "_"
                + scene_name.split("_")[2]
            )

        # if it is control test, directly load the file and return
        self.curr_file = curr_file
        if self.control_test:
            with open(self.curr_file, "rb") as f:
                control_scene_token = pickle.load(f)
            return control_scene_token

        # load the pickle file
        self.curr_file = curr_file
        with open(curr_file, "rb") as fp:
            frame_data = pickle.load(fp)

        image_data = frame_data["tokens"][self.views[0]]["tokens"]
        image_data = np.stack(image_data, axis=0)
        seq_len = image_data.shape[0]
        self.seq_len = seq_len
        self.scene_num = seq_len

        # load the frame indices
        frame_indices = self.get_frame_indices(seq_len)
        if frame_indices is None:
            return None

        self.frame_indices = frame_indices

        data = self.get_format_sceneior_data(frame_data, frame_indices, image_data, idx, curr_file)

        return data

    def get_format_sceneior_data(
        self, frame_data, frame_indices, image_data, idx, curr_file=None
    ):
        # Get formatted scene data, including pose, map, bbox3d, bbox3d_cat, bbox3d_track_id, image, etc.
        # frame_data: dict, a dictionary containing all data
        # frame_indices: list, indices of sampled frames
        # image_data: np.array, image data
        # idx: int, index of the current scene, for record-keeping only
        # curr_file: str, filename of the current scene, for record-keeping
        data = {}
        if self.sample_others:
            bbox3d = []
            bbox3d_track_id = []
            bbox3d_cat = []
            lidar_bbox2d = []
            lidar_bbox2d_cat = []
            lidar_bbox2d_track_id = []
            # [x, y, z, w, l, h, heading, vx, vy, vz, ax, ay, az, wx, wy, wz]
            pose_data = frame_data["ego_pose_all"]
            meta_info = frame_data["meta_info"]
            pose_diff = []
            for i in range(len(frame_indices)):
                # get the ego data
                if i == 0:
                    index = frame_indices[i] - self.sampling_gap
                    assert index >= 0
                else:
                    index = frame_indices[i - 1]
                # the tranlation from the current frame to the next frame
                pose_tr = np.linalg.inv(meta_info[index]["T_lidar2global"]) @ (
                    meta_info[index + self.sampling_gap]["T_lidar2global"]
                    @ np.array([0, 0, 0, 1.0]).T
                )
                # the rotation from the current frame to the next frame
                heading_r = (
                    pose_data[index + self.sampling_gap, 6]
                    - pose_data[index, 6]
                )

                if heading_r >= np.pi:
                    heading_r -= 2 * np.pi
                if heading_r < -np.pi:
                    heading_r += 2 * np.pi

                pose_tr[3] = heading_r
                pose_diff.append(pose_tr)

                bbox3d.append(meta_info[frame_indices[i]]["bboxes_3d"])
                bbox3d_track_id.append(
                    meta_info[frame_indices[i]]["track_ids"]
                )
                bbox3d_cat.append(meta_info[frame_indices[i]]["categories"])

                lidar_bbox2d.append(
                    frame_data["lidar_bboxes"]["CAM_F0"]["bboxes_3d"][
                        frame_indices[i]
                    ]
                )
                lidar_bbox2d_cat.append(
                    frame_data["lidar_bboxes"]["CAM_F0"]["categories"][
                        frame_indices[i]
                    ]
                )
                lidar_bbox2d_track_id.append(
                    frame_data["lidar_bboxes"]["CAM_F0"]["track_ids"][
                        frame_indices[i]
                    ]
                )

            pose_diff = np.stack(pose_diff, axis=0)
            # we only use dx, dy, dheading
            pose_data = pose_diff[:, [0, 1, 3]]

            map_data = frame_data["raster_tokens"]
            map_data = map_data[frame_indices]
            map_data = rearrange(map_data, "t h w -> t (h w)")

            bbox3d = [
                np.array(bbox3d[i]).astype(np.float32)
                for i in range(len(bbox3d))
            ]
            bbox3d_cat = [bbox3d_cat[i] for i in range(len(bbox3d_cat))]
            bbox3d_track_id = [
                bbox3d_track_id[i] for i in range(len(bbox3d_track_id))
            ]

            fliter_index = self.categories_fliter(bbox3d_cat)  #
            for i in range(len(fliter_index)):
                valid_indices = []
                for j in fliter_index[i]:
                    x, y = bbox3d[i][j][
                        :2
                    ]  # Assuming bbox3d[i][idx] contains [x, y, z, ...]
                    if abs(x) > 64 or abs(y) > 64:
                        continue  # Skip agents outside the 64-meter range
                    valid_indices.append(j)
                fliter_index[i] = valid_indices

            for i in range(len(fliter_index)):
                try:
                    bbox3d[i] = bbox3d[i][fliter_index[i]]
                    bbox3d_cat[i] = np.array(bbox3d_cat[i])[
                        fliter_index[i]
                    ].tolist()
                    bbox3d_track_id[i] = np.array(bbox3d_track_id[i])[
                        fliter_index[i]
                    ]
                except:
                    print(
                        "Warning: some error in filting the categories and locations, return None"
                    )
                    return None

            data = {
                "pose": pose_data,
                "map": map_data,
                "pose_diff": pose_data,  # 
                "bbox3d": bbox3d,
                "bbox3d_cat": bbox3d_cat,
                "bbox3d_track_id": bbox3d_track_id,
            }

        # start to get the image data
        if self.sample_img:
            image_data_indices = image_data[frame_indices]
            image_tokens = image_data_indices.reshape(
                image_data_indices.shape[0], -1
            )
            data["image"] = image_tokens

        file_name = np.array(frame_data["tokens"][self.views[0]]["file_list"])[
            frame_indices
        ].tolist()
        scene_name = curr_file.split("/")[-1]
        scene_name = (
            scene_name.split("_")[0]
            + "_"
            + scene_name.split("_")[1]
            + "_"
            + scene_name.split("_")[2]
            + "_"
            + scene_name.split("_")[3]
        )

        # ori_image_path = self.ori_img_base_path + scene_name + "/CAM_F0"
        # image_file_path = []
        # for i in range(len(file_name)):
        #     file_name[i] = ori_image_path + "/" + file_name[i]
        #     image_file_path.append(file_name[i])
        # data["ori_image_path"] = image_file_path

        if self.return_ori_image:
            img_data_dict = dict(
                img_filename=image_file_path,
            )
            if self.img_transform is not None:
                for transform in self.img_transform:
                    img_data_dict = transform(img_data_dict)
                data["ori_image"] = img_data_dict["image"]

        if self.return_lidar_box:
            data["lidar_bbox2d"] = lidar_bbox2d
            data["lidar_bbox2d_cat"] = lidar_bbox2d_cat
        if self.return_lidar_box_id:
            data["lidar_bbox2d_track_id"] = lidar_bbox2d_track_id

        # some keys do not need to be transformed, such as path, lidar_bbox2d_cat
        no_transform_keys = ["lidar_bbox2d_cat", "lidar_bbox2d"]
        no_transform_data = {}
        keys_list = list(data.keys())
        for key in keys_list:
            if ("path" in key) or (key in no_transform_keys):
                no_transform_data[key] = data[key].copy()
                data.pop(key)

        if self.transforms is not None:
            data = self.transforms(data)
            for key in data.keys():
                if isinstance(data[key], list):
                    data[key] = torch.cat(data[key])

        for key in no_transform_data.keys():
            data[key] = no_transform_data[key]

        if self.return_scene_name:
            data["file_name"] = str(idx) + "_" + curr_file
        return data



    def get_current_file(self, idx):
        curr_file = self.files[idx]
        return curr_file


    def __getitem__(self, idx):
        # 根据idx读取对应的文件名
        curr_file = self.get_current_file(idx)
        data = self.get_sceneior_data(idx, curr_file=curr_file)
        # if data is None:
        #     curr_file = self.get_current_file(0)
        #     data = self.get_sceneior_data(0, curr_file)
        return data

    def categories_fliter(self, categories):
        # categories: list of list
        save_index = []
        for index in range(len(categories)):
            save_index_i = []
            for i in range(len(categories[index])):
                cat = categories[index][i]
                if cat in self._vocab:
                    save_index_i.append(i)
            save_index.append(save_index_i)
        return save_index

