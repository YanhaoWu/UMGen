from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from aidisdk.data.enums import TatSensorID
from hatbc.message.message import filter_topics
from torch import Tensor
from hat.registry import OBJECT_REGISTRY

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def image_normalize(img: Union[np.ndarray, Tensor], mean, std, layout):
    # TODO: to use qiwu.data.transforms.image.ImageToTensor
    """Normalize the image with mean and std.

    Args:
        mean (Union[float, Sequence[float]]): Shared mean or sequence of means
            for each channel.
        std (Union[float, Sequence[float]]): Shared std or sequence of stds for
            each channel.
        layout (str): Layout of img, `hwc` or `chw`.

    Returns:
        np.ndarray or torch.Tensor: Normalized image.

    """
    assert layout in ["hwc", "chw"]
    c_index = layout.index("c")

    return_ndarray = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img.astype(np.float32))
        return_ndarray = True
    elif isinstance(img, Tensor):
        img = img.float()
    else:
        raise TypeError

    if isinstance(mean, Sequence):
        assert len(mean) == img.shape[c_index]
    else:
        mean = [mean] * img.shape[c_index]

    if isinstance(std, Sequence):
        assert len(std) == img.shape[c_index]
    else:
        std = [std] * img.shape[c_index]

    dtype = img.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=img.device)
    std = torch.as_tensor(std, dtype=dtype, device=img.device)
    if (std == 0).any():
        raise ValueError(
            "std evaluated to zero after conversion to {}, "
            "leading to division by zero.".format(dtype)
        )
    if c_index == 0:
        mean = mean[:, None, None]
        std = std[:, None, None]
    else:
        mean = mean[None, None, :]
        std = std[None, None, :]
    img.sub_(mean).div_(std)

    if return_ndarray:
        img = img.numpy()

    return img


@OBJECT_REGISTRY.register
class ImageResizer(object):
    def __init__(
        self,
        h,
        w,
        keep_ratio=False,
        interpolation="bilinear",
        multi_stage=True,
        crop_down_front_view=True,
    ):
        """Resize image to a given size.

        Args:
            h (int): height of the output image.
            w (int): width of the output image.
            keep_ratio (bool): whether to keep the aspect ratio.
                Default: False.
            interpolation (str): interpolation method. Default: "bilinear".
            size (tuple): (height, width) of the output image.
            multi_stage (bool): whether to resize the image in multiple stages.
                Default: True.
            crop_down_front_view (bool): whether to crop the bottom of the
                front view image. Default: True.
        """
        assert interpolation in cv2_interp_codes, (
            f"Unknown interpolation: {interpolation}, "
            f"available: {cv2_interp_codes.keys()}"
        )
        self.h = h
        self.w = w
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.multi_stage = multi_stage
        self.crop_down_front_view = crop_down_front_view

    def __call__(self, data):
        # TODO: to use qiwu.data.transforms.image.ResizeTransform2DGenerator

        image_list = data["image"]
        cam_ids = data["cam_id"]
        new_image_list = []
        if "new_intrinsic" in data:
            new_intrinsics = data["new_intrinsic"]
        else:
            new_intrinsics = None
        for i, image in enumerate(image_list):
            trans_mat = np.eye(3)
            h, w, _ = image.shape
            origin_h, origin_w = h, w
            ratio = max(self.h / h, self.w / w)
            while ratio < 0.7 and self.multi_stage:
                new_h, new_w = int(h * 0.7), int(w * 0.7)
                image = cv2.resize(
                    image,
                    (new_w, new_h),
                    interpolation=cv2_interp_codes[self.interpolation],
                )
                h, w, _ = image.shape
                ratio = max(self.h / h, self.w / w)
            if self.keep_ratio:
                scale = self.w / w
                new_w = self.w
                new_h = int(h * scale + 0.5)
            else:
                new_h, new_w = self.h, self.w
            image = cv2.resize(
                image,
                (new_w, new_h),
                interpolation=cv2_interp_codes[self.interpolation],
            )
            scale_h, scale_w = new_h / origin_h, new_w / origin_w
            trans_mat[0, 0] = scale_w
            trans_mat[1, 1] = scale_h
            if new_h > self.h:
                if (
                    self.crop_down_front_view
                    and int(cam_ids[i]) == TatSensorID.IMAGE_FRONT.value
                ):
                    image = image[: self.h]
                else:
                    image = image[new_h - self.h :]
                    trans_mat[1, 2] = self.h - new_h
            if new_intrinsics is not None:
                new_intrinsics[i] = trans_mat @ new_intrinsics[i]
            new_image_list.append(image)
        data["image"] = new_image_list
        if new_intrinsics is not None:
            data["new_intrinsic"] = new_intrinsics
        return data


@OBJECT_REGISTRY.register
class StackImages(object):
    def __init__(self, dim):
        """Stack images in a dictionary.

        Args:
            dim (int): dimension to stack images.
        """
        self.dim = dim

    def __call__(self, data):
        image_list = data["image"]
        if len(image_list):
            if isinstance(image_list[0], torch.Tensor):
                images = torch.stack(image_list, dim=self.dim)
            elif isinstance(image_list[0], np.ndarray):
                images = np.stack(image_list, axis=self.dim)
            data["image"] = images
        return data


@OBJECT_REGISTRY.register
class ImageNormalize(object):
    def __init__(
        self,
        mean,
        std,
        bgr2rgb=False,
        to_tensor=True,
        layout="hwc",
        channel_first=False,
    ):
        """Normalize image.

        Args:
            mean (list): mean value for each channel.
            std (list): std value for each channel.
            bgr2rgb (bool): whether to convert BGR to RGB. Default: False.
            to_tensor (bool): whether to convert image to tensor.
                Default: True.
            layout (str): layout of the input image. Default: "hwc".
            channel_first (bool): whether to put channel first of output image.
                Default: False.
        """
        self.mean = mean
        self.std = std
        self.bgr2rgb = bgr2rgb
        self.to_tensor = to_tensor
        self.layout = layout
        self.channel_first = channel_first

    def __call__(self, data):
        # TODO: to use qiwu.data.transforms.image.ImageToTensor
        c_index = self.layout.index("c")
        other_index = [i for i in range(len(self.layout)) if i != c_index]
        image_list = data["image"]
        new_image_list = []
        for image in image_list:
            image: np.array = image.astype("float32")
            if self.bgr2rgb:
                image = np.flip(image, axis=c_index).copy()
            if self.to_tensor:
                image = torch.from_numpy(image)
            image = image_normalize(
                img=image, mean=self.mean, std=self.std, layout=self.layout
            )

            if self.channel_first and c_index != 0:
                if isinstance(image, torch.Tensor):
                    image = image.permute(c_index, *other_index)
                elif isinstance(image, np.ndarray):
                    image = np.transpose(image, (c_index, *other_index))
            new_image_list.append(image)
        data["image"] = new_image_list
        return data


class UnDistort(object):
    """Undistort the image."""

    def __call__(self, data):
        # TODO: to use qiwu.data.transforms.image.ImageUndistort

        cam_ids = data["cam_id"]
        image_list = data["image"]
        new_image_list = []
        new_cam_intrinsic_list = []
        for image, cam_id in zip(image_list, cam_ids):
            cam_msg = filter_topics(data["label"].messages, [str(cam_id)])[0]
            cam_param = cam_msg.camera_param
            cam_intrinsic = np.array(
                [
                    [cam_param.focal_u, 0, cam_param.center_u],
                    [0, cam_param.focal_v, cam_param.center_v],
                    [0, 0, 1],
                ]
            )
            img_height, img_width = image.shape[:2]
            dist_coeffs = np.array(cam_param.distort)
            new_intrinsic = cv2.getOptimalNewCameraMatrix(
                cam_intrinsic,
                dist_coeffs,
                (img_width, img_height),
                alpha=0,
                newImgSize=(img_width, img_height),
            )[0]
            new_intrinsic[0, 0] = (
                cam_intrinsic[0, 0] / cam_intrinsic[1, 1] * new_intrinsic[1, 1]
            )

            map_x, map_y = cv2.initUndistortRectifyMap(
                cam_intrinsic,
                dist_coeffs,
                None,
                new_intrinsic,
                (img_width, img_height),
                cv2.CV_32FC1,
            )
            undistort_img = cv2.remap(image, map_x, map_y, cv2.INTER_NEAREST)

            new_image_list.append(undistort_img)
            new_cam_intrinsic_list.append(new_intrinsic)

            # update camera parameters
            # cam_param.focal_u = new_intrinsic[0, 0]
            # cam_param.focal_v = new_intrinsic[1, 1]
            # cam_param.center_u = new_intrinsic[0, 2]
            # cam_param.center_v = new_intrinsic[1, 2]
            # cam_param.distort = [0.0 for _ in cam_param.distort]

        data["image"] = new_image_list
        data["new_intrinsic"] = new_cam_intrinsic_list
        return data
