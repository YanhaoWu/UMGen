import cv2
import mmcv  # 
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type="unchanged"):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results["filename"] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results["img"] = [img[..., i] for i in range(img.shape[-1])]
        results["img_shape"] = img.shape
        results["ori_shape"] = img.shape
        # Set initial values for default meta_keys
        results["pad_shape"] = img.shape
        results["scale_factor"] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


class ResizeCropFlipRotImage:
    def __init__(self, data_aug_conf=None, intrinsics=True, no_crop=False):
        self.data_aug_conf = data_aug_conf
        self.min_size = 2.0
        self.intrinsics = intrinsics
        self.no_crop = no_crop
        # print("in ResizeCropFlipRotImage, self.no_crop: ", self.no_crop)

    def __call__(self, results):

        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        # assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"
        H, W = results["img"][0].shape[:2]

        resize, resize_dims, crop, flip, rotate = self._sample_augmentation(
            H, W
        )

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )

            # new_imgs.append(np.array(img).astype(np.float32))
            new_imgs.append(img)
            if self.intrinsics:
                results["intrinsics"][i][:3, :3] = (
                    ida_mat @ results["intrinsics"][i][:3, :3]
                )
        results["img"] = new_imgs
        return results

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self, H, W):
        # H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims

        if self.no_crop:
            newW = fW
            newH = fH
            resize_dims = (newW, newH)

        crop_h = (
            int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
        )
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate


class ToTensor_Collect:
    def __init__(self, intrinsics=True, to_numpy=False):
        self.img_totensor = T.ToTensor()
        self.intrinsics = intrinsics
        self.to_numpy = to_numpy

    def __call__(self, results):
        data_dict = {}
        data_dict["image"] = [self.img_totensor(x) for x in results["img"]]
        data_dict["image"] = torch.stack(data_dict["image"])
        if self.intrinsics:
            data_dict["intrinsics"] = torch.Tensor(results["intrinsics"])
            data_dict["T_cam2global"] = torch.Tensor(results["T_cam2global"])
            data_dict["T_cam2ego"] = torch.Tensor(results["T_cam2ego"])
        if self.to_numpy:
            data_dict["image"] = data_dict["image"].numpy()

        return data_dict
