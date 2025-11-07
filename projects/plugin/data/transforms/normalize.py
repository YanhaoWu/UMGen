from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch


class Normalize_Standard:
    """Normalize data by mean and standard deviation.

    Args:
        data_key (str): key of data to be normalized.
        mean (list): mean of data.
        std (list): standard deviation of data.
        a_min (list or float, optional): minimum value of normalized data.
        a_max (list or float, optional): maximum value of normalized data.
    """

    def __init__(
        self,
        data_key: str,
        mean: List[float],
        std: List[float],
        a_min: Optional[Union[List[float], float]] = None,
        a_max: Optional[Union[List[float], float]] = None,
    ):
        self.mean = np.array(mean, dtype=np.float32)
        self.inv_std = 1.0 / np.array(std, dtype=np.float32)
        self.data_key = data_key

        if a_min is not None:
            self.a_min = np.array(a_min, dtype=np.float32)
        else:
            self.a_min = None

        if a_max is not None:
            self.a_max = np.array(a_max, dtype=np.float32)
        else:
            self.a_max = None

    def __call__(self, data: Dict):
        sample = data[self.data_key]
        if isinstance(sample, (list, tuple)):
            sample = [self.normalize(s) for s in sample]
        else:
            sample = self.normalize(sample)
        data[self.data_key] = sample

        return data

    def normalize(self, sample):
        torch_tensor = isinstance(sample, torch.Tensor)
        if torch_tensor:
            sample = sample.numpy()
        if sample.size == 0:
            sample = sample
        else:
            sample = (sample - self.mean) * self.inv_std
            if self.a_min is not None or self.a_max is not None:
                sample = np.clip(sample, self.a_min, self.a_max)
        if torch_tensor:
            sample = torch.from_numpy(sample)

        return sample

    def unnormalize_ego(self, sample):
        torch_tensor = isinstance(sample, torch.Tensor)
        if torch_tensor:
            sample = sample.numpy()
        if sample.size == 0:
            sample = sample
        else:
            sample = sample / self.inv_std + self.mean
        if torch_tensor:
            sample = torch.from_numpy(sample)

        return sample


class Normalize:
    """Normalize data by max and min data

    Args:
        data_key (str): key of data to be normalized.
        mean (list): mean of data.
        std (list): standard deviation of data.
        a_min (list or float, optional): minimum value of normalized data.
        a_max (list or float, optional): maximum value of normalized data.
    """

    def __init__(self, data_key: str, max_min: dict, min_max_standard_key=[]):
        self.data_key = data_key
        self.max_min = max_min
        if len(min_max_standard_key) != 0:
            print(
                "IN normalize, we found min_max_standard_key ",
                min_max_standard_key,
            )
        self.min_max_standard_key = min_max_standard_key

    def __call__(self, data: Dict):
        for data_key in self.data_key:
            sample = data[data_key]
            max_min = self.max_min[data_key]
            if isinstance(sample, (list, tuple)):
                sample = [self.normalize(s, max_min) for s in sample]
                if data_key in self.min_max_standard_key:
                    sample = [(s + 1) / 2 for s in sample]  # 从-1到1,变为0-1
            else:
                sample = self.normalize(sample, max_min)
                if data_key in self.min_max_standard_key:  # 从-1到1,变为0-1
                    sample = (data[data_key] + 1) / 2

            data[data_key] = sample

        return data

    def normalize(self, sample, max_min):
        min_v = max_min[0]
        max_v = max_min[1]
        torch_tensor = isinstance(sample, torch.Tensor)
        if torch_tensor:
            sample = sample.numpy()
        if sample.size == 0:
            sample = sample
        else:
            sample = (sample - min_v) / (
                max_v - min_v
            )  # 0 ~ 1 (某些值可能会在0~1之外，但是在tokenlize的时候会被约束)
            # if self.a_min is not None or self.a_max is not None:
            #     sample = np.clip(sample, self.a_min, self.a_max)
        if torch_tensor:
            sample = torch.from_numpy(sample)

        return sample

    def unnormalize(self, sample, max_min):
        min_v = max_min[0]
        max_v = max_min[1]
        torch_tensor = isinstance(sample, torch.Tensor)
        if torch_tensor:
            sample = sample.numpy()
        if sample.size == 0:
            sample = sample
        else:
            sample = sample * (max_v - min_v) + min_v
        if torch_tensor:
            sample = torch.from_numpy(sample)

        return sample

    def unnormalize_ego(self, bbox3d):
        if not isinstance(bbox3d, dict):
            data = {}
            target_key = self.data_key
            for index in range(
                len(target_key)
            ):  # bbox3d里面属性的放置顺序就和target_key一致
                key = target_key[index]
                for i in range(len(bbox3d)):
                    if key not in data:
                        data[key] = []
                    if len(bbox3d[i]) != 0:
                        data[key].append(bbox3d[i][index])
                    else:
                        data[key].append(np.array([]))

        for data_key in target_key:
            sample = data[data_key]
            if len(sample) != 0:
                max_min = self.max_min[data_key]
                if isinstance(sample, (list, tuple)):
                    sample = [self.unnormalize(s, max_min) for s in sample]
                else:
                    sample = self.unnormalize(sample, max_min)
                data[data_key] = sample
            else:
                data[data_key] = []

        unnormalize_bbox3d = []
        for index in range(len(bbox3d)):  # Get batch_size
            bbox3d_index = []
            for key in target_key:
                bbox3d_index.append(data[key][index])
            bbox3d_index = np.array(bbox3d_index)
            unnormalize_bbox3d.append(bbox3d_index)

        return unnormalize_bbox3d

    def unnormalize_bbox3d(self, bbox3d):
        if not isinstance(bbox3d, dict):
            data = {}
            target_key = self.data_key
            for index in range(
                len(target_key)
            ):  # bbox3d里面属性的放置顺序就和target_key一致
                key = target_key[index]
                for i in range(len(bbox3d)):
                    if key not in data:
                        data[key] = []
                    if len(bbox3d[i]) != 0:
                        data[key].append(bbox3d[i][:, index])
                    else:
                        data[key].append(np.array([]))

        for data_key in target_key:
            sample = data[data_key]
            if len(sample) != 0:
                max_min = self.max_min[data_key]
                if isinstance(sample, (list, tuple)):
                    if data_key in self.min_max_standard_key:
                        sample = [((s * 2) - 1) for s in sample]
                    sample = [self.unnormalize(s, max_min) for s in sample]
                else:
                    if data_key in self.min_max_standard_key:
                        sample = (sample * 2) - 1
                    sample = self.unnormalize(sample, max_min)
                data[data_key] = sample
            else:
                data[data_key] = []

        unnormalize_bbox3d = []
        for index in range(len(bbox3d)):  # Get batch_size
            bbox3d_index = []
            for key in target_key:
                bbox3d_index.append(data[key][index])
            bbox3d_index = np.stack(bbox3d_index, axis=1)
            unnormalize_bbox3d.append(bbox3d_index)

        return unnormalize_bbox3d


class ToTensor:
    """Convert data to torch.Tensor."""

    def __call__(self, data: Dict):
        return self._dict_to_tensor(data)

    def _dict_to_tensor(self, data: Dict):
        for key, value in data.items():
            if isinstance(value, dict):
                value = self._dict_to_tensor(value)
            elif isinstance(value, (list, tuple)):
                value = self._sequence_to_tensor(value)
            else:
                value = self._data_to_tensor(value)
            data[key] = value
        return data

    def _sequence_to_tensor(self, data: Sequence):
        if len(data) > 0 and isinstance(data[0], np.ndarray):
            if {d.shape for d in data} == 1:
                return self._data_to_tensor(np.stack(data))
            else:
                return [self._data_to_tensor(d) for d in data]
        elif len(data) > 0 and isinstance(data[0], str):
            return data
        elif len(data) > 0 and isinstance(data[0], torch.Tensor):
            return torch.stack(data)
        else:
            try:
                return torch.from_numpy(np.array(data))
            except Exception:
                return data

    def _data_to_tensor(self, data):
        if isinstance(data, np.ndarray):
            if data.dtype == np.float64:
                data = data.astype(np.float32)
            return torch.from_numpy(data.copy())
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, str) or isinstance(data, torch.Tensor):
            return data
        else:
            return data
