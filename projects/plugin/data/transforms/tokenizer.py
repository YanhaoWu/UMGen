from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class Tokenizer:
    """Base class for tokenizers.

    Args:
        start: The starting index for the first token.
        vocab_size: The size of the vocabulary.
        seq_len: The length of the sequence, if the length is fixed.
        special_tokens: A list of special tokens to add to the vocabulary.
            The special tokens will be added to the end of the vocabulary.
        pad_to_length: Pad the tokens to a fixed length.
    """

    def __init__(
        self,
        start: int,
        vocab_size: int,
        seq_len: int,
        special_tokens: Optional[List[str]] = None,
        pad_to_length=None,
    ):
        self._start = start
        self._end = start + vocab_size
        self._vocab_size = vocab_size
        self._seq_len = seq_len

        self.special_token2id = {}

        if special_tokens is not None:
            for token in special_tokens:
                self.special_token2id[token] = self._end
                self._end += 1

        self.pad_to_length = pad_to_length
        if pad_to_length is not None and "<pad>" not in self.special_token2id:
            self.special_token2id["<pad>"] = self._end
            self._end += 1

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def vocab_size(self):
        """The size of the vocabulary, not including special tokens."""
        return self._vocab_size

    @property
    def seq_len(self):
        """The length of the sequence."""
        return self._seq_len

    @property
    def start(self):
        """The starting index for the first token."""
        return self._start

    @property
    def end(self):
        """The ending index for the last token."""
        return self._end

    @property
    def bos_token(self):
        return self.special_token2id.get("<bos>", None)

    @property
    def eos_token(self):
        return self.special_token2id.get("<eos>", None)

    @property
    def pad_token(self):
        return self.special_token2id.get("<pad>", None)

    def __len__(self):
        """The size of the vocabulary, including special tokens."""
        return self._end - self._start

    def add_special_token(self, tokens, pad):
        """Add special tokens to the tokens."""

        if tokens.ndim == 1:
            return self.add_special_token_seq(tokens, pad)
        elif tokens.ndim == 2:
            return self.add_special_token_batch(tokens, pad)
        else:
            raise ValueError("tokens must be 1D or 2D array")

    def add_special_token_seq(self, tokens, pad):
        """Add special tokens to the tokens, for a single sequence."""

        assert tokens.ndim == 1, "tokens must be 1D array"
        if self.bos_token is not None:
            bos = np.array([self.bos_token])
            tokens = np.concatenate([bos, tokens])
        if self.eos_token is not None:
            eos = np.array([self.eos_token])
            tokens = np.concatenate([tokens, eos])
        if pad and self.pad_token is not None:
            if tokens.shape[0] >= self.pad_to_length:
                tokens = tokens[: self.pad_to_length]
            else:
                pad_length = self.pad_to_length - tokens.shape[0]
                pad_nd = np.array([self.pad_token] * pad_length)
                tokens = np.concatenate([tokens, pad_nd])
        return tokens

    def add_special_token_batch(self, tokens, pad):
        """Add special tokens to the tokens, for a batch of sequences."""

        assert tokens.ndim == 2, "tokens must be 2D array"
        if self.bos_token is not None:
            bos = np.full((tokens.shape[0], 1), self.bos_token)
            tokens = np.concatenate([bos, tokens], axis=-1)
        if self.eos_token is not None:
            eos = np.full((tokens.shape[0], 1), self.eos_token)
            tokens = np.concatenate([tokens, eos], axis=-1)
        if pad and self.pad_token is not None:
            if tokens.shape[1] >= self.pad_to_length:
                tokens = tokens[:, : self.pad_to_length]
            else:
                pad_length = self.pad_to_length - tokens.shape[1]
                pad_nd = np.full((tokens.shape[0], pad_length), self.pad_token)
                tokens = np.concatenate([tokens, pad_nd], axis=-1)
        return tokens

    def del_special_token(self, tokens):
        """Delete special tokens from the tokens."""

        if tokens.ndim == 1:
            return self.del_special_token_seq(tokens)
        elif tokens.ndim == 2:
            return self.del_special_token_batch(tokens)
        else:
            raise ValueError("tokens must be 1D or 2D array")

    def del_special_token_seq(self, tokens):
        """Delete special tokens from the tokens, for a single sequence."""

        assert tokens.ndim == 1, "tokens must be 1D array"
        pad_mask = tokens == self.pad_token
        bos_mask = tokens == self.bos_token
        eos_mask = tokens == self.eos_token
        tokens = tokens[~(pad_mask | bos_mask | eos_mask)]
        return tokens

    def del_special_token_batch(self, tokens):
        """Delete special tokens from the tokens, for a batch of sequences."""

        assert tokens.ndim == 2, "tokens must be 2D array"
        if tokens.size == 0:
            return tokens
        n = tokens.shape[0]
        pad_mask = tokens == self.pad_token
        bos_mask = tokens == self.bos_token
        eos_mask = tokens == self.eos_token
        try:
            tokens = tokens[~(pad_mask | bos_mask | eos_mask)]
            tokens = tokens.reshape(n, -1)
        except Exception as e:
            print(
                "There may be some problem with the tokens, such as bos/eos tokens occur in the middle of the sequence"  # noqa
            )
            print("tokens.shape: ", tokens.shape, "n: ", n)
            raise e
        return tokens


class IdentityTokenizer(Tokenizer):
    """Identity tokenizer.

    Used for data that is already tokenized, just need to add special tokens.

    Args:
        data_key: The key in the data dict to use for tokenization.
        start: The starting index for the first token.
        vocab_size: The size of the vocabulary.
        seq_len: The length of the sequence, if the length is fixed.
        special_tokens: A list of special tokens to add to the vocabulary.
        pad_to_length: Pad the tokens to a fixed length.
    """

    def __init__(
        self,
        data_key: str,
        start: int,
        vocab_size: int,
        seq_len: int,
        special_tokens: Optional[List[str]] = None,
        pad_to_length: Optional[int] = None,
    ):
        self.data_key = data_key
        super().__init__(
            start=start,
            vocab_size=vocab_size,
            seq_len=seq_len,
            special_tokens=special_tokens,
            pad_to_length=pad_to_length,
        )

    def __call__(self, data: Dict):
        if isinstance(self.data_key, list):
            for data_key in self.data_key:
                assert (
                    data_key in data
                ), f"Key {self.data_key} not found in data"
                raw_data = data[data_key]
                if isinstance(raw_data, list):
                    tokens = [self.encode(data) for data in raw_data]
                else:
                    tokens = self.encode(raw_data)
            data[data_key] = tokens
        else:
            data_key = self.data_key
            assert data_key in data, f"Key {self.data_key} not found in data"
            raw_data = data[data_key]
            if isinstance(raw_data, list):
                tokens = [self.encode(data) for data in raw_data]
            else:
                tokens = self.encode(raw_data)
            data[data_key] = tokens
        return data

    def encode(self, raw_tokens: Union[np.ndarray, torch.Tensor]):
        if isinstance(raw_tokens, torch.Tensor):
            raw_tokens = raw_tokens.numpy()
        assert isinstance(
            raw_tokens, np.ndarray
        ), "raw_data must be numpy.ndarray or torch.Tensor"

        tokens = raw_tokens + self.start
        return self.add_special_token(tokens, pad=True)

    def decode(self, tokens: Union[np.ndarray, torch.Tensor]):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy()
        assert isinstance(
            tokens, np.ndarray
        ), "raw_data must be numpy.ndarray or torch.Tensor"

        tokens = self.del_special_token(tokens)
        tokens -= self.start
        assert tokens.min() >= 0 and tokens.max() < self.vocab_size
        return tokens


class DigitalBinsTokenizer(Tokenizer):
    """Tokenize data by binning it into a fixed number of bins.

    Args:
        bins: A list of tuples, each tuple is the arguments for np.linspace,
             e.g. [(0, 1, 11), (1, 2, 21)] means the first bin is [0, 0.1, 0.2, ..., 1],
        data_key: The key in the data dict to use for tokenization
        seq_len: The length of the sequence, if the length is fixed
        start: The starting index for the first bin
        special_tokens: A list of special tokens to add to the vocabulary.
        pad_to_length: Pad the tokens to a fixed length.

    Example:
        >>> bins = [(0, 1, 11), (1, 2, 21)]
        >>> data_key = "data"
        >>> tokenizer = DigitalBinsTokenizer(bins, data_key)
    """

    def __init__(
        self,
        bins,
        data_key,
        seq_len,
        start=0,
        special_tokens=None,
        pad_to_length=None,
    ):
        bins = [np.linspace(*b) for b in bins]
        self.bins = np.concatenate(bins)
        self.data_key = data_key

        super().__init__(
            start=start,
            vocab_size=self.bins.shape[0],
            seq_len=seq_len,
            special_tokens=special_tokens,
            pad_to_length=pad_to_length,
        )

    def __call__(self, data: Dict):

        if isinstance(self.data_key, list):
            for data_key in self.data_key:
                # assert data_key in data, f"Key {data_key} not found in data"
                if data_key in data:
                    raw_data = data[data_key]
                    if isinstance(raw_data, list):
                        tokens = [self.encode(data) for data in raw_data]
                    else:
                        tokens = self.encode(raw_data)
                    data[data_key] = tokens
        else:
            data_key = self.data_key
            assert data_key in data, f"Key {self.data_key} not found in data"
            raw_data = data[data_key]
            if isinstance(raw_data, list):
                tokens = [self.encode(data) for data in raw_data]
            else:
                tokens = self.encode(raw_data)
            data[data_key] = tokens
        return data

    def encode(self, raw_data: Union[np.ndarray, torch.Tensor]):
        if isinstance(raw_data, torch.Tensor):
            raw_data = raw_data.numpy()
        assert isinstance(
            raw_data, np.ndarray
        ), "raw_data must be numpy.ndarray or torch.Tensor"

        tokens = np.digitize(raw_data, self.bins)  # 映射到对应的区间,超过最大的按最大的算
        tokens = (
            np.clip(tokens, 0, self.vocab_size - 1) + self.start
        )  # self.start应该是来自于上一个处理的模态
        tokens = self.add_special_token(
            tokens, pad=True
        )  # box3d tokenlizer里，这句话没有发生作用
        return tokens

    def decode(
        self, tokens: Union[np.ndarray, torch.Tensor], keep_order=False
    ):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy()
        assert isinstance(
            tokens, np.ndarray
        ), "raw_data must be numpy.ndarray or torch.Tensor"

        if not keep_order:
            tokens = self.del_special_token(tokens)
            tokens -= self.start
            if tokens.size == 0:
                return np.array([])
            # assert tokens.min() >= 0 and tokens.max() < self.vocab_size, print(tokens.min(), tokens.max())
        else:
            tokens -= self.start
            if tokens.size == 0:
                return np.array([])
        right = np.clip(tokens, 0, self.bins.shape[0] - 1)
        left = np.clip(tokens - 1, 0, self.bins.shape[0] - 1)
        values = (self.bins[left] + self.bins[right]) / 2
        return values


class TextTokenizer(Tokenizer):
    """Tokenize text data.

    Args:
        vocab_file: The vocab file
        data_key: The key in the data dict to use for tokenization
        seq_len: The length of the sequence, if the length is fixed
        start: The starting index for the first token
        special_tokens: A list of special tokens to add to the vocabulary
        pad_to_length: Whether to pad the tokens to a fixed length
    """

    def __init__(
        self,
        vocab_file,
        data_key,
        seq_len,
        start=0,
        special_tokens=None,
        pad_to_length=None,
    ):
        self.vocab_file = vocab_file
        self.data_key = data_key
        self._vocab = []
        with open(vocab_file, "r") as fr:
            for line in fr:
                text = line.strip()
                if text:
                    self._vocab.append(text)

        super().__init__(
            start=start,
            vocab_size=len(self._vocab),
            seq_len=seq_len,
            special_tokens=special_tokens,
            pad_to_length=pad_to_length,
        )

    def __call__(self, data: Dict):
        assert self.data_key in data, f"Key {self.data_key} not found in data"

        raw_data_frames = data[self.data_key]
        tokens = [self.encode(raw_data) for raw_data in raw_data_frames]
        data[self.data_key] = tokens
        return data

    def encode(self, raw_data: List[str]):
        tokens = np.array([self._vocab.index(data) for data in raw_data])
        tokens += self.start
        tokens = self.add_special_token(tokens, pad=True)
        return tokens

    def decode(
        self, tokens: Union[np.ndarray, torch.Tensor], keep_order=False
    ):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy()
        assert isinstance(
            tokens, np.ndarray
        ), "raw_data must be numpy.ndarray or torch.Tensor"

        if not keep_order:
            tokens = self.del_special_token(tokens)
            if tokens.size == 0:
                return np.array([])
            tokens -= self.start
            # tokens = np.clip(tokens, 0, self.vocab_size - 1)
            assert tokens.min() >= 0 and tokens.max() < self.vocab_size
            return [self._vocab[t] for t in tokens]
        else:
            if tokens.size == 0:
                return np.array([])
            tokens -= self.start
            output = []
            for t in tokens:
                if (t <= len(self._vocab) - 1) & (t >= 0):  # 说明在范围内，否则是空Tokens
                    output.append(self._vocab[t])
                else:
                    output.append("none")
            return output


#


class BBox3DTokenizer(Tokenizer):
    """Tokenize bounding box data.

    Args:
        bins: A list of tuples, each tuple is the arguments for np.linspace
        category_file: The category file
        start: The starting index for the first bin
        pad_to_length: Pad the tokens to a fixed length
        bbox_size: The data size of one bbox, e.g. 10 for center, lwh, yaw, speed
        special_tokens: A list of special tokens to add to the vocabulary
    """

    def __init__(
        self,
        bins,
        category_file,
        start,
        pad_to_length,
        bbox_size=10,
        special_tokens=None,
        target_key=["bbox3d"],
        with_track_ids=False,  # for degbugging
        shift_object_order_pro=False,  # 是否shift object的顺序来避免相邻物体之间的特征依赖
    ):
        self.with_track_ids = with_track_ids
        self.target_key = target_key
        self.bbox3d_tokenizer = DigitalBinsTokenizer(
            bins, target_key, bbox_size, start
        )
        self.cat_tokenizer = TextTokenizer(
            category_file, "bbox3d_cat", 1, start + len(self.bbox3d_tokenizer)
        )
        self.bbox_size = bbox_size
        vocab_size = (
            self.bbox3d_tokenizer.vocab_size + self.cat_tokenizer.vocab_size
        )

        self.shift_object_order_pro = shift_object_order_pro

        assert pad_to_length is not None, "pad_to_length must be specified"
        seq_len = (
            pad_to_length * (bbox_size + 1)
            + ("<bos>" in special_tokens)
            + ("<eos>" in special_tokens)
        )
        super().__init__(
            start=start,
            vocab_size=vocab_size,
            seq_len=seq_len,
            special_tokens=special_tokens,
            pad_to_length=pad_to_length,
        )
        #  pad_token -> null token
        self.frame_null_bbox_token = np.full(
            (pad_to_length, self.bbox_size + 1 + 1 * self.with_track_ids),
            self.pad_token,
        )
        self.frame_null_mask = np.full((pad_to_length, pad_to_length), False)
        if self.with_track_ids:
            # 最后一位置为-1
            self.frame_null_bbox_token[:, -1] = -1
        self.pad_to_length = pad_to_length

    def get_box_mask(self, bbox_tokens, track_id):
        # 获得指定track_id objects的mask
        frame_null_mask = np.full(
            (len(bbox_tokens), self.pad_to_length), False
        )
        for i in range(len(bbox_tokens)):
            bbox_id = bbox_tokens[i][:, -1]
            track_id_i = track_id[i].tolist()
            # 获得mask, 如果bbox_id在track_id中,则为True
            mask = np.isin(bbox_id, track_id_i)
            frame_null_mask[i] = mask

        return frame_null_mask

    def __call__(self, data: Dict):

        if "bbox3d_track_id" in data:  # 在nuplan dataset里面的使用
            assert (
                "bbox3d_track_id" in data
            ), "BBoxTokenizer requires bbox_track_id"
            # data["bbox"] = [
            #     b.reshape(b.shape[0], self.bbox_size) for b in data["bbox"]
            # ]
            data = self.bbox3d_tokenizer(data)
            data = self.cat_tokenizer(data)

            bbox_tokens = data["bbox3d"]
            category_tokens = data["bbox3d_cat"]
            track_ids = data["bbox3d_track_id"]
            # mask_index = data["mask_index"]
            # mask_trans_index = data["mask_trans_index"]

            bbox_tokens = [
                np.concatenate([bbox, cat[:, None]], axis=-1)
                for bbox, cat in zip(bbox_tokens, category_tokens)
            ]

            bbox_tokens, all_bbox_track_ids = self.bbox_slotting(
                bbox_tokens,
                track_ids.copy(),
                with_track_ids=self.with_track_ids,
                shift_object_order_pro=self.shift_object_order_pro,
            )  # 在这一步进行了填0操作
            if "lidar_bbox2d_track_id" in data.keys():
                frame_null_mask = self.get_box_mask(
                    bbox_tokens, data["lidar_bbox2d_track_id"]
                )
                data["view_bbox3d_mask"] = frame_null_mask
                del data["lidar_bbox2d_track_id"]
            bbox_tokens = np.array(bbox_tokens)
            if self.with_track_ids:
                bbox_tokens = bbox_tokens[:, :, :-1]  # 不要track_id

            bbox_tokens = bbox_tokens.reshape(bbox_tokens.shape[0], -1)
            bbox_tokens = self.add_special_token(
                bbox_tokens, pad=False
            )  # ? 两个Special Token都拼在最后

            # all_mask_indexs = np.array(all_mask_indexs)

            if "bbox3d_trans" in data:
                bbox_trans_tokens = data["bbox3d_trans"]
                bbox_trans_tokens = [
                    np.concatenate([bbox, cat[:, None]], axis=-1)
                    for bbox, cat in zip(bbox_trans_tokens, category_tokens)
                ]
                (
                    bbox_trans_tokens,
                    all_bbox_track_ids_trans,
                ) = self.bbox_slotting(
                    bbox_trans_tokens,
                    track_ids.copy(),
                    with_track_ids=self.with_track_ids,
                    shift_object_order_pro=self.shift_object_order_pro,
                    all_bbox_track_ids=all_bbox_track_ids,
                )  # 在这一步进行了填0操作
                assert sum(
                    all_bbox_track_ids_trans == all_bbox_track_ids
                ) == len(all_bbox_track_ids), print(
                    "all_bbox_track_ids_trans",
                    all_bbox_track_ids_trans,
                    "all_bbox_track_ids",
                    all_bbox_track_ids,
                )
                bbox_trans_tokens = np.array(bbox_trans_tokens)
                box_order = self.obtain_z_order_id(
                    bbox_trans_tokens.copy()
                )  # 只获得order,不使用
                bbox_trans_tokens = bbox_trans_tokens.reshape(
                    bbox_trans_tokens.shape[0], -1
                )
                bbox_trans_tokens = self.add_special_token(
                    bbox_trans_tokens, pad=False
                )  # ? 两个Special Token都拼在最后
                # all_mask_trans_indexs = np.array(all_mask_trans_indexs)

                data["bbox3d_order"] = box_order

                data["bbox3d_trans"] = bbox_trans_tokens

            data["bbox3d"] = bbox_tokens
            # data["mask_index"] = all_mask_indexs

            # data["mask_trans_index"] = all_mask_trans_indexs

            del data["bbox3d_track_id"]
            # del data["bbox_track_id"]
            del data["bbox3d_cat"]
            # del data["image"]
            # del data["bbox"]
            # del data["category"]
            # del data["pose"]

        else:  # 在PAR_AR里面的encode和decode

            data = self.bbox3d_tokenizer(data)
            data = self.cat_tokenizer(data)

            bbox_tokens = data[
                self.target_key[0]
            ]  # self.target_key： ['bbox3d'] or ['bbox3d_trans']
            category_tokens = data["bbox3d_cat"]

            bbox_tokens = [
                np.concatenate([bbox, cat[:, None]], axis=-1)
                for bbox, cat in zip(bbox_tokens, category_tokens)
            ]

            bbox_tokens = np.array(bbox_tokens)
            data[self.target_key[0]] = bbox_tokens.tolist()

            if "mask_trans_index" in data:
                mask_trans_index = data["mask_trans_index"]
                mask_trans_index = np.array(mask_trans_index)
                data["mask_trans_index"] = mask_trans_index

        return data

    def obtain_z_order_id(self, tokens):
        # 按照z字形对Object进行排序,平面结果中的x,y的顺序
        # 返回每个场景中的排序结果
        order_all = []
        for i in range(len(tokens)):
            bbox = tokens[i].reshape(60, -1)
            non_empty_index = bbox[:, 0] != 1029
            bbox_posi = bbox[1:, :3]
            # 先进行栅格化，临近位置可以视作一样
            bbox_posi = bbox_posi // 20
            # 按照y从高到低排序，x也是从高到低
            order_i = (
                np.lexsort((bbox_posi[:, 0], bbox_posi[:, 1])) + 1
            )  # 因为不放进去ego
            order_i = np.insert(order_i, 0, 0)
            order_all.append(order_i)
        return np.array(order_all)

    def del_unreason_tokens(self, tokens):
        max_tokens = self.cat_tokenizer.vocab_size
        min_tokens = 0
        cat_index_save = []
        for cat_t in tokens:
            cat_t = cat_t - self.cat_tokenizer.start
            index_filter = (cat_t < min_tokens) | (cat_t > max_tokens)
            # cat_t = cat_t[~index_filter]
            cat_index_save.append(~index_filter)
        return cat_index_save

    def encode(
        self, bbox: Union[np.ndarray, torch.Tensor], category: List[str]
    ):
        bboxes = self.bbox3d_tokenizer.decode(bbox)
        cat = self.cat_tokenizer.decode(category)
        return bboxes, cat

    def decode_single_objects(self, tokens):
        # 默认进来的维度是B, T, S
        bbox_tokens = tokens[0, :, :-1].copy()
        cat_tokens = tokens[0, 0, -1:].copy()
        bboxes = self.bbox3d_tokenizer.decode(bbox_tokens, keep_order=True)
        cats = self.cat_tokenizer.decode(
            tokens[0, 0, -1:].copy(), keep_order=True
        )
        return bboxes, cats

    def decode(
        self,
        tokens: Union[np.ndarray, torch.Tensor],
        keep_order=False,
        no_special=False,
    ):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.numpy()
        assert isinstance(
            tokens, np.ndarray
        ), "raw_data must be numpy.ndarray or torch.Tensor"
        if not keep_order:

            token_list = []

            for i in range(len(tokens)):

                token_list.append(
                    self.del_special_token(tokens[i], no_special=no_special)
                )

            tokens = token_list

            bbox_tokens = [t[:, :-1] for t in tokens]  # 最后一位是cat
            cat_tokens = [t[:, -1] for t in tokens]
            fliterd_index = self.del_unreason_tokens(cat_tokens)
            for i in range(len(bbox_tokens)):
                bbox_tokens[i] = bbox_tokens[i][fliterd_index[i]]
                cat_tokens[i] = cat_tokens[i][fliterd_index[i]]

        else:
            if not no_special:
                tokens_new = []
                for deal_tokens in tokens:
                    try:
                        bos_mask = deal_tokens == self.bos_token
                        eos_mask = deal_tokens == self.eos_token
                        new_tokens = deal_tokens[
                            ~(bos_mask | eos_mask)
                        ]  # 保留空tokens
                        new_tokens = new_tokens.reshape(-1, self.bbox_size + 1)
                    except:
                        print("bos or eos token in prediction wrong position")
                        new_tokens = deal_tokens.reshape(
                            -1, self.bbox_size + 1
                        )

                    tokens_new.append(new_tokens)
                tokens = tokens_new
                bbox_tokens = [t[:, :-1] for t in tokens]  # 最后一位是cat
                cat_tokens = [t[:, -1] for t in tokens]

            else:  # 考虑bbox3d带了batch维度的情况
                if len(tokens.shape) == 3:
                    B, T, S = tokens.shape
                    tokens = tokens.reshape(
                        B, T, self.pad_to_length, -1
                    )  # B, T, 60, 11
                    bbox_tokens = tokens[:, :, :, :-1]  # 最后一位是cat
                    cat_tokens = tokens[:, :, :, -1]

                    bbox_tokens = bbox_tokens.reshape(
                        B * T, self.pad_to_length, self.bbox_size
                    )
                    cat_tokens = cat_tokens.reshape(B * T, self.pad_to_length)
                else:
                    T, S = tokens.shape
                    tokens = tokens.reshape(T, self.pad_to_length, -1)

                    bbox_tokens = tokens[:, :, :-1]  # 最后一位是cat
                    cat_tokens = tokens[:, :, -1]

                    cat_tokens = cat_tokens.reshape(T, self.pad_to_length)
                    bbox_tokens = bbox_tokens.reshape(
                        T, self.pad_to_length, self.bbox_size
                    )

        bboxes = [
            self.bbox3d_tokenizer.decode(t, keep_order=keep_order)
            for t in bbox_tokens
        ]
        cats = [
            self.cat_tokenizer.decode(t, keep_order=keep_order)
            for t in cat_tokens
        ]
        return bboxes, cats

    def del_special_token(self, tokens, no_special=False):
        if not no_special:
            bos_mask = tokens == self.bos_token
            eos_mask = tokens == self.eos_token
            assert np.sum(bos_mask) == 1, "num of bos_mask should be 1"
            assert np.sum(eos_mask) == 1, "num of eos_mask should be 1"
            assert (
                bos_mask[0] == True
            ), "bos_mask should be at the first position"
            assert (
                eos_mask[-1] == True
            ), "eos_mask should be at the last position"
            tokens = tokens[~(bos_mask | eos_mask)]

        try:
            tokens = tokens.reshape(-1, self.bbox_size + 1)
        except Exception as e:
            print(
                "There may be some problem with the tokens, such as bos/eos tokens occur in the middle of the sequence"  # noqa
            )
            print(
                "tokens.shape: ",
                tokens.shape,
                "self.bbox_size: ",
                self.bbox_size,
            )
            raise e

        pad_mask = tokens == self.pad_token
        # Remove the bbox tokens that has any pad tokens
        tokens = tokens[~np.any(pad_mask, axis=1)]  # 只要有pad，这一个box就不要了
        return tokens

    def bbox_slotting(
        self,
        bbox_tokens,
        track_ids,
        mask_index=None,
        with_track_ids=False,
        shift_object_order_pro=0.5,
        all_bbox_track_ids=None,
    ):
        if with_track_ids:
            for i in range(len(bbox_tokens)):
                bbox_tokens[i] = np.concatenate(
                    [bbox_tokens[i], track_ids[i][:, None]], axis=-1
                )

        if all_bbox_track_ids is None:
            all_bbox_track_ids = np.concatenate(
                [
                    track_ids[:] if np.any(track_ids) else np.array([])
                    for track_ids in track_ids
                ]
            )
            if np.any(all_bbox_track_ids):
                _, idx = np.unique(all_bbox_track_ids, return_index=True)
                all_bbox_track_ids = all_bbox_track_ids[np.sort(idx)]
            # There can be self.max_bbox_num bboxes at most in the video clip
            if all_bbox_track_ids.size > self.pad_to_length:
                all_bbox_track_ids = all_bbox_track_ids[
                    : self.pad_to_length
                ]  # 只保留了前60个objects. 20 frame只保留60个objects?

            shift_object_order = (
                torch.rand((1)).item() < shift_object_order_pro
            )
            if shift_object_order:
                print("shiftting orders")
                np.random.shuffle(all_bbox_track_ids[1:])
                # print("new order is ", all_bbox_track_ids)

        track_id_2_slot_mapping = {
            track_id: i for i, track_id in enumerate(all_bbox_track_ids)
        }

        all_frame_bbox_tokens = []
        all_mask_indexs = []

        if mask_index is not None:
            for frame_track_ids, frame_bbox_token, frame_mask_index in zip(
                track_ids, bbox_tokens, mask_index
            ):
                # If there is no any bboxes in this frame,
                # then its bbox tokens are all null_bbox_token
                if not np.any(frame_track_ids):
                    all_frame_bbox_tokens.append(self.frame_null_bbox_token)
                    all_mask_indexs.append(self.frame_null_mask)
                else:
                    # If one frame has track_ids that are not in the
                    # all_bbox_track_ids, we need to remove them
                    frame_all_diff_ids = np.setdiff1d(
                        frame_track_ids, all_bbox_track_ids
                    )
                    if frame_all_diff_ids.size > 0:
                        keep_ids = [
                            i
                            for i, ids in enumerate(frame_track_ids)
                            if ids not in frame_all_diff_ids
                        ]
                        frame_track_ids = frame_track_ids[keep_ids]
                        frame_bbox_token = frame_bbox_token[keep_ids, ...]
                        # frame_mask_index是一个矩阵，保留剩下的box之间的关系``
                        frame_mask_index = frame_mask_index[keep_ids][
                            :, keep_ids
                        ]
                    # self.frame_null_bbox_token is a self.max_bbox_num x 11 array
                    # filled with null_bbox_token
                    if np.any(frame_track_ids):
                        temp_tokens = np.copy(self.frame_null_bbox_token)
                        temp_mask = np.copy(self.frame_null_mask)

                        slots = [
                            track_id_2_slot_mapping[track_id]
                            for track_id in frame_track_ids
                        ]
                        # Replace null bbox tokens with real bbox tokens
                        temp_tokens[slots] = frame_bbox_token

                        # Extend the real mask
                        # Calculate the number of columns to add ensuring it's not negative
                        # num_cols_to_add = max(0, temp_mask.shape[1] - frame_mask_index.shape[1])

                        # frame_mask_index = np.concatenate((frame_mask_index, np.zeros((frame_mask_index.shape[0], temp_mask.shape[1]-frame_mask_index.shape[1]), dtype=bool)), axis=1)
                        # Replace null mask with real mask
                        temp_mask[np.ix_(slots, slots)] = frame_mask_index

                        all_frame_bbox_tokens.append(temp_tokens)
                        all_mask_indexs.append(temp_mask)
                    else:
                        all_frame_bbox_tokens.append(
                            self.frame_null_bbox_token
                        )
                        all_mask_indexs.append(self.frame_null_mask)

            return all_frame_bbox_tokens, all_bbox_track_ids, all_mask_indexs

        else:
            for frame_track_ids, frame_bbox_token in zip(
                track_ids, bbox_tokens
            ):
                # If there is no any bboxes in this frame,
                # then its bbox tokens are all null_bbox_token
                if not np.any(frame_track_ids):
                    all_frame_bbox_tokens.append(self.frame_null_bbox_token)
                else:
                    # If one frame has track_ids that are not in the
                    # all_bbox_track_ids, we need to remove them
                    frame_all_diff_ids = np.setdiff1d(
                        frame_track_ids, all_bbox_track_ids
                    )
                    if frame_all_diff_ids.size > 0:
                        keep_ids = [
                            i
                            for i, ids in enumerate(frame_track_ids)
                            if ids not in frame_all_diff_ids
                        ]
                        frame_track_ids = frame_track_ids[keep_ids]
                        frame_bbox_token = frame_bbox_token[keep_ids, ...]

                    # self.frame_null_bbox_token is a self.max_bbox_num x 11 array
                    # filled with null_bbox_token
                    if np.any(frame_track_ids):
                        temp_tokens = np.copy(self.frame_null_bbox_token)
                        slots = [
                            track_id_2_slot_mapping[track_id]
                            for track_id in frame_track_ids
                        ]
                        # Replace null bbox tokens with real bbox tokens
                        temp_tokens[slots] = frame_bbox_token
                        all_frame_bbox_tokens.append(temp_tokens)
                    else:
                        all_frame_bbox_tokens.append(
                            self.frame_null_bbox_token
                        )

            return all_frame_bbox_tokens, all_bbox_track_ids


# class BBoxTokenizer(Tokenizer):
#     """Tokenize bounding box data.

#     Args:
#         bins: A list of tuples, each tuple is the arguments for np.linspace
#         category_file: The category file
#         start: The starting index for the first bin
#         pad_to_length: Pad the tokens to a fixed length
#         bbox_size: The data size of one bbox, e.g. 16 for 3d bbox with 8 2d corners
#         special_tokens: A list of special tokens to add to the vocabulary
#     """

#     def __init__(
#         self,
#         bins,
#         category_file,
#         start,
#         pad_to_length,
#         bbox_size=16,
#         special_tokens=None,
#     ):
#         self.bbox_tokenizer = DigitalBinsTokenizer(
#             bins, "bbox", bbox_size, start
#         )
#         self.cat_tokenizer = TextTokenizer(
#             category_file, "category", 1, start + len(self.bbox_tokenizer)
#         )
#         self.bbox_size = bbox_size
#         vocab_size = (
#             self.bbox_tokenizer.vocab_size + self.cat_tokenizer.vocab_size
#         )

#         assert pad_to_length is not None, "pad_to_length must be specified"
#         seq_len = (
#             pad_to_length * (bbox_size + 1)
#             + ("<bos>" in special_tokens)
#             + ("<eos>" in special_tokens)
#         )
#         super().__init__(
#             start=start,
#             vocab_size=vocab_size,
#             seq_len=seq_len,
#             special_tokens=special_tokens,
#             pad_to_length=pad_to_length,
#         )

#         self.frame_null_bbox_token = np.full(
#             (pad_to_length, self.bbox_size + 1), self.pad_token
#         )

#     def __call__(self, data: Dict):
#         assert (
#             "bbox_track_id" in data and "bbox" in data and "category" in data
#         ), "BBoxTokenizer requires bbox_track_id, bbox and category in data"
#         data["bbox"] = [
#             b.reshape(b.shape[0], self.bbox_size) for b in data["bbox"]
#         ]
#         data = self.bbox_tokenizer(data)
#         data = self.cat_tokenizer(data)

#         bbox_tokens = data["bbox"]
#         category_tokens = data["category"]
#         track_ids = data["bbox_track_id"]

#         bbox_tokens = [
#             np.concatenate([bbox, cat[:, None]], axis=-1)
#             for bbox, cat in zip(bbox_tokens, category_tokens)
#         ]

#         bbox_tokens = self.bbox_slotting(bbox_tokens, track_ids)    # 在这一步进行了填0操作
#         bbox_tokens = np.array(bbox_tokens)
#         bbox_tokens = bbox_tokens.reshape(bbox_tokens.shape[0], -1)
#         bbox_tokens = self.add_special_token(bbox_tokens, pad=False)    # self.box_token 9232?
#         data["bbox"] = bbox_tokens
#         del data["bbox_track_id"]
#         del data["category"]

#         return data

#     def encode(
#         self, bbox: Union[np.ndarray, torch.Tensor], category: List[str]
#     ):
#         bbox_tokens = self.bbox_tokenizer.encode(bbox)
#         cat_tokens = self.cat_tokenizer.encode(category)
#         return bbox_tokens, cat_tokens

#     def decode(self, tokens: Union[np.ndarray, torch.Tensor]):
#         if isinstance(tokens, torch.Tensor):
#             tokens = tokens.numpy()
#         assert isinstance(
#             tokens, np.ndarray
#         ), "raw_data must be numpy.ndarray or torch.Tensor"

#         tokens = [self.del_special_token(token) for token in tokens]
#         bbox_tokens = [t[:, :-1] for t in tokens]
#         cat_tokens = [t[:, -1] for t in tokens]

#         bboxes = [self.bbox_tokenizer.decode(t) for t in bbox_tokens]
#         cats = [self.cat_tokenizer.decode(t) for t in cat_tokens]
#         return bboxes, cats

#     def del_special_token(self, tokens):
#         bos_mask = tokens == self.bos_token
#         eos_mask = tokens == self.eos_token
#         tokens = tokens[~(bos_mask | eos_mask)]
#         try:
#             tokens = tokens.reshape(-1, self.bbox_size + 1)
#         except Exception as e:
#             print(
#                 "There may be some problem with the tokens, such as bos/eos tokens occur in the middle of the sequence"  # noqa
#             )
#             print(
#                 "tokens.shape: ",
#                 tokens.shape,
#                 "self.bbox_size: ",
#                 self.bbox_size,
#             )
#             raise e

#         pad_mask = tokens == self.pad_token
#         # Remove the bbox tokens that has any pad tokens
#         tokens = tokens[~np.any(pad_mask, axis=1)]
#         return tokens

#     def bbox_slotting(self, bbox_tokens, track_ids):
#         all_bbox_track_ids = np.concatenate(
#             [
#                 track_ids[:] if np.any(track_ids) else np.array([])
#                 for track_ids in track_ids
#             ]
#         )
#         if np.any(all_bbox_track_ids):
#             _, idx = np.unique(all_bbox_track_ids, return_index=True)
#             all_bbox_track_ids = all_bbox_track_ids[np.sort(idx)]
#         # There can be self.max_bbox_num bboxes at most in the video clip
#         if all_bbox_track_ids.size > self.pad_to_length:
#             all_bbox_track_ids = all_bbox_track_ids[: self.pad_to_length]
#         track_id_2_slot_mapping = {
#             track_id: i for i, track_id in enumerate(all_bbox_track_ids)
#         }

#         all_frame_bbox_tokens = []
#         for frame_track_ids, frame_bbox_token in zip(track_ids, bbox_tokens):
#             # If there is no any bboxes in this frame,
#             # then its bbox tokens are all null_bbox_token
#             if not np.any(frame_track_ids):
#                 all_frame_bbox_tokens.append(self.frame_null_bbox_token)
#             else:
#                 # If one frame has track_ids that are not in the
#                 # all_bbox_track_ids, we need to remove them
#                 frame_all_diff_ids = np.setdiff1d(
#                     frame_track_ids, all_bbox_track_ids
#                 )
#                 if frame_all_diff_ids.size > 0:
#                     keep_ids = [
#                         i
#                         for i, ids in enumerate(frame_track_ids)
#                         if ids not in frame_all_diff_ids
#                     ]
#                     frame_track_ids = frame_track_ids[keep_ids]
#                     frame_bbox_token = frame_bbox_token[keep_ids, ...]

#                 # self.frame_null_bbox_token is a self.max_bbox_num x 17 array
#                 # filled with null_bbox_token
#                 if np.any(frame_track_ids):
#                     temp_tokens = np.copy(self.frame_null_bbox_token)
#                     slots = [
#                         track_id_2_slot_mapping[track_id]
#                         for track_id in frame_track_ids
#                     ]
#                     # Replace null bbox tokens with real bbox tokens
#                     temp_tokens[slots] = frame_bbox_token
#                     all_frame_bbox_tokens.append(temp_tokens)
#                 else:
#                     all_frame_bbox_tokens.append(self.frame_null_bbox_token)

#         return all_frame_bbox_tokens
