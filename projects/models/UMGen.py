import inspect
import logging
import math
import os
import sys

# print(sys.path)

ego_whl = {
    "nuplan": {"w": 2.297, "l": 5.176, "h": 1.777},
    "waymo": {"w": 2.33, "l": 5.28, "h": 2.33},
}
import time
from argparse import Namespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from numpy import cos, inf, mean, pi
from numpy.core.numeric import outer
from torch import nn
from torch.nn import functional as F
from tqdm import trange
from projects.models.module import (
    GMLP,
    BlockOAR,
    BlockTAR,
    Decoder,
    LayerNorm,
    position_encoding_init,
)
from projects.plugin.data.transforms.normalize import Normalize
from projects.plugin.data.transforms.token_transform import (
    ego_transform,
    pose_transform,
)
from projects.plugin.data.transforms.tokenizer import DigitalBinsTokenizer
from projects.plugin.misc.misc import BoxOverlap
from projects.registry import MODELS


__all__ = [
    "UMGen",
]


logger = logging.getLogger(__name__)

@MODELS.register_module()
class UMGen(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ===== 1. Basic Configuration and Vocabulary Sizes =====
        self.config = config
        self.aux_vocab_size = config.aux_vocab_size
        self.pose_vocab_size = config.pose_vocab_size
        self.map_vocab_size = config.map_vocab_size
        self.bbox3d_vocab_size = config.bbox3d_vocab_size
        self.img_vocab_size = config.img_vocab_size
        self.bos_eos = config.bos_eos

        # ===== 2. Task and Sequence Settings =====
        self.max_frame_len = config.max_frame_len              # maximum length of frame sequence
        self.cond_frame = config.cond_frame                    # conditioning frame length
        self.task = config.task                                # modality order per task
        self.task_num = config.task_num                        # number of tasks
        self.task_name_id = config.task_name_id                # mapping: task_name -> id
        self.task_names = [k for k in self.task_name_id.keys()]
        self.task_id_name = {v: k for k, v in self.task_name_id.items()}
        self.task_prob = config.task_prob if hasattr(config, "task_prob") else None
        self.vocab_len = config.vocab_len                      # modality vocab sizes
        self.token_len = config.token_len                      # token sequence length per type
        self.seq_len = config.seq_len                          # max input token length

        # ===== 3. Model Architecture Hypertarameters =====
        self.n_tar_layer = config.n_tar_layer
        self.n_ego_tar_layer = config.n_ego_tar_layer
        self.n_ego_ca_layer = config.n_ego_ca_layer
        self.n_map_tar_layer = config.n_map_tar_layer
        self.n_box_tar_layer = config.n_box_tar_layer
        self.n_oar_layer = config.n_oar_layer
        if hasattr(config, "n_image_ar_layer"):
            self.n_image_ar_layer = config.n_image_ar_layer

        # ===== 4. Optional Config Parameters =====
        self.loss_ego_scale = config.loss_ego_scale if hasattr(config, "loss_ego_scale") else 1.0
        self.map_transform = config.map_transform if hasattr(config, "map_transform") else True

        # ===== 5. Embedding & Dropout Settings =====
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.ar_dropout = config.ar_dropout
        self.bias = config.bias
        self.cond_prob = config.cond_prob                       # probability of conditioning on TAR
        self.add_t_pos = config.add_t_pos if hasattr(config, "add_t_pos") else False

        # ===== 6. Sampling Parameters =====
        self.sfmx_temp = config.sfmx_temp
        self.top_k = config.top_k
        self.top_k_map = config.top_k_map if hasattr(config, "top_k_map") else self.top_k
        self.topk_image = 16
        self.p = config.p
        self.p_map = config.p_map if hasattr(config, "p_map") else self.p

        # ===== 7. Embedding Dimensions =====
        self.n_img_embd = config.n_img_embd
        self.n_map_embd = config.n_map_embd

        # ===== 8. Tokenizer and Normalizer =====
        self.ego_tokenlizer = config.ego_tokenlizer
        self.ego_norm = config.ego_norm
        self.box3d_tokenlizer = config.box3d_tokenlizer
        self.agent_norm = config.agent_norm

        # ===== 9. Token Sampling Strategy =====
        assert self.config.sample_method in ["topk", "topp"]
        if self.config.sample_method == "topk":
            self.token_sampler = self.topk
            self.sample_param = self.top_k
            self.sample_param_map = self.top_k_map
        else:
            self.token_sampler = self.sample_top_p
            self.sample_param = self.p
            self.sample_param_map = self.p_map

        # ===== 10. Device Settings =====
        self.device_set = config.device_set if hasattr(config, "device_set") else torch.device("cuda")

        # ===== 11. Split Configuration =====
        self.split_map_tar = config.split_map_tar
        self.split_map_ar = config.split_map_ar
        self.split_box_tar = config.split_box_tar

        # ===== 12. Positional Encoding =====
        self.fouier_pe = position_encoding_init(1024, self.n_embd).to(self.device_set)
        self.bbox3d_spatial_posi = position_encoding_init(1030, self.n_embd, start_index=1024).to(self.device_set)
        self.add_posi_embedd = config.add_posi_embedd
        self.grid_centers = self.get_grid_centers(grid_size=32, space_size=128)
        self.add_spatial_pos_embedd_on_map = config.add_spatial_pos_embedd_on_map

        if self.add_spatial_pos_embedd_on_map:
            normalize_centers = (self.grid_centers + 64) / 128
            normalize_centers = normalize_centers.numpy()
            bins = np.linspace(0.0, 1.0, 1024)
            grid_center_token = np.digitize(normalize_centers, bins)
            grid_center_token = torch.tensor(grid_center_token, dtype=torch.long).to(self.device_set)
            grid_center_token_x = grid_center_token[:, :, 0].reshape(1024, 1)
            grid_center_token_y = grid_center_token[:, :, 1].reshape(1024, 1)
            grid_center_x_posi_embedd = self.bbox3d_spatial_posi[grid_center_token_x]
            grid_center_y_posi_embedd = self.bbox3d_spatial_posi[grid_center_token_y]
            self.grid_center_posi_embedding = (grid_center_x_posi_embedd + grid_center_y_posi_embedd).squeeze()

        # ===== 13. Inference and Constraint Settings =====
        self.no_born = config.no_born if hasattr(config, "no_born") else False
        if hasattr(config, "no_born"):
            print("set no_born to", self.no_born)

        if hasattr(config, "rule_constrain"):
            self.rule_constrain = config.rule_constrain
            # print("rule_constrain is", self.rule_constrain)
            if self.rule_constrain:
                self.box_overlap = BoxOverlap()
        else:
            self.rule_constrain = False
            # print("we do not use rule_constrain in generation")

        # ===== 14. Training and Task Flags =====
        self.sample_img = config.sample_img
        self.train_only_ego = config.train_only_ego
        self.only_ar = config.only_ar
        self.ego_size = ego_whl["nuplan"]

        # ===== 15. UMGen Model Definition =====
        transformer_dict = nn.ModuleDict(
            dict(
                # Embeddings
                egoe=nn.Embedding(3, self.n_embd),
                axe=nn.Embedding(self.aux_vocab_size, self.n_embd),
                be=nn.Embedding(self.bbox3d_vocab_size, self.n_embd),
                tpe=nn.Embedding(self.max_frame_len, self.n_embd),
                spe=nn.Embedding(self.seq_len, self.n_embd),
                tske=nn.Embedding(self.task_num, self.n_embd),

                # Ego & Parallel Blocks
                ego_tar=nn.ModuleList([BlockTAR(self.config) for _ in range(self.n_ego_tar_layer)]),
                ln_ego_tar=LayerNorm(self.n_embd, bias=self.bias),
                ln_ego=LayerNorm(self.n_embd, bias=self.bias),


                TAR=nn.ModuleList([BlockTAR(self.config) for _ in range(self.n_tar_layer)]),
                OAR=nn.ModuleList([BlockOAR(self.config, causal=True) for _ in range(self.n_oar_layer)]),

                ln_tar=LayerNorm(self.n_embd, bias=self.bias),
                ln_oar=LayerNorm(self.n_embd, bias=self.bias),

                # Heads
                head_tar_aux=nn.Linear(self.n_embd, self.aux_vocab_size, bias=False),
                head_tar_pose=nn.Linear(self.n_embd, self.pose_vocab_size, bias=False),
                head_tar_map=nn.Linear(self.n_embd, self.map_vocab_size, bias=False),
                head_ar_aux=nn.Linear(self.n_embd, self.aux_vocab_size, bias=False),
                head_ar_pose=nn.Linear(self.n_embd, self.pose_vocab_size, bias=False),
                head_ar_map=nn.Linear(self.n_embd, self.map_vocab_size, bias=False),
                head_ar_bbox3d=nn.Linear(self.n_embd, self.bbox3d_vocab_size, bias=False),
                drop=nn.Dropout(self.dropout),

                # Ego Attention
                ego_cross_attn=nn.ModuleList([Decoder(self.config) for _ in range(self.n_ego_ca_layer)]),
                head_ego=nn.Linear(self.n_embd, self.pose_vocab_size, bias=False),
            )
        )

        # ===== 16. Additional Modules =====
        self.map_mlp_pre = GMLP(self.n_map_embd, 4 * self.n_embd, self.n_embd, dropout=0.0)

        # ===== 17. Additional Enhancements =====
        self.split_image_ar = config.split_image_ar if hasattr(config, "split_image_ar") else False
        # print("image ar split", self.split_image_ar) if self.split_image_ar else print("we do not split image ar")
        # ===== 18. Additional Head Layers =====
        if self.config.n_step == 1:
            transformer_dict["head_tar_bbox3d"] = nn.Linear(self.n_embd, self.bbox3d_vocab_size, bias=False)
        else:
            transformer_dict["head_tar_n_step_bbox3d"] = nn.Linear(
                self.n_embd, self.bbox3d_vocab_size * self.config.n_step, bias=False
            )

        if self.split_map_tar:
            transformer_dict["map_tar"] = nn.ModuleList([BlockTAR(self.config) for _ in range(self.n_map_tar_layer)])
            transformer_dict["ln_map_tar"] = LayerNorm(self.n_embd, bias=self.bias)

        if self.sample_img:
            transformer_dict["head_tar_img"] = nn.Linear(self.n_embd, self.img_vocab_size, bias=False)
            transformer_dict["head_ar_img"] = nn.Linear(self.n_embd, self.img_vocab_size, bias=False)
            self.img_mlp_pre = GMLP(self.n_img_embd, 4 * self.n_embd, self.n_embd, dropout=0.0)

            if self.split_box_tar:
                transformer_dict["box_tar"] = nn.ModuleList([BlockTAR(self.config) for _ in range(self.n_box_tar_layer)])
                transformer_dict["ln_box_tar"] = LayerNorm(self.n_embd, bias=self.bias)

            if self.split_image_ar:
                transformer_dict["image_ar"] = nn.ModuleList([BlockOAR(self.config, causal=True) for _ in range(self.n_image_ar_layer)])
                transformer_dict["ln_image_ar"] = LayerNorm(self.n_embd, bias=self.bias)

        self.transformer = nn.ModuleDict(transformer_dict)

        # ===== 20. Load Pretrained Codebooks =====
        map_codebook = torch.load(config.map_codebook, map_location="cpu").to(self.device_set)
        self.map_codebook = nn.Embedding.from_pretrained(map_codebook)
        del map_codebook

        img_codebook = torch.load(config.img_codebook, map_location="cpu").to(self.device_set)
        self.img_codebook = nn.Embedding.from_pretrained(img_codebook)
        del img_codebook
        print("Loaded img codebook")

        if self.device_set == torch.device("cpu"):
            self.fouier_pe = nn.Parameter(self.fouier_pe)
            self.bbox3d_spatial_posi = nn.Parameter(self.bbox3d_spatial_posi)
            if self.add_spatial_pos_embedd_on_map:
                self.grid_center_posi_embedding = nn.Parameter(self.grid_center_posi_embedding)

        # ===== 21. Logging and Metrics =====
        logger.info("number of parameters: %.2fB" % (self.get_num_params() / 1e9,))
        print("number of parameters: %.2fB" % (self.get_num_params() / 1e9))
        self.metric_dict = {}
        self.val_step = 0
        self.val_epoch_step = 0
        self.val_max_step = 0
        self.global_step = 0



        def _init_weights(self, module: nn.Module) -> None:
            """Initialize the weights of the module.

            Args:
                module (nn.Module): The module to initialize the weights for.
            """
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = True):
        """Return the number of parameters in the model.

        For non-embedding count, the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing
        these params are actually used as weights in the final layer,
        so we include them.

        Args:
            non_embedding (bool, optional): Whether to exclude the
                embedding parameters. Defaults to True.

        Returns:
            int: The number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.spe.weight.numel()
            n_params -= self.transformer.tpe.weight.numel()
            n_params -= self.transformer.tske.weight.numel()

        return n_params

    def build_affine_matrix(self, theta, x, y):
        affine_matrix = theta.new_zeros((2, 3))
        affine_matrix[0, 0] = torch.cos(-theta)
        affine_matrix[0, 1] = -torch.sin(-theta)
        affine_matrix[0, 2] = -y
        affine_matrix[1, 0] = torch.sin(-theta)
        affine_matrix[1, 1] = torch.cos(-theta)
        affine_matrix[1, 2] = -x

        return affine_matrix

    def affine_transform(self, x, pose_diff, res=4.0, interp_mode="bilinear"):
        # transform the map feature with the pose difference
        x = x.clone().detach()
        B, T, S, C = x.size()
        H, W = int(np.sqrt(S)), int(np.sqrt(S))
        x = rearrange(x, "b t s c -> (b t) c s", b=B, t=T, s=S, c=C)
        x = rearrange(x, "n c (h w) -> n c h w", h=H, w=W)
        pose_diff = rearrange(pose_diff, "b t c -> (b t) c")
        theta = pose_diff[:, 2]

        dx = 2 * (pose_diff[:, 0] / res) / W
        dy = 2 * (pose_diff[:, 1] / res) / H

        affine_matrix = [
            self.build_affine_matrix(theta[i], dx[i], dy[i])
            for i in range(B * T)
        ]
        affine_matrix = torch.stack(affine_matrix, dim=0)  # transform matrix

        grid = F.affine_grid(
            affine_matrix, (B * T, C, H, W), align_corners=False
        )
        t_x = F.grid_sample(
            x,
            grid,
            mode=interp_mode,
            padding_mode="zeros",
            align_corners=False,
        )
        t_x = rearrange(t_x, "n c h w -> n (h w) c")
        t_x = rearrange(t_x, "(b t) s c -> b t s c", b=B, t=T, s=S, c=C)
        t_x = t_x.to(x.dtype)

        return t_x


    def get_grid_centers(self, grid_size=32, space_size=128):
        """Calculate the center coordinates of each grid cell.

        :param grid_size: Grid size (default 32)
        :param space_size: Total space size (default 128m)
        :return: Tensor of grid center coordinates (grid_centers_x, grid_centers_y)
        """
        # Calculate the size of each grid cell
        cell_size = space_size / grid_size

        # Generate grid indices
        grid_x = torch.arange(0, grid_size)
        grid_y = torch.arange(0, grid_size)

        # Generate grid coordinates using torch.meshgrid
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y)

        # Compute the center coordinates for each grid cell
        center_x = (grid_x + 0.5) * cell_size - space_size / 2
        center_y = (grid_y + 0.5) * cell_size - space_size / 2

        center_x = -center_x
        center_y = -center_y

        coordinates = torch.stack([center_x, center_y], dim=-1)

        return coordinates

    def get_grid_coordinates(
        self, x, y, grid_size=32, space_size=128, exchange_xy=False
    ):
        """Map real-world coordinates (x, y) to grid coordinates.

        :param x: Real-world x coordinate
        :param y: Real-world y coordinate
        :param grid_size: Grid size (default 32)
        :param space_size: Total space size (default 128m)
        # :param exchange_xy: Whether to swap x and y coordinates (default False),
        #   because in the map before and after decoding, x and y are swapped.
        :return: Grid coordinates (grid_x, grid_y)
        """
        if exchange_xy:
            x, y = y, x

        # Calculate the size of each grid cell
        cell_size = space_size / grid_size

        # Convert real-world coordinates to grid coordinates
        # Since the origin is at the center of the grid, shift coordinates into grid range
        grid_x = ((-x + space_size / 2) / cell_size).long()
        grid_y = ((-y + space_size / 2) / cell_size).long()

        return grid_x, grid_y

    def add_spatial_pos_emb(self, x):
        # Decode x and then add positional embeddings
        B, T, S = x.size()
        x = x.reshape(B, T, self.config.pad_to_length, -1)
        x_posi = x[:, :, :, 0]
        y_posi = x[:, :, :, 1]

        x_posi_embedd = self.bbox3d_spatial_posi[x_posi]
        y_posi_embedd = self.bbox3d_spatial_posi[y_posi]

        posi_embedding = (
            x_posi_embedd + y_posi_embedd
        )  # B, T, 60, C — positional embedding for each object

        # Add one dimension before the last axis
        posi_embedding = posi_embedding.unsqueeze(-2)
        # Expand that dimension to match the number of attributes (11)
        num_attritube = int(S / self.config.pad_to_length)
        posi_embedding = posi_embedding.expand(
            -1, -1, -1, num_attritube, -1
        )  # 11 is the number of attributes
        # Merge the 3rd and 4th dimensions into one
        posi_embedding = posi_embedding.reshape(B, T, S, -1)

        return posi_embedding


    def get_mod_emb_pre(
        self, inputs, mod, add_posi_embedd=False, add_map_posi_embedd=False
    ):
        feats = inputs[mod]
        if mod == "bbox3d":
            feats = self.transformer.be(feats)
            if add_posi_embedd:
                posi_embedding = self.add_spatial_pos_emb(inputs[mod])
                feats = feats + posi_embedding

        elif mod == "map":
            feats = self.map_codebook(feats)
            feats = self.map_mlp_pre(feats)
            if add_map_posi_embedd:
                B, T, S, C = feats.size()
                expanded_posi_embedding = (
                    self.grid_center_posi_embedding.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(B, T, -1, -1)
                )  # 在每个step Map的posi embedding一致
                feats = feats + expanded_posi_embedding

        elif mod == "pose":
            # feats = self.transformer.pe(feats)
            feats = self.fouier_pe[feats]

        elif mod == "image":
            feats = self.img_codebook(feats)
            feats = self.img_mlp_pre(feats)

        return feats

    def add_bos_eos(self, inputs, mod):
        feats = inputs[mod]
        B, T, S, C = feats.size()
        bos_eos_token = feats.new_tensor(self.bos_eos[mod], dtype=int)
        bos_eos_emb = self.transformer.axe(bos_eos_token)
        bos_eos_emb = bos_eos_emb.expand(B, T, -1, -1)
        feats = torch.cat(
            [bos_eos_emb[:, :, [0], :], feats, bos_eos_emb[:, :, [-1], :]],
            dim=2,
        )

        return feats

    def add_pos_emb(
        self, token_emb: torch.Tensor, add_t_pos: bool = True
    ) -> torch.Tensor:
        """Get the embeddings of tokens.

        Args:
            tokens (torch.Tensor): The input tokens.
            add_t_pos (bool, optional): Whether to add temporal position
                embeddings. Defaults to True.

        Returns:
            torch.Tensor: The embeddings of tokens.
        """
        if token_emb.numel() == 0:
            return token_emb.new_tensor([])
        B, T, S, C = token_emb.size()
        # get the token embedding [b, t, s, n_embd]
        # token_emb = self.transformer.te(tokens)
        device = token_emb.device
        # get the sequence position embedding [1, 1, s, n_embd]
        seq_pos = torch.arange(0, S, dtype=torch.long, device=device)
        seq_pos_emb = self.transformer.spe(seq_pos)[None, None, :, :]
        if add_t_pos:
            # get the frame temporal position embedding [1, t, 1, n_embd]
            t_pos = torch.arange(0, T, dtype=torch.long, device=device)
            temporal_pos_emb = self.transformer.tpe(t_pos)[None, :, None, :]
            # add the temporal and seqv uence PEs to the token embedding
            token_emb = token_emb + seq_pos_emb + temporal_pos_emb
        else:
            # add the sequence PEs to the token embedding
            token_emb = token_emb + seq_pos_emb

        return token_emb

    def add_task_emb(self, task_name, embs):
        # get the task embedding [1, 1, 1, n_embd]
        task_token = self.task_name_id[task_name]
        task_token = embs.new_tensor([task_token], dtype=torch.long)
        task_emb = self.transformer.tske(task_token)[None, None, :, :]

        # expand the task embedding to [B, T, 1, n_embd]
        B, T, S, C = embs.size()
        task_emb = task_emb.expand(B, T, -1, -1)
        embs = torch.cat([task_emb, embs], dim=-2)

        return embs

    def get_mod_emb_post(self, embs, mod_order):
        d_embs = {}
        c_embs = {}
        for mod in mod_order:
            d_embs[mod] = embs[mod][:, :, [0, -1], :]
            c_embs[mod] = embs[mod][:, :, 1:-1, :]

        return d_embs, c_embs

    def get_targets(self, targets, mod_order):
        d_targets = {}
        c_targets = {}
        for mod in mod_order:
            c_target = targets[mod]
            B, T, S = c_target.size()
            c_targets[mod] = c_target

            bos_eos = self.bos_eos[mod]
            bos_eos = c_target.new_tensor(bos_eos)
            bos_eos = bos_eos.expand(B, T, -1)
            d_targets[mod] = bos_eos

        if "bbox3d" in mod_order:
            T_pose = targets["pose"].size(1)
            d_targets["bbox3d"] = d_targets["bbox3d"][:, :T_pose, :]

        return d_targets, c_targets

    def d_logit(self, d_embs, mod_order, stage="tar"):
        d_logits_all = {}
        for mod in mod_order:
            d_emb = d_embs[mod]
            if stage == "tar":
                logits = self.transformer.head_tar_aux(d_emb)
            else:
                logits = self.transformer.head_ar_aux(d_emb)
            d_logits_all[mod] = logits

        return d_logits_all

    def d_loss(self, d_logits, d_targets, mod_order):
        d_loss_all = {}
        for mod in mod_order:
            d_logit = d_logits[mod]
            d_target = d_targets[mod]
            # calculate the loss for the discrete part
            d_loss_all[mod] = F.cross_entropy(
                d_logit.view(-1, d_logit.size(-1)),
                d_target.contiguous().view(-1),
                ignore_index=-1,
            )

        return d_loss_all

    def c_logit(self, c_embs, mod_order, stage="tar"):
        c_logits_all = {}
        for mod in mod_order:
            c_emb = c_embs[mod]
            if stage == "tar":
                if mod == "pose":
                    logits = self.transformer.head_tar_pose(c_emb)
                elif mod == "map":
                    logits = self.transformer.head_tar_map(c_emb)
                elif mod == "bbox3d":
                    if self.config.n_step != 1:
                        logits = self.transformer.head_tar_n_step_bbox3d(c_emb)
                        logits = torch.split(
                            logits, self.bbox3d_vocab_size, dim=-1
                        )
                    else:
                        logits = self.transformer.head_tar_bbox3d(c_emb)
                elif mod == "image":
                    logits = self.transformer.head_tar_img(c_emb)

                else:
                    raise ValueError(f"modality {mod} not supported")
            else:
                if mod == "pose":
                    logits = self.transformer.head_ar_pose(c_emb)
                elif mod == "map":
                    logits = self.transformer.head_ar_map(c_emb)
                elif mod == "bbox3d":
                    logits = self.transformer.head_ar_bbox3d(c_emb)
                elif mod == "image":
                    logits = self.transformer.head_ar_img(c_emb)
                else:
                    raise ValueError(f"modality {mod} not supported")
            c_logits_all[mod] = logits

        return c_logits_all


    def get_transformed_priors(self, inputs, targets):
        priors = {}
        temp_inputs = {"map": inputs["map"].clone().detach()}
        priors["pose_diff"] = targets["pose_diff"]
        if "map" in inputs:
            map_feature = self.get_mod_emb_pre(temp_inputs, "map")
            priors["map_warped"] = self.affine_transform(
                map_feature,
                priors["pose_diff"],
            )
        return priors

    def forward_ego_net(
        self,
        inputs: Dict[str, torch.Tensor],
        task_name: str = "pose_map_image",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the ego change.

        Args:
            inputs (Dict[str, torch.Tensor]): The input tensors for
                different modalities.
            task_name (str, optional): The name of the task.
                Default is "pose_map_image".

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The logits from the PAR network and the PAR network embeddings.
        """
        # form the input sequence and targets according to the modality order
        assert task_name in self.task
        mod_order = self.task[task_name]
        # mod_order = ["pose", "map", "bbox3d"]
        inputs_seq = {}
        for mod in mod_order:
            assert mod in inputs
            inputs_seq[mod] = self.get_mod_emb_pre(
                inputs, mod, add_posi_embedd=self.add_posi_embedd
            )
            inputs_seq[mod] = self.add_bos_eos(inputs_seq, mod)
        tar_emb = torch.cat([inputs_seq[m] for m in mod_order], dim=-2).cuda()
        tar_emb = self.add_pos_emb(tar_emb)
        # forward the token embedding into the transformer
        tar_emb = self.transformer.drop(tar_emb)
        kvcache_t = [None] * self.n_ego_tar_layer
        for block, kvcache_t_block in zip(self.transformer.ego_tar, kvcache_t):
            tar_emb, _ = block(tar_emb, kvcache=kvcache_t_block)
        tar_emb = self.transformer.ln_ego_tar(tar_emb)

        B, T, S, C = tar_emb.size()
        ego_tokens = torch.tensor(
            [0, 1, 2], dtype=torch.long, device=tar_emb.device
        )
        ego_emb = self.transformer.egoe(ego_tokens)
        ego_emb = ego_emb.expand(B, T, -1, -1)
        ego_emb = self.add_pos_emb(ego_emb)

        # do the ego query cross attention with the historic scene info
        ego_emb = self.transformer.drop(ego_emb)
        kvcache_t = [None] * self.n_ego_ca_layer
        for block, kvcache_t_block in zip(
            self.transformer.ego_cross_attn, kvcache_t
        ):
            ego_emb, _ = block(ego_emb, tar_emb, kvcache=kvcache_t_block)
        ego_emb = self.transformer.ln_ego(ego_emb)
        return ego_emb



    def forward_tar_net(
        self,
        inputs: Dict[str, torch.Tensor],
        task_name: str = "image",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the PAR network.

        Args:
            inputs (Dict[str, torch.Tensor]): The input tensors for
                different modalities.
            task_name (str, optional): The name of the task.
                Default is "image".

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The logits from the PAR network and the PAR network embeddings.
        """
        # form the input sequence and targets according to the modality order
        assert task_name in self.task
        mod_order = self.task[task_name]
        inputs_seq = {}
        priors = {}
        for mod in mod_order:
            assert mod in inputs
            if mod == "pose":
                inputs_seq[mod] = self.get_mod_emb_pre(inputs, mod)
                pose_diff_cpu = self.decode_pose(
                    inputs
                )  #
                pose_diff = pose_diff_cpu.cuda()
            if mod == "map":
                inputs_seq[mod] = self.get_mod_emb_pre(
                    inputs,
                    mod,
                    add_map_posi_embedd=self.add_spatial_pos_embedd_on_map,
                )

                # correspond to the "action-aware map alignments" module
                if self.map_transform:
                    inputs_seq["map_warped"] = self.affine_transform(
                        inputs_seq[mod], pose_diff
                    )
                    inputs_seq[mod] = (
                        inputs_seq["map_warped"] + inputs_seq[mod]
                    )
                    priors["map_warped"] = inputs_seq["map_warped"].clone()

            if mod == "bbox3d":
                if self.config.box_transform:
                    inputs_seq["bbox3d_warped"] = self.transform_bbox3d(
                        inputs["bbox3d"], pose_diff_cpu
                    )
                    inputs_seq[mod] = inputs_seq[
                        "bbox3d_warped"
                    ]  #
                    priors["bbox3d_warped"] = inputs_seq[
                        "bbox3d_warped"
                    ].clone()
                else:
                    inputs_seq[mod] = inputs["bbox3d"]

                inputs_seq[mod] = self.get_mod_emb_pre(
                    inputs_seq, mod, add_posi_embedd=self.add_posi_embedd
                )

            if mod == "image":
                inputs_seq[mod] = self.get_mod_emb_pre(inputs, mod)

            inputs_seq[mod] = self.add_bos_eos(inputs_seq, mod)

        # concatenate the token embeddings from different modalities to form the input sequence
        tar_emb = torch.cat([inputs_seq[m] for m in mod_order], dim=-2).cuda()
        tar_emb = self.add_pos_emb(tar_emb)

        # forward the token embedding into the TAR network
        tar_emb = self.transformer.drop(tar_emb)
        kvcache_t = [None] * self.n_tar_layer
        for block, kvcache_t_block in zip(self.transformer.TAR, kvcache_t):
            tar_emb, _ = block(tar_emb, kvcache=kvcache_t_block)
        tar_emb = self.transformer.ln_tar(tar_emb)

        # save the token embeddings for different modalities
        mod_tar_embs = torch.split(tar_emb, [self.token_len[mod] for mod in mod_order], dim=2)
        tar_embs = {}
        for mod, mod_tar_emb in zip(mod_order, mod_tar_embs):
            tar_embs[mod] = mod_tar_emb

        return tar_embs, priors, pose_diff


    def forward_tar_for_map(
        self,
        inputs: Dict[str, torch.Tensor],
        task_name: str = "pose_map",
    ) -> Tuple[torch.Tensor]:
        mod_order = self.task[task_name]
        inputs_seq = {}
        priors = {}
        for mod in mod_order:
            assert mod in inputs
            if mod == "pose":
                inputs_seq[mod] = self.get_mod_emb_pre(inputs, mod)
                pose_diff_cpu = self.decode_pose(inputs)  #
                pose_diff = pose_diff_cpu.cuda()
            if mod == "map":
                inputs_seq[mod] = self.get_mod_emb_pre(inputs, mod)

                # correspond to the "action-aware map alignments" module
                inputs_seq["map_warped"] = self.affine_transform(inputs_seq[mod], pose_diff)  

                inputs_seq[mod] = inputs_seq["map_warped"] + inputs_seq[mod]
                priors["map_warped"] = inputs_seq["map_warped"].clone()
            inputs_seq[mod] = self.add_bos_eos(inputs_seq, mod)

        tar_emb = torch.cat([inputs_seq[m] for m in mod_order], dim=-2).cuda()
        tar_emb = self.add_pos_emb(tar_emb)
        tar_emb = self.transformer.drop(tar_emb)
        kvcache_t = [None] * self.n_map_tar_layer
        for block, kvcache_t_block in zip(self.transformer.map_tar, kvcache_t):
            tar_emb, _ = block(tar_emb, kvcache=kvcache_t_block)
        tar_emb = self.transformer.ln_map_tar(tar_emb)

        mod_tar_embs = torch.split(tar_emb, [self.token_len[mod] for mod in mod_order], dim=2)
        tar_embs = {}
        for mod, mod_tar_emb in zip(mod_order, mod_tar_embs):
            tar_embs[mod] = mod_tar_emb
        return tar_embs, priors

    def forward_tar_for_box(
        self,
        inputs: Dict[str, torch.Tensor],
        task_name: str = "pose_map_bbox3d",
    ) -> Tuple[torch.Tensor]:
        #    """Predict the box """
        mod_order = self.task[task_name]
        inputs_seq = {}
        priors = {}
        for mod in mod_order:
            assert mod in inputs
            if mod == "pose":
                inputs_seq[mod] = self.get_mod_emb_pre(inputs, mod)
                pose_diff_cpu = self.decode_pose(inputs)
                pose_diff = pose_diff_cpu.cuda()
            if mod == "map":
                inputs_seq[mod] = self.get_mod_emb_pre(inputs, mod)
                inputs_seq["map_warped"] = self.affine_transform(
                    inputs_seq[mod], pose_diff
                )
                inputs_seq[mod] = inputs_seq["map_warped"] + inputs_seq[mod]
                priors["map_warped"] = inputs_seq["map_warped"].clone()
            if mod == "bbox3d":
                if self.config.box_transform:
                    inputs_seq["bbox3d_warped"] = self.transform_bbox3d(
                        inputs["bbox3d"], pose_diff_cpu
                    )
                    inputs_seq[mod] = inputs_seq[
                        "bbox3d_warped"
                    ]  #
                    priors["bbox3d_warped"] = inputs_seq[
                        "bbox3d_warped"
                    ].clone()
                else:
                    inputs_seq[mod] = inputs["bbox3d"]
                inputs_seq[mod] = self.get_mod_emb_pre(
                    inputs_seq, mod, add_posi_embedd=self.add_posi_embedd
                )
            inputs_seq[mod] = self.add_bos_eos(inputs_seq, mod)

        tar_emb = torch.cat([inputs_seq[m] for m in mod_order], dim=-2).cuda()
        tar_emb = self.add_pos_emb(tar_emb)
        tar_emb = self.transformer.drop(tar_emb)
        kvcache_t = [None] * self.n_box_tar_layer
        for block, kvcache_t_block in zip(self.transformer.box_tar, kvcache_t):
            tar_emb, _ = block(tar_emb, kvcache=kvcache_t_block)
        tar_emb = self.transformer.ln_box_tar(tar_emb)
        mod_tar_embs = torch.split(
            tar_emb, [self.token_len[mod] for mod in mod_order], dim=2
        )
        tar_embs = {}
        for mod, mod_tar_emb in zip(mod_order, mod_tar_embs):
            tar_embs[mod] = mod_tar_emb
        return tar_embs, priors



    @staticmethod
    def get_mod_tokens(tokens, idx_begin, idx_end, mod=None):
        res_tokens = {}
        if mod is not None:
            assert isinstance(mod, list)
            for m in mod:
                if idx_begin >= tokens[m].size(1):
                    res_tokens[m] = None
                else:
                    res_tokens[m] = tokens[m][
                        :, idx_begin:idx_end, ...
                    ].clone()
        else:
            for m in tokens.keys():
                if idx_begin >= tokens[m].size(1):
                    res_tokens[m] = None
                else:
                    res_tokens[m] = tokens[m][
                        :, idx_begin:idx_end, ...
                    ].clone()
        return res_tokens


    def topk(self, logits: torch.Tensor, top_k: Optional[int] = None):
        """Optionally crop the logits to only the top k options."""
        if top_k is None:
            top_k = self.top_k
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = -float("Inf")

        if len(logits.shape) == 3:
            pred_token = self.sfmx_temp_sampling(logits[0, ...])[:, None, :]
            pred_token = pred_token.transpose(0, -1)
        elif len(logits.shape) == 2:
            pred_token = self.sfmx_temp_sampling(logits)[:, :, None]
        else:
            raise ValueError("logits shape not supported in topk sampling")
        return pred_token

    def sample_top_p(self, probs, p):
        """Perform top-p (nucleus) sampling on a probability distribution.

        Args:
            probs (torch.Tensor): Probability distribution tensor.
            p (float): Probability threshold for top-p sampling.

        Returns:
            torch.Tensor: Sampled token indices.

        Note:
            Top-p sampling selects the smallest set of tokens whose cumulative probability mass
            exceeds the threshold p. The distribution is renormalized based on the selected tokens.
        """
        # probs += torch.abs(torch.min(probs))
        # probs = probs / torch.sum(probs)
        flattened_flag = 0
        probs_save = probs.clone()

        if len(probs.shape) == 4:  # 处于训练状态，第一个维度是batch
            B, T, S, V = probs.size()
            probs = probs.view(B * T * S, -1)
            flattened_flag = 1

        elif len(probs.shape) == 3:  # 处于训练状态，第一个维度是batch
            B, S, V = probs.size()  # (1, 3, 1024)
            probs = probs.view(B * S, -1)
            flattened_flag = 2

        probs = probs.clone() / self.sfmx_temp
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(probs, dim=-1)

        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = (probs_sum - probs_sort) > p
        # mask = probs_sum > p

        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        next_token = torch.multinomial((probs_sort), num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        if flattened_flag == 1:
            next_token = next_token.view(B, T, S, -1)
        elif flattened_flag == 2:
            next_token = next_token.view(B, -1, S)

        # return next_token, probs_save
        return next_token

    def sfmx_temp_sampling(self, logits: torch.Tensor):
        """Do the sampling with softmax temperature."""
        logits = logits.clone() / self.sfmx_temp
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        tokens = torch.multinomial(probs, num_samples=1)
        return tokens

    def d_token_pos(self, mod_order):
        d_pos = {}
        curr_pos = 0
        for mod in mod_order:
            curr_pos += 1
            d_pos[curr_pos] = self.bos_eos[mod][0]
            curr_pos = curr_pos + self.token_len[mod] - 1
            d_pos[curr_pos] = self.bos_eos[mod][1]
        return d_pos

    def pos_mod(self, pos, mod_order):
        curr_pos = 0
        for mod in mod_order:
            curr_pos += 1
            if curr_pos <= pos <= curr_pos + self.token_len[mod] - 1:
                return mod
            curr_pos = curr_pos + self.token_len[mod] - 1

    def infer_ego_net(
        self,
        inputs: Dict[str, torch.Tensor],
        task_name: str = "pose_map_bbox3d",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform inference on the ego-prediction network."""
        ego_emb = self.forward_ego_net(inputs, task_name)
        logits_tar = self.transformer.head_ego(ego_emb)
        logits_tar = logits_tar[:, -1, :, :]
        # do the sampling with logits
        pred_token = self.token_sampler(logits_tar, self.sample_param)
        return pred_token


    def decode_pose(self, data):
        if isinstance(data["pose"], list):
            pose_tokens = torch.cat(data["pose"], dim=2)
        else:
            pose_tokens = data["pose"]
        B, T, S = pose_tokens.size()
        pose_tokens = rearrange(pose_tokens, "b t s -> (b t) s")
        if pose_tokens.device.type == "cuda":
            pose_tokens = pose_tokens.cpu()

        pose_values = self.ego_tokenlizer.decode(pose_tokens)
        pose_values = self.ego_norm.unnormalize_ego(pose_values)
        pose_values = pose_tokens.new_tensor(
            pose_values, dtype=torch.float32
        ).view(B, T, S)

        return pose_values




    def sample_next_token(
        self,
        curr_seq_len,
        curr_emb,
        mod,
        out_tokens,
        res_tokens,
        d_token_pos,
        previous_frame_tokens,
        cond_tar_emb=None,
        control_objects=None,
        mod_order=None,
        max_objects=None,
    ):
        """
        Sample the next token based on the mode and current embedding.
        """
        if curr_seq_len in d_token_pos:
            next_token = torch.tensor(d_token_pos[curr_seq_len]).cuda()
            next_token = self.transformer.axe(next_token)[None, None, None, :]
            out_tokens = torch.cat([out_tokens, next_token], dim=-2)
            return next_token, out_tokens, res_tokens

        else:
            if mod == "pose":
                logits_ar = self.transformer.head_ar_pose(curr_emb)
                logits_ar = logits_ar[:, -1, -1, :]
                next_token = self.token_sampler(logits_ar)
                res_tokens[mod].append(next_token)
                next_token = self.fouier_pe[next_token]
                out_tokens = torch.cat([out_tokens, next_token], dim=2)

            elif mod == "map":
                logits_ar = self.transformer.head_ar_map(curr_emb)[:, -1, -1, :]
                next_token = self.token_sampler(logits_ar, self.sample_param_map)
                if len(next_token.size()) == 2:
                    next_token = next_token.unsqueeze(1)
                res_tokens[mod].append(next_token)
                next_token = self.map_codebook(next_token)
                next_token = self.map_mlp_pre(next_token)
                out_tokens = torch.cat([out_tokens, next_token], dim=2)

            elif mod == "bbox3d":
                logits_ar = self.transformer.head_ar_bbox3d(curr_emb)[:, -1, -1, :]
                next_token = self.token_sampler(logits_ar.clone(), self.sample_param)

                # 计算 bbox3d token id
                if "map" in mod_order:
                    bbox3d_token_id = curr_seq_len - self.token_len["pose"] - self.token_len["map"] - 2
                else:
                    bbox3d_token_id = curr_seq_len - self.token_len["pose"] - 2

                previous_tokens = previous_frame_tokens["bbox3d"][:, -1, bbox3d_token_id].clone()
                bbox_tokens_start_index = 1032
                if control_objects is not None:
                    object_id = (curr_seq_len - bbox_tokens_start_index) // 11
                    if object_id in control_objects:
                        logits_ar[:, -1] = -np.inf
                        logits_tar = self.transformer.head_tar_bbox3d(cond_tar_emb)[:, -1, -1, :]
                        logits_tar[:, -1] = -np.inf
                        next_token = self.token_sampler(logits_tar, self.sample_param)

                # 避免 pad token
                if (
                    (next_token.item() == self.box3d_tokenlizer.pad_token)
                    and self.config.merage_ar_tar
                    and (previous_tokens.item() != self.box3d_tokenlizer.pad_token)
                    and (not self.only_ar)
                ):
                    if self.config.n_step != 1:
                        logits_tar = self.transformer.head_tar_n_step_bbox3d(cond_tar_emb)
                        logits_tar = torch.split(logits_tar, self.bbox3d_vocab_size, dim=-1)[0]
                        logits_tar = logits_tar[:, -1, -1, :]
                    else:
                        logits_tar = self.transformer.head_tar_bbox3d(cond_tar_emb)[:, -1, -1, :]
                    next_token = self.token_sampler(logits_tar, self.sample_param)

                if (previous_tokens.item() == self.box3d_tokenlizer.pad_token) and self.no_born:
                    print("FORCE NO BORN IN INFERNCE BBOX3D")
                    next_token = torch.tensor(self.box3d_tokenlizer.pad_token).cuda()
                    if object_id > max_objects:
                        next_token = torch.tensor(self.box3d_tokenlizer.pad_token).cuda()
                        next_token = next_token.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        res_len = self.out_seq_len - curr_seq_len
                        res_tokens[mod].extend([next_token] * res_len)
                        return next_token, out_tokens, res_tokens

                if self.rule_constrain:
                    next_token = self.rule_based_constraint(
                        current_token=next_token,
                        infered_token=res_tokens[mod],
                        out_tokens=out_tokens,
                        previous_tokens=previous_tokens,
                        curr_seq_len=curr_seq_len,
                    )

                if len(next_token.size()) == 2:
                    next_token = next_token.unsqueeze(1)
                res_tokens[mod].append(next_token)
                next_token = self.transformer.be(next_token)
                out_tokens = torch.cat([out_tokens, next_token], dim=2)

            elif mod == "image":
                logits_ar = self.transformer.head_ar_img(curr_emb)[:, -1, -1, :]
                next_token = self.token_sampler(logits_ar, self.topk_image).long()
                res_tokens[mod].append(next_token)
                next_token = self.img_codebook(next_token)
                next_token = self.img_mlp_pre(next_token)
                out_tokens = torch.cat([out_tokens, next_token], dim=2)

        return next_token, out_tokens, res_tokens











    def infer_oar_net(
        self,
        tar_emb: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        pred_task: str = "image",
        init_tokens: Optional[torch.Tensor] = None,
        cond_on_tar: bool = False,
        kvcache_ar: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        previous_frame_tokens: Optional[torch.Tensor] = None,
        control_objects=None,
        max_objects=100,
    ) -> Dict[str, torch.Tensor]:

        assert pred_task in self.task_names
        mod_order = self.task[pred_task]
        task_token = self.task_name_id[pred_task]
        task_token = torch.tensor([task_token]).cuda()
        mod_len = [self.token_len[mod] for mod in mod_order]
        out_seq_len = sum(mod_len)

        init_mods = []
        res_tokens = {}
        for mod in mod_order:
            res_tokens[mod] = []
        task_emb = self.transformer.tske(task_token)[None, None, :, :]
        b = 1
        if input_features is not None:
            b, _, _, _ = tar_emb["pose"].size()

        # use the predefined tokens and don't infer these tokens any more, used for control
        out_tokens = task_emb.new_tensor([], dtype=torch.long)
        exist_seq_len = 0
        self.decoded_bbox = []
        if init_tokens is not None:
            cond_tokens = []
            for mod in mod_order:
                if (
                    mod in init_tokens
                    and init_tokens[mod] is not None
                    and torch.numel(init_tokens[mod]) > 0
                ):
                    init_mods.append(mod)
                    res_tokens[mod] = init_tokens[mod].clone()
                    init_tokens[mod] = self.get_mod_emb_pre(init_tokens, mod)
                    init_tokens[mod] = self.add_bos_eos(init_tokens, mod)
                    cond_tokens.append(init_tokens[mod])
            if len(cond_tokens) > 0:
                cond_tokens = torch.cat(cond_tokens, dim=-2).cuda()
                out_tokens = cond_tokens
                exist_seq_len = cond_tokens.size(-2)
                b, _, _, _ = out_tokens.size()

        # infer the remained tokens
        d_token_pos = self.d_token_pos(mod_order)
        curr_seq_len = exist_seq_len
        map_warped = True

        # loop through the sequence length to get the tokens
        for i in range(exist_seq_len, out_seq_len):
            curr_seq_len += 1
            mod = self.pos_mod(curr_seq_len, mod_order)
            # concat the task embedding and the OAR embedding
            if b > 1:
                task_emb = task_emb.expand(b, 1, -1, -1)
            curr_emb = torch.cat([task_emb, out_tokens], dim=-2)
            # add the sequence position embeddings

            assert tar_emb is not None
            if mod == "map" and map_warped is None:
                pose_diff = self.decode_pose(res_tokens)
                map_warped = self.affine_transform(
                    input_features["map"][:, [-1], 1:-1, :], pose_diff
                )
                tar_emb[mod][:, [-1], 1:-1, :] += map_warped

            # add the condition embeddings from the PAR network
            cond_tar_emb = [tar_emb[m] for m in mod_order]
            cond_tar_emb = torch.cat(cond_tar_emb, dim=-2)[
                :, [-1], :curr_seq_len, :
            ]
            curr_emb = curr_emb + cond_tar_emb

            # forward the OAR embedding through the OAR network with kvcache
            if kvcache_ar is None:
                kvcache_ar = [torch.zeros(0), torch.zeros(0)] * self.n_oar_layer
            elif isinstance(kvcache_ar, list) and kvcache_ar[0][0].size(0) > 0:
                curr_emb = curr_emb[:, :, [-1], :]
            new_kvcache_ar = []
            for ar_block, kvcache_ar_block in zip(
                self.transformer.OAR, kvcache_ar
            ):
                curr_emb, cache_ele_ar = ar_block(
                    curr_emb, kvcache=kvcache_ar_block
                )
                new_kvcache_ar.append(cache_ele_ar)
            kvcache_ar = new_kvcache_ar
            curr_emb = self.transformer.ln_oar(curr_emb)
            
            # sample the tokens based on the OAR embedding
            next_token, out_tokens, res_tokens = self.sample_next_token(
                curr_seq_len,
                curr_emb,
                mod,
                out_tokens,
                res_tokens,
                d_token_pos,
                previous_frame_tokens,
                cond_tar_emb=cond_tar_emb,
                control_objects=control_objects,
                mod_order=mod_order,
                max_objects=max_objects,
            )




        # cat the predicted tokens
        for mod in mod_order:
            if mod in init_mods:
                continue
            res_tokens[mod] = torch.cat(res_tokens[mod], dim=2)

        return res_tokens

    def rule_based_constraint(
        self,
        current_token,
        infered_token,
        out_tokens,
        previous_tokens,
        curr_seq_len,
    ):
        # rule the generateion of the bbox3d tokens
        # improve the ability to aviod the coliision of new-born objecets
        pre_mod_token_len = 1032  #
        num_attritube = 11

        object_id = (
            curr_seq_len - pre_mod_token_len
        ) // num_attritube  #
        token_id = (
            curr_seq_len - pre_mod_token_len
        ) % num_attritube  #

        if current_token == self.box3d_tokenlizer.pad_token:
            return current_token

        if token_id == 0:
            # decode this box
            bbox_token = infered_token[
                -(num_attritube - 1) :
            ].copy()  # (1, 1, 11)
            bbox_token.append(current_token)
            bbox_token = (
                torch.cat(bbox_token, dim=0)
                .squeeze()
                .unsqueeze(0)
                .unsqueeze(0)
            )  # (11, 1, 1)

            bbox3d, bbox3d_class = self.box3d_tokenlizer.decode_single_objects(
                bbox_token.detach().cpu().numpy()
            )  # 1,1,11
            bbox3d = self.agent_norm.unnormalize_bbox3d(bbox3d[None, ...])[0][
                0
            ]
            # if 'none'
            # collision check
            if len(self.decoded_bbox) == 0:
                ego_bbox = np.array(
                    [
                        [
                            0,
                            0,
                            0,
                            self.ego_size["l"],
                            self.ego_size["w"],
                            self.ego_size["h"],
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                )
                self.decoded_bbox.append(ego_bbox.flatten())

            self.decoded_bbox.append(bbox3d)

            collision = self.box_overlap.check_collision(
                self.decoded_bbox, fliter=True
            )  # 

            previous_exist = (
                previous_tokens.item() == self.box3d_tokenlizer.pad_token
            )
            if previous_exist and collision:  #
                clean_collision = True
            # elif (previous_tokens.item() - current_token)>=200:
            #     clean_collision = True
            elif len(self.decoded_bbox) > 30 and previous_exist:
                clean_collision = True
            else:
                clean_collision = False

            if clean_collision:
                for j in range(num_attritube - 1):
                    index_j = j + 1
                    infered_token[-index_j] = (
                        torch.tensor(self.box3d_tokenlizer.pad_token)
                        .cuda()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    out_tokens[0, 0, -index_j] = self.transformer.be(
                        infered_token[-index_j].clone()
                    )
                current_token = (
                    torch.tensor(self.box3d_tokenlizer.pad_token)
                    .cuda()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                self.decoded_bbox.pop()
                return current_token

            else:
                return current_token

        else:
            return current_token

    def infer_ego_pose(
        self,
        inputs: Dict[str, torch.Tensor],
        cond_frames: int = 19,
        task_name: str = "pose_map_image",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_len = inputs["pose"].size(1)
        pred_tokens = []
        for i in range(input_len - cond_frames):
            curr_inputs = self.get_mod_tokens(inputs, i, cond_frames + i)
            ego_tokens = self.infer_ego_net(curr_inputs, task_name)
            pred_tokens.append(ego_tokens)
        pred_tokens = torch.cat(pred_tokens, dim=1)
        gt_tokens = inputs["pose"][:, cond_frames:, ...]
        pred_values = self.decode_pose({"pose": pred_tokens})
        gt_values = self.decode_pose({"pose": gt_tokens})
        pred_values[..., 2] = pred_values[..., 2] * 180.0 / np.pi
        gt_values[..., 2] = gt_values[..., 2] * 180.0 / np.pi

        return pred_values, gt_values

    def _inference(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        pred_task: str = "image",
        init_tokens: Optional[torch.Tensor] = None,
        control_test=False,  # control的object tokens to control the agent
        cond_on_tar: bool = False,
        test_map_affine: bool = False,
        all_gt_tokens=None,
        max_objects=100,
    ) -> Dict[str, torch.Tensor]:
        """Perform single frame inference on the model.
        Args:
            inputs (Dict[str, Any]): The input tensors for different
                modalities. Default is None.
            pred_task (str, optional): The predicted task. Default is "image".
            init_tokens (Optional[torch.Tensor]): The initial tokens.
                Default is None.
            cond_on_tar (bool): Whether to condition on TAR embeddings.
                Default is True.
            control_test (bool): Whether to control the agent or ego vehicle.
            all_gt_tokens: The ground truth tokens for all the modality
            max_objects: The maximum number of objects in the scene.
        Returns:
            Dict[str, torch.Tensor]: The inferred tokens for each module.
        """
        # task_name defines the modality order
        task_name = pred_task
        mod_order = self.task[pred_task]



        # Step1. Infer the Ego actions if not provided
        # Correspond to the "Ego-action Prediction" part in the method fig.
        if (
            init_tokens is not None
            and "pose" in init_tokens
            and init_tokens["pose"] is not None
        ):
            inputs["pose"] = torch.cat(
                [inputs["pose"], init_tokens["pose"]], dim=1
            )[:, 1:, ...]
        else:
            ego_tokens = self.infer_ego_net(inputs, task_name)
            inputs["pose"] = torch.cat(
                [inputs["pose"], ego_tokens], dim=1
            )[:, 1:, ...]
            if init_tokens is None:
                init_tokens = {}
            init_tokens["pose"] = ego_tokens

        # Set the controlled objects if we want to control the agent vehicle
        valid_index = None
        if (
            "bbox3d" in init_tokens
            and init_tokens["bbox3d"] is not None
            and control_test
        ):
            valid_index = init_tokens["bbox3d"][0, -1, :] != -1
            inputs["bbox3d"][0, -1, valid_index] = init_tokens["bbox3d"][
                0, -1, valid_index
            ]
            valid_index = valid_index.reshape(60, -1)
            control_objects = valid_index.any(dim=1)
            control_objects = np.where(
                control_objects.cpu().numpy() == True
            )
            init_tokens["bbox3d"] = None
        else:
            control_objects = None



        # Step2. Forward the input sequence through the cascade TAR network
        # Correspond to the "Temporal Autoregressive" part in the method fig.
        # Noteably, in the further exploration, we found that build repeted TAR modules for each modality leads to better performance. They share the same architecture as the original TAR module.
        if inputs is not None:

            if self.split_map_tar:
                tar_embs_dict, priors_maps_dict = self.forward_tar_for_map(inputs, task_name="pose_map")
                tar_embs_map = tar_embs_dict["map"]

            if self.split_box_tar:
                tar_embs_dict, priors_bbox3d_dict = self.forward_tar_for_box(inputs, task_name="pose_map_bbox3d")
                tar_embs_bbox3d = tar_embs_dict["bbox3d"]
                if not self.split_map_tar:
                    tar_embs_map = tar_embs_dict["map"]

            tar_emb, input_features, pose_diff = self.forward_tar_net(inputs, task_name)

            if self.split_map_tar or self.split_box_tar:  
                tar_emb["map"] = tar_embs_map
            if self.split_box_tar:
                tar_emb["bbox3d"] = tar_embs_bbox3d

            # take the aligned map features as prior in the OAR part
            if self.map_transform:
                if (
                    self.config.add_spatial_pos_embedd_on_map
                    and "map" in mod_order
                ):
                    input_features["map_warped"] = priors_maps_dict["map_warped"]  #
                if "map" in mod_order:
                    prior_map = torch.zeros_like(tar_emb["map"])
                    prior_map[:, :, 1:-1, :] += input_features["map_warped"]
                    tar_emb["map"] = (tar_emb["map"] + prior_map)  # res affine map features
        else:
            tar_emb, input_features = None, None


        # to aviod image token as init tokens.
        if init_tokens is not None:
            for mod in init_tokens:
                if (
                    (mod != "pose")
                    and (mod != "map")
                    and (mod != "bbox3d")
                ):  
                    init_tokens[mod] = None


        # Step3. Forward the tokens through the OAR network to get the predicted tokens
        # Correspond to the "Orederd  Autoregressin" part in the method fig.
        out_tokens = self.infer_oar_net(
            tar_emb,
            input_features,
            pred_task=task_name,
            init_tokens=init_tokens,
            cond_on_tar=cond_on_tar,
            previous_frame_tokens=inputs,
            control_objects=control_objects,
            max_objects=max_objects,
        )

        return out_tokens

    def inference(
        self,
        new_frames: int,
        cond_frames: int = 1,
        input_cond_frames: int = -1,
        pred_task: str = "image",
        input_cond_tokens: Optional[Dict[str, torch.Tensor]] = None,
        init_tokens: Optional[Dict[str, torch.Tensor]] = None,
        cond_on_tar: bool = False,
        test_map_affine: bool = False,
        max_objects=100,
        control_test=False,  #
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Perform inference using the TAR-OAR GPT model.

        Args:
        new_frames:    number of frames we predict
        cond_frames:   number of max historical frames used for conditioning
        input_cond_frames:  number of input historical frames, this whill increase with the inference
        pred_task:     the predicted task (pose-map-bbox3d-image)
        input_cond_tokens: historical tokens for conditioning
        init_tokens:   init tokens, such tokens will not be inferred, but directly used for conditioning other tokens
        cond_on_tar:   use both TAR and OAR for inference
        max_objects:   the maximum number of objects in the scene
        control_test:  whether to control the agent or ego vehicle.
        Returns:
        Dict[str, torch.Tensor]: The inferred tokens for each module.
        """
        assert pred_task in self.task_name_id
        mod_order = self.task[pred_task]

        if input_cond_frames == -1:
            input_cond_frames = cond_frames

        # get the history temporally conditioning tokens
        # history condtions
        if input_cond_tokens is not None:
            assert isinstance(input_cond_tokens, dict)
            out_tokens = {
                mod: input_cond_tokens[mod].clone() for mod in mod_order
            }
            cond_tokens = {
                mod: input_cond_tokens[mod].clone() for mod in mod_order
            }
        else:
            out_tokens = {}

        out_tokens = self.get_mod_tokens(
            out_tokens, 0, input_cond_frames, mod=mod_order
        )
        cond_tokens = self.get_mod_tokens(
            cond_tokens, 0, input_cond_frames, mod=mod_order
        )

        for idx in trange(new_frames):
            self.frame_idx = idx
            # update cond tokens
            if cond_tokens["pose"].shape[1] > cond_frames:
                cond_tokens = self.get_mod_tokens(
                    cond_tokens, -cond_frames, None, mod=mod_order
                )
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if init_tokens is not None:
                        assert isinstance(init_tokens, dict)
                        # get the guiding modality tokens for the AR network
                        curr_init_tokens = self.get_mod_tokens(
                            init_tokens, idx, idx + 1, mod=None
                        )
                        curr_init_tokens_copy = curr_init_tokens.copy()
                        if (
                            "pose" in curr_init_tokens
                        ):  #
                            if curr_init_tokens["pose"] is None:
                                init_tokens = None
                                control_test = False
                                curr_init_tokens_copy = None

                    else:
                        curr_init_tokens = None
                        curr_init_tokens_copy = None
                    # generate one frame tokens
                    tokens_dict = self._inference(
                        inputs=cond_tokens.copy(),
                        pred_task=pred_task,
                        init_tokens=curr_init_tokens_copy,
                        cond_on_tar=cond_on_tar,
                        test_map_affine=test_map_affine,
                        all_gt_tokens=input_cond_tokens,
                        max_objects=max_objects,
                        control_test=control_test,
                    )

            # concatenate the generated tokens and existing token
            # use the predefined tokens for control
            save_order = mod_order.copy()
            for mod in mod_order:
                if init_tokens is not None:
                    if mod in init_tokens and not (
                        control_test and mod == "bbox3d"
                    ):  
                        cond_tokens[mod] = torch.cat(
                            [cond_tokens[mod], curr_init_tokens[mod][:, :, :]],
                            dim=1,
                        )
                        out_tokens[mod] = torch.cat(
                            [out_tokens[mod], curr_init_tokens[mod][:, :, :]],
                            dim=1,
                        )
                    else:
                        cond_tokens[mod] = torch.cat(
                            [cond_tokens[mod], tokens_dict[mod]], dim=1
                        )
                        out_tokens[mod] = torch.cat(
                            [out_tokens[mod], tokens_dict[mod]], dim=1
                        )
                else:
                    cond_tokens[mod] = torch.cat(
                        [cond_tokens[mod], tokens_dict[mod]], dim=1
                    )
                    out_tokens[mod] = torch.cat(
                        [out_tokens[mod].cpu(), tokens_dict[mod].cpu()],
                        dim=1,  # 都放回cpu,避免OOD
                    )

        # convert to ny.array
        for mod in save_order:
            out_tokens[mod] = out_tokens[mod].cpu().numpy()
        return out_tokens




