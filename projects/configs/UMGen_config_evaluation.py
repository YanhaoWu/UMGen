import os
from argparse import Namespace
from pickle import TRUE
from re import T

sampling_gap = 4
box_transform = False
add_posi_embedd = True
add_spatial_pos_embedd_on_map = True  # running
ego_normolize = "standard"  # standard min_max
bbox_normlize = "min_max"  # standard min_max
pred_task = "pose_map_bbox3d_image"  # 'pose_map_bbox3d' 'pose_map' pose_map_bbox3d_image
split_map_tar = True    # used to further enhance the UMGen.
split_box_tar = True
min_max_standard_key = []
map_transform = True
n_step = 1  # number of steps for tar prediction
n_step_ar = 1  # number of steps for ar prediction

merage_ar_tar = True
only_ar = False
init_token_mod = None
split_map_ar = False
ar_dropout = 0
return_ori_image = False

n_tar_layer = 36
n_oar_layer = 36


n_ego_tar_layer = 12
n_ego_ca_layer = 12  #

n_map_tar_layer = 24  #
n_box_tar_layer = 24

n_head = 16
n_embd = 768

sp_list = None
sample_img = True

cond_tar_method = "sum"  
cond_frame = 20  #
block_size = ( cond_frame + n_step )
val_block_size = cond_frame + n_step
debug_device_id = [2]
summary_save_path = "../output/UMGen"
mask_temporal_pro = 0.0

# CheckPoint Setting ds-last-ckpt
checkpoint_path = '/horizon-bucket/mono6/users/yanhao1.wu/wyh_modelzoo/20240914_144000/mp_rank_00_model_states_39ep.pt'
allow_miss = (True,)
ignore_extra = (True,)

# Evaluator Setting
evaluator = Namespace(
    kernel_mul=1.0,
    kernel_num=1,
    attribute=["posi", "whl", "yaw", "speed", "cat"],
)


# Vocabulary Setting
pose_vocab_size = 1024
bbox3d_vocab_size = (
    1024 + 4
)  #   1024 for bbox3d, 3 for cat, 2 for start and end token, 1 for empty token
map_vocab_size = 8192
img_vocab_size = 8192

# Token Setting
map_token_len = 32 * 32 + 2
n_img_embd = 16
n_map_embd = 16  # in codebook encoder

# model setting
n_bbox3d_tar_layer = (24,)
n_bbox3d_map_ca_layer = (16,)
n_bbox3d_ego_ca_layer = (16,)



dropout = 0.15
bias = False
top_k = 5
top_k_map = 5

sample_method = "topp"  # topk topp
p = 0.4
sfmx_temp = 1.0
flash_attention = True
cond_prob = 1
log_freq = 50
posi_embed_type = None  # 
n_posiembed = 12 if posi_embed_type is not None else 0
ar_local_attention = False  # 
radius_threshold = 25  # 
num_attritube = 10


# tokenlizer_settting
category_file_path = "projects/configs/category.txt"
bbox3d_tokenlizer_target_key = ["bbox3d"]
box_token_start = 0  #
data_key = (
    "bbox_posi_x",
    "bbox_posi_y",
    "bbox_posi_z",
    "bbox_wlh_l",
    "bbox_wlh_w",
    "bbox_wlh_h",
    "bbox_yaw",
    "bbox_speed_x",
    "bbox_speed_y",
    "bbox_speed_z",
)
target_key = [data_key]

data_key_ego = ("pose_dx", "pose_dy", "pose_dheading")
target_key_ego = [data_key_ego]

bins_ego = [(-1.0, 1.0, 1024)]

# normalize range
normalize_range = {
    "bbox_posi_x": (-64, 64),
    "bbox_posi_y": (-64, 64),
    "bbox_posi_z": (-5, 5),
    "bbox_wlh_l": (0, 15),
    "bbox_wlh_w": (0, 4),
    "bbox_wlh_h": (0, 5),  #
    "bbox_yaw": (-3.14, 3.14),
    "bbox_speed_x": (-20, 20),
    "bbox_speed_y": (-15, 15),
    "bbox_speed_z": (-0.3, 0.3),
}  # pose?
#  dx [-1, 10], dy [-4, 4], dheading[-1.0, 1.0]
if "bbox_speed_x" in min_max_standard_key:
    print(
        "speed of bbox is normalized using standard parameters (mean and std)"
    )
    normalize_range["bbox_speed_x"] = (0, 10)
    normalize_range["bbox_speed_y"] = (0, 10)
    normalize_range["bbox_speed_z"] = (0, 1)  # 

agent_bins = [(0.0, 1.0, 1024)]
# Task Setting
task_name_id = {
    "pose_map_bbox3d_image": 6,
}
task_num = 7


data_root_val = [
    "data/tokenized_origin_scenes",  # noqa
]


# Transform the data to objects and dictionaries
import os

import sys
from argparse import Namespace
from copy import deepcopy

from projects.models.UMGen import UMGen
from projects.plugin.data.datasets.UMGen_nuplan_dataset import (
    NuPlanTokenDataset,
)
from projects.plugin.data.transforms.common import (
    MergeAttribute,
    SplitAttriute,
)
from projects.plugin.data.transforms.normalize import (
    Normalize,
    Normalize_Standard,
    ToTensor,
)
from projects.plugin.data.transforms.tokenizer import (  # BBoxTokenizer,
    BBox3DTokenizer,
    DigitalBinsTokenizer,
    IdentityTokenizer,
)

pad_to_length = 60
# init tokenizer
ego_pose_tokenizer = DigitalBinsTokenizer(
    bins=bins_ego,
    data_key="pose",
    seq_len=3,
    special_tokens=None,
    start=0,
)

bbox3d_tokenizer = BBox3DTokenizer(
    bins=agent_bins,  # 
    category_file=category_file_path,  #
    start=0,
    special_tokens=[],
    pad_to_length=pad_to_length,
    target_key=bbox3d_tokenlizer_target_key,
    shift_object_order_pro=0,
)

bbox3d_tokenizer_val = BBox3DTokenizer(
    bins=agent_bins,  # 
    category_file=category_file_path,  # 
    start=0,
    special_tokens=[],
    pad_to_length=pad_to_length,
    target_key=bbox3d_tokenlizer_target_key,
    shift_object_order_pro=0,
)


agent_norm = Normalize(
    data_key=data_key,
    max_min=normalize_range,
    min_max_standard_key=min_max_standard_key,
)

ego_norm = Normalize_Standard(
    data_key="pose",
    mean=[0, 0, 0],
    std=[
        10.0,
        4.0,
        1.0,
    ],  # dx [-1, 10], dy [-4, 4], dheading[-1.0, 1.0]
)



transforms = [
    SplitAttriute(input_key=["bbox3d"], target_key=target_key),
    agent_norm,
    MergeAttribute(
        input_key=["bbox3d"], target_key=target_key, merage_name=["bbox3d"]
    ),
    ego_norm,
    bbox3d_tokenizer,
    ego_pose_tokenizer,
    ToTensor(),
]

transforms_val = [
    SplitAttriute(input_key=["bbox3d"], target_key=target_key),
    agent_norm,
    MergeAttribute(
        input_key=["bbox3d"], target_key=target_key, merage_name=["bbox3d"]
    ),
    ego_norm,
    bbox3d_tokenizer,
    ego_pose_tokenizer,
    ToTensor(),
]


# decode and encode
encode_pipeline = [
    SplitAttriute(input_key=["bbox3d"], target_key=target_key),
    agent_norm,
    MergeAttribute(
        input_key=["bbox3d"], target_key=target_key, merage_name=["bbox3d"]
    ),
    bbox3d_tokenizer,
    ToTensor(),
]

bos_eos = {
    "pose": [0, 1],
    "map": [2, 3],
    "bbox3d": [4, 5],
    "image": [6, 7],
}
aux_vocab_size = 8
vocab_len = {
    "bbox3d": len(bbox3d_tokenizer),  # 1024
    "map": 2,
    "pose": len(ego_pose_tokenizer) + 2,
    "image": 2,
}
token_len = {
    "bbox3d": bbox3d_tokenizer.seq_len + 2,  # 662
    "map": 32 * 32 + 2,
    "pose": ego_pose_tokenizer.seq_len + 2,
    "image": 16 * 32 + 2,
}
seq_len = 2207  # 660 + 2 + 32 * 32 + 2 + 3 + 2 + 16 * 32 + 2    # 660: box tokens, 32*32: map tokens, 3: pose tokens, 2: start and end token for each mod


from projects.plugin.data.transforms.loading import (
    LoadMultiViewImageFromFiles,
    ResizeCropFlipRotImage,
    ToTensor_Collect,
)

# -------------------------------------------------------------------------
#
# configure the val dataset
#
# ------------------------------------------------------------------------
from projects.plugin.data.transforms.normalize import (
    Normalize,
    Normalize_Standard,
    ToTensor,
)

ida_aug_conf = dict(
    final_dim=(256, 512),
    bot_pct_lim=(0.0, 0.0),
)


image_transforms = [
    LoadMultiViewImageFromFiles(to_float32=True),
    ResizeCropFlipRotImage(ida_aug_conf, intrinsics=False),
    ToTensor_Collect(
        intrinsics=False, to_numpy=True
    ),  
]



# -------------------------------------------------------------------------
#
# model setting
#
# ------------------------------------------------------------------------
task = {
    "pose_map_bbox3d_image": ["pose", "map", "bbox3d", "image"],
    "pose_map_bbox3d": ["pose", "map", "bbox3d"],
    "pose_map": ["pose", "map"],
    "bbox3d": ["bbox3d"],
    "bbox3d_trans": ["bbox3d_trans"],
}


vocab_size = len(
    bbox3d_tokenizer
)  #  3 + 1024 + 2 + 1 # 3: catgories, 1024: bins for bbox, 2: <bos>, <eos>, 1: <pad>

model_config = Namespace(
    pred_task=pred_task,
    max_frame_len=100,  
    cond_frame=cond_frame,
    pose_vocab_size=pose_vocab_size,
    map_vocab_size=map_vocab_size,
    img_vocab_size=img_vocab_size,
    bbox3d_vocab_size=bbox3d_vocab_size,
    bos_eos=bos_eos,
    aux_vocab_size=aux_vocab_size,
    vocab_size=vocab_size,  # 
    mod_name=["bbox3d", "map", "pose"],  # 
    box3d_tokenlizer=bbox3d_tokenizer,
    agent_norm=agent_norm,
    ego_tokenlizer=ego_pose_tokenizer,
    ego_norm=ego_norm,
    task=task,
    task_prob=None,
    task_name_id=task_name_id,  # 
    task_num=task_num,
    vocab_len=vocab_len,
    token_len=token_len,
    map_codebook="projects/tokenizer/weights/map_codebook.pth",  # noqa
    img_codebook="projects/tokenizer/weights/img_codebook.pth",  # noqa
    pad_to_length=pad_to_length,
    seq_len=seq_len,  # 16x32 + 2 + 60x17 + 2 + 3 + 2
    n_tar_layer=n_tar_layer,  # 24
    n_oar_layer=n_oar_layer,  # 24
    n_bbox3d_tar_layer=n_bbox3d_tar_layer,
    n_bbox3d_map_ca_layer=n_bbox3d_map_ca_layer,
    n_bbox3d_ego_ca_layer=n_bbox3d_ego_ca_layer,
    n_ego_tar_layer=n_ego_tar_layer,
    n_ego_ca_layer=n_ego_ca_layer,
    n_map_tar_layer=n_map_tar_layer,
    n_box_tar_layer=n_box_tar_layer,
    n_head=n_head,  # 12
    n_embd=n_embd,  # 2048 
    n_img_embd=n_img_embd,
    n_map_embd=n_map_embd,
    dropout=dropout,  # 
    ar_dropout=ar_dropout,  # 
    add_posi_embedd=add_posi_embedd,
    add_spatial_pos_embedd_on_map=add_spatial_pos_embedd_on_map,
    bias=False,
    top_k=top_k,  # 64,
    top_k_map=top_k_map,
    sample_method="topp",  # topk topp
    p=p,
    sfmx_temp=sfmx_temp,  # softmax temperature
    flash_attention=flash_attention,
    cond_prob=cond_prob,  # 0.25,
    cond_tar_method=cond_tar_method,  # attention or  sum
    re_order_object=False,
    res_transform=False,  #
    box_transform=box_transform,  # 
    bbox_token_range=(0, 1023),
    add_t_pos=False,
    # other information
    save_path=summary_save_path,
    # device_ids=device_ids,
    submit=False,
    log_freq=log_freq,
    # model settings
    ar_local_attention=ar_local_attention,  #
    radius_threshold=radius_threshold,
    num_attritube=num_attritube,
    mask_temporal_pro=mask_temporal_pro,
    # local_range=local_range,
    split_map_tar=split_map_tar,
    split_map_ar=split_map_ar,
    split_box_tar=split_box_tar,
    split_image_ar=False,
    only_ar=only_ar,
    sample_img=sample_img,
    map_transform=map_transform,
    # for test
    noisy_test=False,
    n_posiembed=n_posiembed,
    posi_embed_type=posi_embed_type,
    n_step=n_step,
    n_step_ar=n_step_ar,
    block_size=block_size,
    merage_ar_tar=merage_ar_tar,
    train_only_ego=False,  #
    # decode and encode
    encode_pipeline=encode_pipeline,
)


model = dict(
    type=UMGen,
    config=model_config,
)

# -------------------------------------------------------------------------
# infer tast setting
import json

infer_task_config = {
    "project_name": "UMGen",
    "set_num_new_frames": 30,
    "cond_frames": 20,
    "sampling_gap": sampling_gap,
    "top_p": 0.4,
    "top_k": top_k,
    "sample_method": "topk",
    "re_order_object": False,
    "cond_tar_method": "sum",
    "gpu_id": "2",
    "force_download_weight": False,
    "save_video": True,
    "save_output": False,
    "only_generate_video": True,
    "debug": False,
    "generate_without_condition": False,
    "dataset": "nuplan",
    "evaluator": evaluator,
    "image_transforms": image_transforms,
    "map_transform": map_transform,
}
