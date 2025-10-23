import argparse
import os
import sys
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)
from pickle import FALSE, TRUE
from re import T
import numpy as np

from pytorch_lightning import Trainer
from mmcv.utils  import Registry, build_from_cfg
from mmcv import Config
from projects.registry import MODELS, DATASETS
from projects.tools.infer_fun import (
    load_model_paramter,
    mearge_params,
    mk_savedir,
    set_dataset_config,
    set_inference_setting,
    set_model_config,
)
from projects.tools.model_pl import UMGen_PL
import torch

import argparse

parser = argparse.ArgumentParser(description="UMGen_Evaluation")

parser.add_argument(
    "--pred_task",
    type=str,
    default="pose_map_bbox3d_image",
    help="pose_map | pose_map_bbox3d | pose_map_bbox3d_image",
)

parser.add_argument(
    "--ckpt_dir",
    type=str,
    default="data/weights/UMGen_Large.pt",
    help="path to the trained weights",
)

parser.add_argument(
    "--model_scale",
    type=str,
    default="larger",
    help="stander | larger",
)

parser.add_argument(
    "--infer_task",
    type=str,
    default="control",
    help="control | video",
)

parser.add_argument(
    "--rule_constrain",
    type=bool,
    default=True,
    help="use rule to constrain the bbox, especially for new-born objects",
)

parser.add_argument(
    "--set_num_new_frames",
    type=int,
    default=10,
    help="number of new frames (e.g. 30 for control test, 200 for video)",
)

parser.add_argument(
    "--spe_text",
    type=str,
    default="UMGen_Evaluating",
    help="special text tag for output naming",
)

parser.add_argument(
    "--force_vis",
    type=bool,
    default=True,
    help="force to visualize each scene",
)

parser.add_argument(
    "--put_text",
    type=bool,
    default=True,
    help="whether to overlay text on generated scenes",
)

parser.add_argument(
    "--save_video",
    type=bool,
    default=True,
    help="whether to generate video instead of images",
)

parser.add_argument(
    "--debug",
    type=bool,
    default=False,
    help="enable debug mode",
)

parser.add_argument(
    "--output_path",
    default="output/UMGen/",
    help="directory to save results",
)

parser.add_argument(
    "--map_decoder_weights_path",
    default="data/weights/map_vae.ckpt",
    help="path to pretrained map VAE weights",
)

parser.add_argument(
    "--image_decoder_weights_path",
    default="data/weights/image_vae.tar",
    help="path to pretrained image VAE weights",
)

parser.add_argument(
    "--launcher",
    type=str,
    choices=["torch", "mpi"],
    default=None,
    help="job launcher for multi-machine training",
)

args = parser.parse_args()


# infer_task:
# vedeo: used to generate video for visualization, all modalities are inferred by UMGen
# control: used to generate control signals for closed-loop simuation, some modalities are predefined

# load the basic config file
args.config_path = "projects/configs/UMGen_config_evaluation.py"

from projects.configs.UMGen_config_evaluation import (
    NuPlanTokenDataset,
    agent_norm,
    bbox3d_tokenizer,
    data_root_val,
    ego_norm,
    ego_pose_tokenizer,
    encode_pipeline,
    transforms_val,
)

args.dataset_type = NuPlanTokenDataset
args.dataset_name = "nuplan"
args.categories_file = "projects/configs/category.txt"
args.start_index = 10
args.data_test_root = data_root_val  #  data_root_val
args.transforms_val = transforms_val

args.bbox3d_tokenizer = bbox3d_tokenizer
args.agent_norm = agent_norm
args.ego_pose_tokenizer = ego_pose_tokenizer
args.ego_norm = ego_norm
args.sp_list = None  #
if args.infer_task == "control":
    # inference from the given dataset
    args.data_test_root = (
        ["data/controlled_scenes"]
    )


# --------------------------------- Set env -------------------------------- #
np.set_printoptions(suppress=True)

# load model configs and dataset
cfg = Config.fromfile(args.config_path)
infer_task_config = cfg["infer_task_config"]
# merage the args and config file
meraged_setting = mearge_params(infer_task_config, args)
model_config = cfg["model"]
model_config["config"].device_set = torch.device("cpu")  #
model_config["config"] = set_model_config(
    model_config["config"],
    meraged_setting,
    bbox3d_tokenizer,
    ego_pose_tokenizer,
)
# build dataset
dataset_config = set_dataset_config(meraged_setting)
infer_dataset = build_from_cfg(dataset_config, DATASETS)
# build models
model = build_from_cfg(model_config, MODELS)
model.eval()
# build dataloader
infer_loader = torch.utils.data.DataLoader(
    infer_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
)
# set the output path
output_path = args.output_path
meraged_setting['token_save_path'] = os.path.join(output_path, "saved_token/")  
meraged_setting['video_save_base_path'] = os.path.join(output_path,"video/")  
meraged_setting = set_inference_setting(meraged_setting, model_config, args.infer_task)
mk_savedir(meraged_setting)
# loading
if not args.debug:
    print("loading model from",meraged_setting.ckpt_dir)
    checkpoint = torch.load(meraged_setting.ckpt_dir, map_location="cpu")
    model = load_model_paramter(model, checkpoint)
else:
    print("no loading model, in debug mode")
# pytorch-lightning test
model_pl = UMGen_PL(meraged_setting, model, infer_loader)
trainer = Trainer(devices=[0], accelerator="cuda")
trainer.test(model_pl)
print("Sucess")
