# 
import argparse
import os

from typing_extensions import Concatenate

def mearge_params(source, target):
    # merage dicts
    if not isinstance(target, dict):
        target = target.__dict__
    for key in source:
        if key in target:
            if type(source[key]) == dict:
                mearge_params(source[key], target[key])
            else:
                source[key] = target[key]
    for key in target:
        if key not in source:
            source[key] = target[key]
    sample_img = "image" in target["pred_task"]
    source["sample_img"] = sample_img
    return source


def mk_savedir(meraged_setting):
    token_save_path = meraged_setting.token_save_path
    video_save_base_path = meraged_setting.video_save_base_path
    ckpt_save_dir = meraged_setting.ckpt_dir

    if not os.path.exists(token_save_path):
        print("maked dirs : ", token_save_path)
        os.makedirs(token_save_path, exist_ok=True)

    if not os.path.exists(video_save_base_path):
        os.makedirs(video_save_base_path, exist_ok=True)
        print("maked dirs : ", video_save_base_path)

    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir, exist_ok=True)
        print("maked dirs : ", ckpt_save_dir)


def load_model_paramter(model, checkpoint):
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    # print(model.load_state_dict(state_dict["module"], strict=False))  #
    model.load_state_dict(state_dict["module"], strict=False)
    return model





def set_inference_setting(meraged_setting, model_config, infer_task):
    # Setting configures based on the inference task
    if not isinstance(meraged_setting, argparse.Namespace):
        meraged_setting = argparse.Namespace(**meraged_setting)

    if not isinstance(model_config, argparse.Namespace):
        model_config = argparse.Namespace(**model_config)

    if infer_task == "video":
        num_new_frames = meraged_setting.set_num_new_frames
        meraged_setting.init_token_mod = None
        input_cond_frames = 20
    elif "control" in infer_task:
        input_cond_frames = 13  # 12 20
        num_new_frames = 30
        meraged_setting.init_token_mod = None
    else:
        num_new_frames = -1
        meraged_setting.init_token_mod = None
        input_cond_frames = 20

    meraged_setting.num_new_frames = num_new_frames
    meraged_setting.infer_from_gt = False  
    meraged_setting.input_cond_frames = input_cond_frames

    return meraged_setting


def set_model_config(
    model_config, in_param, bbox3d_tokenizer, ego_pose_tokenizer
):
    if not isinstance(in_param, argparse.Namespace):
        in_param = argparse.Namespace(**in_param)

    model_config.dropout = 0
    model_config.top_k = in_param.top_k
    model_config.p = in_param.top_p
    model_config.sample_method = in_param.sample_method

    model_config.num_attritube = 10  # 
    model_config.pad_to_length = 60  # number to object tokens to pad

    if in_param.sample_img:
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

    else:
        bos_eos = {"pose": [0, 1], "map": [2, 3], "bbox3d": [4, 5]}
        aux_vocab_size = 6
        vocab_len = {
            "bbox3d": len(bbox3d_tokenizer),  # 1024
            "map": 2,
            "pose": len(ego_pose_tokenizer) + 2,
        }
        token_len = {
            "bbox3d": bbox3d_tokenizer.seq_len + 2,  # 662
            "map": 32 * 32 + 2,
            "pose": ego_pose_tokenizer.seq_len + 2,
        }
        seq_len = 1693  # 660 + 2 + 32 * 32 + 2 + 3 + 2      # 660: box tokens, 32*32: map tokens, 3: pose tokens, 2: start and end token for each mod

    model_config.bos_eos = bos_eos
    model_config.aux_vocab_size = aux_vocab_size
    model_config.vocab_len = vocab_len
    model_config.token_len = token_len
    model_config.seq_len = seq_len

    if "stander" == in_param.model_scale:
        model_config.n_tar_layer = 24
        model_config.n_ar_layer = 24
    elif "larger" == in_param.model_scale:
        model_config.n_tar_layer = 36
        model_config.n_ar_layer = 36
    elif "debug" == in_param.model_scale:
        model_config.n_tar_layer = 1
        model_config.n_ar_layer = 1
        model_config.n_map_tar_layer = 1
        model_config.n_bbox3d_tar_layer = 1
        model_config.n_bbox3d_map_ca_layer = 1
        model_config.n_bbox3d_ego_ca_layer = 1
        model_config.n_step = 1
        print("Warning, using debug setting in model config, smallest model")
    else:
        raise ValueError("Unknow model scale")
    model_config.rule_constrain = in_param.rule_constrain
    return model_config


def set_dataset_config(in_param):
    # 根据输入参数设置数据集参数
    if not isinstance(in_param, argparse.Namespace):
        in_param = argparse.Namespace(**in_param)
    if "control" in in_param.infer_task:
        control_test = True
    else:
        control_test = False

    if "compare_video" == in_param.infer_task:
        # 这个和video的区别是他使用的是 init pose map bbox3d模态
        long_sceniors = True
        return_ori_image = False  # 返回真实图像 # 暂时不返回
        img_data_aug = in_param.image_transforms
        # print("return_ori_image is true, it casuse a more big cost")
        dataset_block_size = 20  # 按照block size一个block一个block的取，这个参数影响出视频的效果，越大对推理越友好，越小模态一致性更好(这是因为目前读取时每个clip中只保留了60个Objects，所以不同的Block衔接时会有些Objects不连续)
    if "video" == in_param.infer_task:
        long_sceniors = False
        return_ori_image = False
        img_data_aug = None
        dataset_block_size = (
            in_param.set_num_new_frames + in_param.cond_frames
        )  #
    else:
        long_sceniors = False
        return_ori_image = False
        img_data_aug = None
        dataset_block_size = in_param.set_num_new_frames + in_param.cond_frames

    # 根据 in_param 设置 model_config
    val_dataset_config = dict(
        type=in_param.dataset_type,
        data_root=in_param.data_test_root,
        training=False,
        block_size=dataset_block_size,  # set_num_new_frames要给大, 注意, block size会影响每一帧出现的objects数量。因为固定在一段时间内采样60个objects
        categories_file=in_param.categories_file,
        views=["CAM_F0"],
        sampling_gap=in_param.sampling_gap,
        transforms=in_param.transforms_val,
        inference_flag=True,
        start_index=in_param.start_index,
        sp_list=in_param.sp_list,
        sample_img=in_param.sample_img,
        return_scene_name=True,
        control_test=control_test,
        long_sceniors=long_sceniors,
        img_transform=img_data_aug,
        return_ori_image=return_ori_image,
    )

    return val_dataset_config
