import pickle
import os
import numpy as np
pkl_path = '/home/wuyanhao/WorkSpace/UMGen/data/controlled_scenes'
# 读取这个路径下的所有pkl文件
pkl_files = os.listdir(pkl_path)
# 遍历所有pkl文件
for pkl_file in pkl_files:
    # 读取pkl文件
    with open(os.path.join(pkl_path, pkl_file), 'rb') as f:
        data = pickle.load(f)
    # 提取control_tokens
    # print(data)

    # # 去除 data['dataset_token']中的'ori_image_path'和'file_name', 保留其他所有信息
    data['dataset_token'].pop('ori_image_path', None)
    data['dataset_token'].pop('file_name', None)
    data.pop('dataset_split', None)
    data.pop('scene_mode', None)
    data['scene_name'] = pkl_file.split('.')[0]
    print(data['scene_name'])


    # 重新保存
    with open(os.path.join(pkl_path, pkl_file), 'wb') as f:
        pickle.dump(data, f)
