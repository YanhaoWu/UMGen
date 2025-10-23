import glob
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.functional import Tensor

x_range = (-64, 64)
y_range = (-64, 64)
z_range = (-5, 5)
ego_whl = {"w": 2.297, "l": 5.176, "h": 1.777}

POLYLINE_TYPE = {
    # for lane
    "TYPE_UNDEFINED": -1,
    "TYPE_FREEWAY": 1,
    "TYPE_SURFACE_STREET": 2,
    "TYPE_BIKE_LANE": 3,
    # for roadline
    "TYPE_UNKNOWN": -1,
    "TYPE_BROKEN_SINGLE_WHITE": 6,
    "TYPE_SOLID_SINGLE_WHITE": 7,
    "TYPE_SOLID_DOUBLE_WHITE": 8,
    "TYPE_BROKEN_SINGLE_YELLOW": 9,
    "TYPE_BROKEN_DOUBLE_YELLOW": 10,
    "TYPE_SOLID_SINGLE_YELLOW": 11,
    "TYPE_SOLID_DOUBLE_YELLOW": 12,
    "TYPE_PASSING_DOUBLE_YELLOW": 13,
    # for roadedge
    "TYPE_ROAD_EDGE_BOUNDARY": 15,
    "TYPE_ROAD_EDGE_MEDIAN": 16,
    # for stopsign
    "TYPE_STOP_SIGN": 17,
    # for crosswalk
    "TYPE_CROSSWALK": 18,
    # for speed bump
    "TYPE_SPEED_BUMP": 19,
}


def fliter_and_map_object(box):
    # find unempty box and map the id
    non_empty_index = []
    # if len(box) != 60:
    #     print(
    #         "Warning, the number of object is not 60, you may have drop the empty tokens, the object id may be invalid")
    for i in range(len(box)):
        bbox_i = box[i]
        if (
            not (bbox_i[0] >= 63 or bbox_i[1] > 63) and bbox_i[3] != 15
        ):  #  Max range is 64, and 15 is the lost token
            non_empty_index.append(i)

    return box[non_empty_index], non_empty_index


def create_video_from_images(image_folder, video_path):
    images = glob.glob(f"{image_folder}/*.png")
    images.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    video = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (width, height)
    )
    for image in images:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()


# def draw_box(img, box):


def draw_ego(new_image, pose, color_draw=(0, 0, 255)):

    arrow_length = 3 * 4  #
    arrow_thickness = 3

    x, y = 0, 0
    l, w, h = ego_whl["l"] * 10, ego_whl["w"] * 10, ego_whl["h"] * 10
    yaw = pose[2]
    x += 80
    y += 80
    x = x * 10
    y = y * 10

    center_points = (int(x), int(y))

    # Calculate the corner points of the box
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    l_half = l / 2
    w_half = w / 2

    l_half = l / 2
    w_half = w / 2

    # Calculate the corner points in the original orientation
    corner_points = np.array(
        [
            [-l_half, -w_half],
            [l_half, -w_half],
            [l_half, w_half],
            [-l_half, w_half],
        ]
    ).astype(int)

    # Translate the corner points to the image center origin
    corner_points[:, 0] += x
    corner_points[:, 1] += y

    # draw the boxes
    for k in range(4):
        cv2.line(
            new_image,
            tuple(corner_points[k]),
            tuple(corner_points[(k + 1) % 4]),
            color_draw,
            2,
        )

    # draw the speed
    speed_x, speed_y = pose[0], pose[1]
    speed_y = -speed_y
    end_point = (
        int(x + speed_x * arrow_length),
        int(y + speed_y * arrow_length),
    )
    cv2.arrowedLine(
        new_image,
        center_points,
        end_point,
        color_draw,
        arrow_thickness,
        cv2.LINE_AA,
    )


def drwa_box(
    self,
    new_image,
    box,
    color=(70, 70, 70),
    show_id=None,
    p=None,
    addtion_ego=False,
):


    arrow_length = self.bbox3d_arrow_length
    arrow_thickness = self.bbox3d_arrow_thickness
    # box=box.reshape(-1, 1)
    origin_info = dict(
        bbox_posi=box[:, 0:3],
        bbox_whl=box[:, 3:6],
        bbox_yaw=box[:, 6:7],
        bbox_speed=box[:, 7:10],
        bbox_cat=box[:, 10:11],
        # bbox_collision=box[:, 11:12],
    )

    for j in range(box.shape[0]):
        x, y, z = origin_info["bbox_posi"][j]
        y = -y
        l, w, h = origin_info["bbox_whl"][j] * self.bbox3d_lwh_scale
        yaw = -origin_info["bbox_yaw"][j][0]
        if p is not None:
            if isinstance(p, int):
                max_pj = max(p)
                p_j = p[j]
            else:
                p_j = p[j]

        x += self.bbox3d_x_range[1]
        y += self.bbox3d_y_range[1]

        x = x * self.bbox3d_xyz_scle
        y = y * self.bbox3d_xyz_scle

        center_points = (int(x), int(y))

        # Calculate the corner points of the box
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        l_half = l / 2
        w_half = w / 2

        # Calculate the corner points in the original orientation
        corner_points = np.array(
            [
                [-l_half, -w_half],
                [l_half, -w_half],
                [l_half, w_half],
                [-l_half, w_half],
            ]
        )

        # Rotate the corner points around the center point
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corner_points_rotated = np.dot(corner_points, rotation_matrix.T)

        # Translate the corner points to the image center origin
        corner_points_rotated[:, 0] += x
        corner_points_rotated[:, 1] += y

        # Convert the corner points to integers
        corner_points_rotated = corner_points_rotated.astype(int)

        # Draw ego car
        if j == 0 and not addtion_ego:

            color_draw = self.ego_color

        # Draw other agents
        elif p is not None:
            if isinstance(p, int):
                if p_j == 20:
                    color_draw = color
                else:
                    color_draw = (0, int(255 * (p_j / max_pj)), 0)
            else:
                if p_j:
                    color_draw = (0, 0, 255)  
                else:
                    color_draw = color
        else:
            color_draw = color

        # Draw rotated boxes
        for k in range(4):
            cv2.line(
                new_image,
                tuple(corner_points_rotated[k]),
                tuple(corner_points_rotated[(k + 1) % 4]),
                color_draw,
                2,
            )

        # draw speed
        # cv2.putText(new_image, f'O Box {j+1}', (int(x - l/2), int(y - w/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        if origin_info["bbox_speed"] is not None:
            speed_x, speed_y, speed_z = origin_info["bbox_speed"][j]
            yaw_bbox = origin_info["bbox_yaw"][j]
            speed_y = -speed_y
            yaw_bbox = -yaw_bbox
            if self.rotate_speed:
                speed_x = speed_x * np.cos(yaw_bbox) + speed_y * np.sin(
                    yaw_bbox
                )
                speed_y = -speed_x * np.sin(yaw_bbox) + speed_y * np.cos(
                    yaw_bbox
                )
                end_point = (
                    int(x + speed_x * arrow_length),
                    int(y + speed_y * arrow_length),
                )
            else:
                end_point = (
                    int(x + speed_x * arrow_length),
                    int(y + speed_y * arrow_length),
                )
            cv2.arrowedLine(
                new_image,
                center_points,
                end_point,
                color_draw,
                arrow_thickness,
                cv2.LINE_AA,
            )

        # draw id
        if show_id is not None:
            cv2.putText(
                new_image,
                f"{show_id[j]}",
                (int(x - l / 2), int(y - w / 2) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

    return new_image


def visulize_objects_in_image(
    box=None,
    anno_box=None,
    collision=None,
    anno_collision=None,
    test_object=18,
    map=None,
    video_save_path="output/videos/test_0.mp4",
    project_name="test",
    scene_name="0",
    spe_text="p=0.5",
    save_video=True,
    set_index=None,
    addtion_ego=False,
    pose=None,
):
    # inputs are box and anno_box after decode and unnormalize
    # addtion_ego: weather to draw ego independently
    if pose is not None:
        addtion_ego = True
    else:
        addtion_ego = False

    arrow_length = 3
    arrow_thickness = 2
    save_base_path = os.path.join("output/image_cache", project_name, scene_name)
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)
        print("make dir ", save_base_path)

    if anno_box is not None:
        len_seq = anno_box.shape[0]
    else:
        len_seq = box.shape[0]

    for i in range(len_seq - 1):
        new_image = np.zeros((1600, 1600, 3), dtype=np.uint8)

        a_box_count = 0
        p_box_count = 0
        if anno_box is not None:
            a_box = anno_box[i]
            if anno_collision is not None:
                a_collision = anno_collision[i][test_object]
            else:
                a_collision = None

            new_image = drwa_box(
                new_image, a_box, p=a_collision, addtion_ego=addtion_ego
            )
            a_box_count = a_box.shape[0]

        if box is not None:
            p_box = box[i]
            if collision is not None:
                p_collision = collision[i][test_object]
            else:
                p_collision = None

            p_box, p_id = fliter_and_map_object(p_box)
            new_image = drwa_box(
                new_image,
                p_box,
                color=(255, 0, 0),
                show_id=p_id,
                p=p_collision,
                addtion_ego=addtion_ego,
            )
            p_box_count = p_box.shape[0]

        if addtion_ego:
            draw_ego(new_image, pose[i].copy(), color_draw=(0, 0, 255))

        frame_number = i
        cv2.putText(
            new_image,
            f"Frame {i}: p_box count = {p_box_count}, a_box count = {a_box_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            new_image,
            f"Project: {project_name}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        if spe_text is not None:
            cv2.putText(
                new_image,
                f"{spe_text}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        if set_index is None:
            save_img_path = f"{save_base_path}/{i}.png"
        else:
            save_img_path = f"{save_base_path}/{set_index}.png"

        cv2.imwrite(save_img_path, new_image)

        cv2.putText(
            new_image,
            f"Scene: {scene_name}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    if save_video:
        print("Start generating videos...")
        # Call the function to create the video
        create_video_from_images(save_base_path, video_save_path)
        print("Finish generating vidoes, saved to", video_save_path)
        cmd = "rm -rf " + save_base_path
        os.system(cmd)
        print("delete img cache")
    else:
        print("Saved images to", save_img_path)


class Visulizer:
    def __init__(
        self,
        video_save_path="output/videos/",
        video_pretext="test",
        width=256,
        height=256,
        project_name="test",
        spe_text="p=0.5",
        save_video=True,
        addtion_ego=False,
        resort_attritube=None,
        bbox3d_arrow_length_scale=1,
        rotate_speed=False,
        map_type="token",  # point / token
        dataset="nuplan",  # waymo / nuplan
        cond_frames=20,  # condition frames
        put_text=True,
    ):
        # video related setting
        self.video_save_base_path = video_save_path
        self.video_pretext = video_pretext
        self.width = width
        self.height = height
        self.project_name = project_name
        self.spe_text = spe_text
        self.save_video = save_video
        self.addtion_ego = addtion_ego
        self.resort_attritube = resort_attritube
        self.rotate_speed = rotate_speed  #
        self.map_type = map_type
        self.cond_frames = cond_frames
        self.put_text_on_img = put_text
        assert self.map_type in ["point", "token"]
        # img realted setting
        if self.width == 256:
            self.font_scale = 0.3
            self.font_tickness = 1  # 
            self.line_tickness = 2  # 
            self.bbox3d_arrow_length = (
                1 * bbox3d_arrow_length_scale
            )  # 
            self.bbox3d_arrow_thickness = 1  #
            self.text_posi = [
                (5, 20),
                (5, 30),
                (5, 40),
                (5, 50),
                (6, 60),
                (7, 70),
            ]
        else:
            self.font_scale = 0.5
            self.font_tickness = 1
            self.line_tickness = 2
            self.bbox3d_arrow_length = 1.5  #
            self.bbox3d_arrow_thickness = 2  #
            self.text_posi = [
                (10, 30),
                (10, 60),
                (10, 90),
                (10, 120),
                (10, 150),
                (10, 180),
            ]

        self.ego_arrow_length = self.bbox3d_arrow_length * 4  # m/s m/0.25s
        self.ego_arrow_thickness = self.bbox3d_arrow_thickness  #

        self.temp_cache_base_path = "output/tmp_cache"

        self.prediction_box_color = (0, 255, 0)  # 绿色
        self.ego_color = (0, 0, 255)
        self.anno_box_color = (255, 0, 0)

        # map related info
        self.map_x_range = (-64, 64)
        self.map_y_range = (-64, 64)

        # scale the bbox3d ego to the image
        self.bbox3d_x_range = (-64, 64)
        self.bbox3d_y_range = (-64, 64)
        self.bbox3d_lwh_scale = self.width / (
            self.bbox3d_x_range[1] - self.bbox3d_x_range[0]
        )  # 
        self.bbox3d_xyz_scle = self.width / (
            self.bbox3d_x_range[1] - self.bbox3d_x_range[0]
        )  # 一

        self.ego_x_range = self.bbox3d_x_range
        self.ego_y_range = self.bbox3d_y_range
        self.ego_lwh_scale = self.width / (
            self.ego_x_range[1] - self.ego_x_range[0]
        )  # 
        self.ego_xyz_scle = self.width / (
            self.ego_x_range[1] - self.ego_x_range[0]
        )  # 

        self.rotate_box = True  # 

        # for decode imgs
        seed = 0
        torch.manual_seed(seed)
        self.weights = None

        self.back_ground_color = (128, 128, 128)
        # for waymo
        self.waymo_color_setting = {
            "-1": (255, 0, 0),
            "1": (255, 0, 0),
            "2": (255, 0, 0),
            "3": (255, 0, 0),
            # ??
            "0": (255, 255, 255),  # white color
            "4": (255, 255, 255),  # white color
            "5": (255, 255, 255),  # white color
            "14": (255, 255, 255),  # white color
            # for roadline
            "6": (255, 255, 255),  # white color
            "7": (255, 255, 255),  # white color
            "8": (255, 255, 255),  # white color
            "9": (0, 255, 255),  # yellow color
            "10": (0, 255, 255),  # yellow color
            "11": (0, 255, 255),  # yellow color
            "12": (0, 255, 255),  # yellow color
            "13": (0, 255, 255),  # yellow color
            # for roadedge
            "15": (255, 0, 0),  # blue color
            "16": (255, 0, 0),  # blue color
            # for stopsign
            "17": (255, 0, 0),  # blue color
            # for crosswalk
            "18": (255, 0, 0),  # blue color
            # for speed bump
            "19": (255, 0, 0),  # blue color
        }

        assert dataset in ["waymo", "nuplan"]
        self.dataset = dataset
        if self.dataset == "nuplan":
            self.ego_whl = {"w": 2.297, "l": 5.176, "h": 1.777}
        else:
            self.ego_whl = {"w": 2.33, "l": 5.28, "h": 2.33}

        # update the scale
        self.scale_box_to_map()

        # for tmp_cache
        self.temp_cache_path = os.path.join(
            self.temp_cache_base_path, self.project_name
        )  # 
        if not os.path.exists(self.temp_cache_path):
            os.makedirs(self.temp_cache_path, exist_ok=True)

    def scale_box_to_map(self):

        width = self.width
        map_range = self.map_x_range[1] - self.map_x_range[0]
        pixel_per_meter = width / map_range

        self.bbox3d_lwh_scale = pixel_per_meter
        self.bbox3d_xyz_scle = pixel_per_meter

        self.ego_lwh_scale = pixel_per_meter
        self.ego_xyz_scle = pixel_per_meter

        print(
            "Update the scale of bbox3d and ego correspond to map range, in visulize scale_box_to_map fun"
        )

    def visulize_objects(
        self,
        box=None,  # 
        anno_box=None,  # 
        collision=None,  # 
        anno_collision=None,
        test_object=0,
        set_index=None,
        scene_name="0",
        images_all=None,
        view_mask=None,
    ):
        # inputs are box and anno_box after decode and unnormalize
        cache_path = os.path.join(
            self.temp_cache_path, self.project_name, scene_name
        )
        if not os.path.exists(cache_path):
            os.makedirs(cache_path, exist_ok=True)
            print("make dir ", cache_path)

        len_seq = 0
        if anno_box is not None:
            len_seq = anno_box.shape[0]
        if box is not None:
            len_seq = box.shape[0]

        if len_seq == 0:
            print("No box to visulize")
            return None, None, None

        image_all = []
        a_box_count_all = []
        p_box_count_all = []

        for i in range(len_seq):
            if images_all is not None:
                new_image = images_all[i]
            else:
                new_image = np.zeros(
                    (self.width, self.height, 3), dtype=np.uint8
                )
                new_image[:, :] = self.back_ground_color

            a_box_count = 0
            p_box_count = 0

            # drwa anno_box
            if anno_box is not None:
                a_box = anno_box[i]
                if anno_collision is not None:
                    a_collision = anno_collision[i]
                else:
                    a_collision = None
                new_image = self.draw_box(
                    new_image,
                    a_box,
                    color=self.anno_box_color,
                    p=a_collision,
                    addtion_ego=self.addtion_ego,
                )
                a_box_count = a_box.shape[0]

            # draw predict_box
            if box is not None:
                p_box = box[i]
                if collision is not None:
                    p_collision = collision[i]
                else:
                    p_collision = None
                p_box, p_id = fliter_and_map_object(p_box)
                new_image = self.draw_box(
                    new_image,
                    p_box,
                    color=self.prediction_box_color,
                    show_id=p_id,
                    p=p_collision,
                    addtion_ego=self.addtion_ego,
                )
                p_box_count = p_box.shape[0]
            if set_index is None:
                save_img_path = f"{cache_path}/{i}.png"
            else:
                save_img_path = f"{cache_path}/{set_index}.png"
            cv2.imwrite(save_img_path, new_image)

            image_all.append(new_image)
            a_box_count_all.append(a_box_count)
            p_box_count_all.append(p_box_count)

        return image_all, a_box_count_all, p_box_count_all

    def draw_ego(self, images, poses, color_draw=(0, 0, 255)):

        arrow_length = self.ego_arrow_length
        arrow_thickness = self.ego_arrow_thickness

        if self.rotate_box:
            poses = self.transform_box(poses, np.pi / 2)
            poses[:, 2] = np.pi / 2
        else:
            poses[:, 2] = 0

        new_images = []
        for i in range(len(poses)):
            if images is not None:
                new_image = images[i]
            else:
                new_image = np.zeros(
                    (self.width, self.height, 3), dtype=np.uint8
                )
                new_image[:, :] = self.back_ground_color

            pose = poses[i]
            x, y = 0, 0
            l, w, h = (
                self.ego_whl["l"] * self.ego_lwh_scale,
                self.ego_whl["w"] * self.ego_lwh_scale,
                self.ego_whl["h"] * self.ego_lwh_scale,
            )
            yaw = pose[2]

            x = x * self.ego_xyz_scle
            y = y * self.ego_xyz_scle

            x += self.width / 2
            y += self.height / 2

            center_points = (int(x), int(y))

            # Calculate the corner points of the box
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            l_half = l / 2
            w_half = w / 2

            l_half = l / 2
            w_half = w / 2

            # Calculate the corner points in the original orientation
            corner_points = np.array(
                [
                    [-l_half, -w_half],
                    [l_half, -w_half],
                    [l_half, w_half],
                    [-l_half, w_half],
                ]
            ).astype(int)

            rotation_matrix = np.array(
                [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]]
            )
            corner_points_rotated = np.dot(
                corner_points, rotation_matrix.T
            ).astype(int)

            # Translate the corner points to the image center origin
            corner_points_rotated[:, 0] += int(x)
            corner_points_rotated[:, 1] += int(y)

            # draw the boxes
            for k in range(4):
                cv2.line(
                    new_image,
                    tuple(corner_points_rotated[k]),
                    tuple(corner_points_rotated[(k + 1) % 4]),
                    color_draw,
                    self.line_tickness,
                )

            # draw the speed
            speed_x, speed_y = pose[0], pose[1]
            speed_y = -speed_y
            end_point = (
                int(x + speed_x * arrow_length),
                int(y + speed_y * arrow_length),
            )
            cv2.arrowedLine(
                new_image,
                center_points,
                end_point,
                color_draw,
                arrow_thickness,
                cv2.LINE_AA,
            )

            new_images.append(new_image)

        return new_images

    def transform_box(self, box, yaw, translation=None):
        # transform the box with yaw and translation
        # box: [n, 3]
        # yaw: 1
        # return: [n, 3]
        ones = np.ones((box.shape[0], 1))
        box3d_i_posi = np.concatenate((box[:, 0:2], ones), axis=-1)

        rotation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        rotated_point = (
            rotation_matrix @ box3d_i_posi.transpose(1, 0)
        ).transpose(1, 0)
        trans_box_posi = rotated_point[:, :2]
        if translation is not None:
            trans_box_posi += translation

        trans_box_yaw = box[:, -1] + yaw

        box[:, :2] = trans_box_posi
        box[:, -1] = trans_box_yaw

        return box

    def draw_box(
        self,
        new_image,
        box,
        color=(70, 70, 70),
        show_id=None,
        p=None,
        addtion_ego=False,
    ):
        arrow_length = self.bbox3d_arrow_length
        arrow_thickness = self.bbox3d_arrow_thickness

        if p is not None:
            if not isinstance(p, list):
                p = p.tolist()

        if self.rotate_box:
            box_temp = np.concatenate((box[:, 0:2], box[:, 6:7]), axis=-1)
            box_temp = self.transform_box(box_temp, np.pi / 2)
            box[:, 0:2] = box_temp[:, 0:2]
            box[:, 6] = box_temp[:, -1]

            box_temp = np.concatenate((box[:, 7:9], box[:, 6:7]), axis=-1)
            box_temp = self.transform_box(box_temp, np.pi / 2)
            box[:, 7:9] = box_temp[:, 0:2]

        # box=box.reshape(-1, 1)
        origin_info = dict(
            bbox_posi=box[:, 0:3],
            bbox_whl=box[:, 3:6],
            bbox_yaw=box[:, 6:7],
            bbox_speed=box[:, 7:10],
            bbox_cat=box[:, 10:11],
            # bbox_collision=box[:, 11:12],
        )

        for j in range(box.shape[0]):
            x, y, z = origin_info["bbox_posi"][j]
            y = -y
            l, w, h = origin_info["bbox_whl"][j] * self.bbox3d_lwh_scale
            yaw = -origin_info["bbox_yaw"][j][0]

            x = x * self.bbox3d_xyz_scle
            y = y * self.bbox3d_xyz_scle

            x += self.width / 2
            y += self.height / 2

            center_points = (int(x), int(y))

            # Calculate the corner points of the box
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            l_half = l / 2
            w_half = w / 2

            # Calculate the corner points in the original orientation
            corner_points = np.array(
                [
                    [-l_half, -w_half],
                    [l_half, -w_half],
                    [l_half, w_half],
                    [-l_half, w_half],
                ]
            )

            # Rotate the corner points around the center point
            rotation_matrix = np.array(
                [[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]]
            )
            corner_points_rotated = np.dot(corner_points, rotation_matrix.T)

            # Translate the corner points to the image center origin
            corner_points_rotated[:, 0] += x
            corner_points_rotated[:, 1] += y

            # Convert the corner points to integers
            corner_points_rotated = corner_points_rotated.astype(int)

            # 
            if j == 0 and not addtion_ego:
                color_draw = self.ego_color

            # 
            elif p is not None:
                test_id = show_id[j] if show_id is not None else j
                if test_id in p:
                    # 粉红色
                    color_draw = (255, 0, 255)
                    # color_draw = () # 粉棕色
                else:
                    color_draw = color
            elif l < 4 or w < 4:
                color_draw = (0, 165, 255)  # 
            else:
                color_draw = color

            # Draw rotated boxes
            for k in range(4):
                cv2.line(
                    new_image,
                    tuple(corner_points_rotated[k]),
                    tuple(corner_points_rotated[(k + 1) % 4]),
                    color_draw,
                    self.line_tickness,
                )
            # cv2.putText(new_image, f'O Box {j+1}', (int(x - l/2), int(y - w/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            if origin_info["bbox_speed"] is not None:
                speed_x, speed_y, speed_z = origin_info["bbox_speed"][j]
                yaw_bbox = origin_info["bbox_yaw"][j]
                if self.rotate_speed:  # 
                    if self.rotate_box:
                        yaw_bbox = -(yaw_bbox - np.pi / 2)
                    else:
                        yaw_bbox = -yaw_bbox
                    speed_x_rotate = speed_x * np.cos(
                        yaw_bbox
                    ) + speed_y * np.sin(yaw_bbox)
                    speed_y_rotate = -speed_x * np.sin(
                        yaw_bbox
                    ) + speed_y * np.cos(yaw_bbox)
                    speed_y_rotate = -speed_y_rotate
                    end_point = (
                        int(x + speed_x_rotate * arrow_length),
                        int(y + speed_y_rotate * arrow_length),
                    )
                else:
                    speed_y = -speed_y
                    end_point = (
                        int(x + speed_x * arrow_length),
                        int(y + speed_y * arrow_length),
                    )
                cv2.arrowedLine(
                    new_image,
                    center_points,
                    end_point,
                    color_draw,
                    arrow_thickness,
                    cv2.LINE_AA,
                )

            # 绘制id
            if self.put_text_on_img:
                if show_id is not None:
                    cv2.putText(
                        new_image,
                        f"{show_id[j]}",
                        (int(x - l / 2), int(y - w / 2) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        (0, 255, 0),
                        self.font_tickness,
                    )

        return new_image

    def put_text(
        self,
        images,
        a_box_count_all=None,
        p_box_count_all=None,
        scene_name=None,
        pose=None,
        real_pose=None,
        spe_text=None,
    ):
        new_images = []
        for i in range(len(images)):
            if i < self.cond_frames:
                set_color = (0, 0, 255)  # 
            else:
                set_color = (255, 255, 255)  # 
            new_image = images[i].copy()  # without copy will casuse bug
            if a_box_count_all is not None:
                a_box_count = a_box_count_all[i]
                p_box_count = p_box_count_all[i]
            else:
                a_box_count = 0
                p_box_count = 0

            if spe_text is not None:
                cv2.putText(
                    new_image,
                    f"{spe_text}",
                    self.text_posi[0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    set_color,
                    self.font_tickness,
                )
            else:
                cv2.putText(
                    new_image,
                    f"Frame {i}: pbox={p_box_count}, abox={a_box_count}",
                    self.text_posi[0],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    set_color,
                    self.font_tickness,
                )
                cv2.putText(
                    new_image,
                    f"Project: {self.project_name}",
                    self.text_posi[1],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    set_color,
                    self.font_tickness,
                )
                if self.spe_text is not None:
                    cv2.putText(
                        new_image,
                        f"{self.spe_text}",
                        self.text_posi[2],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        set_color,
                        self.font_tickness,
                    )
                if scene_name is not None:
                    cv2.putText(
                        new_image,
                        f"Scene: {scene_name}",
                        self.text_posi[3],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        set_color,
                        self.font_tickness,
                    )
                if pose is not None:
                    # 
                    pose = np.round(pose, 2)
                    cv2.putText(
                        new_image,
                        f"Pose: ({pose[i][0]:.2f}, {pose[i][1]:.2f}, {pose[i][2]:.2f})",
                        self.text_posi[4],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale,
                        set_color,
                        self.font_tickness,
                    )
                if real_pose is not None:
                    real_pose = np.round(real_pose, 2)
                    if i >= len(real_pose):
                        cv2.putText(
                            new_image,
                            f"GTPose: out of annotaion",
                            self.text_posi[5],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_scale,
                            set_color,
                            self.font_tickness,
                        )
                    else:
                        cv2.putText(
                            new_image,
                            f"GTPose: ({real_pose[i][0]:.2f}, {real_pose[i][1]:.2f}, {real_pose[i][2]:.2f})",
                            self.text_posi[5],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_scale,
                            set_color,
                            self.font_tickness,
                        )

            new_images.append(new_image)
        return new_images

    def generate_img_and_video(
        self,
        images_all,
        set_index=None,
        scene_name="0",
        base_video_save_path=None,
    ):

        for i in range(len(images_all)):
            new_image = images_all[i]

            if set_index is None:
                save_img_path = f"{self.temp_cache_path}/{i}.png"
            else:
                save_img_path = f"{self.temp_cache_path}/{set_index}.png"
            cv2.imwrite(save_img_path, new_image)

        if self.save_video:
            if base_video_save_path is None:
                base_video_save_path = self.video_save_base_path

            if not os.path.exists(base_video_save_path):
                os.makedirs(base_video_save_path, exist_ok=True)
                print("make dir ", base_video_save_path)

            video_save_path = os.path.join(
                base_video_save_path, f"{self.video_pretext}_{scene_name}.mp4"
            )

            # print("Start generating videos...")
            # Call the function to create the video
            create_video_from_images(self.temp_cache_path, video_save_path)
            print("Finish generating vidoes, saved to", video_save_path)
            cmd = "rm -rf " + self.temp_cache_path
            os.system(cmd)
            # print("delete img cache")
            if not os.path.exists(self.temp_cache_path):
                os.makedirs(self.temp_cache_path, exist_ok=True)

        else:
            print("Saved images to", save_img_path)

    def postprocess_image(self, image, renormalize=True):
        if renormalize:
            image = torch.clamp((image.float() + 1.0) / 2.0, min=0.0, max=1.0)
        else:
            image = torch.clamp(image.float(), min=0.0, max=1.0)
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = (image * 255).astype(np.uint8)
        return image

    def to_rgb(self, x, seed=0):
        if self.weights is None:
            self.weights = torch.randn(3, x.shape[0], 1, 1).to(x)
        x = F.conv2d(x, weight=self.weights.clone())
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x

    def decode_maps_from_raster(self, rasters):

        decoded_map = []
        for i in range(rasters.shape[0]):
            raster = rasters[i]
            rasters_rgb = self.to_rgb(raster.to(torch.int32))

            decoded_map.append(rasters_rgb)

        return torch.stack(decoded_map)

    def draw_map(self, maps, images_all):

        maps = self.postprocess_image(maps)
        generated_imgs = []
        for i in range(len(maps)):
            new_image = maps[i]
            new_image = new_image.transpose(
                1, 2, 0
            )  #

            new_image = cv2.resize(
                new_image,
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )
            if images_all is not None:
                image_pre = images_all[i]
            else:
                image_pre = np.zeros(
                    (self.width, self.height, 3), dtype=np.uint8
                )
                image_pre[:, :] = self.back_ground_color

            if new_image.shape != image_pre.shape:
                padding_img = np.zeros(
                    (self.width, self.height, 3), dtype=np.uint8
                )
                padding_height, padding_width = padding_img.shape[:2]
                new_height, new_width = new_image.shape[:2]
                start_y = (padding_height - new_height) // 2
                start_x = (padding_width - new_width) // 2
                padding_img[
                    start_y : start_y + new_height,
                    start_x : start_x + new_width,
                ] = new_image
                new_image = padding_img

            # 
            mask = np.any(image_pre != self.back_ground_color[0], axis=-1)
            # 
            new_image[mask] = image_pre[mask]

            generated_imgs.append(new_image)
        return generated_imgs

    def rotate_image(self, images_all):
        for i in range(len(images_all)):
            new_image = images_all[i]
            new_image = np.rot90(new_image, 2)
            images_all[i] = new_image
        return images_all

    def concatenate_images(self, image_mutil, mode="horizontal"):
        max_width = 0
        max_height = 0
        total_width = 0
        total_hegiht = 0
        frames = len(list(image_mutil.values())[0])
        for key, images in image_mutil.items():
            for img in images:
                max_height = max(max_height, img.shape[0])
                max_width = max(max_width, img.shape[1])
            total_width += images[0].shape[1]
            total_hegiht += images[0].shape[0]
        img_gap = 20  # 拼接的图像的中间距离

        # 优先处理ori_image
        key_list = list(image_mutil.keys())
        if "ori_image" in key_list:
            key_list.remove("ori_image")
            key_list.insert(0, "ori_image")

        concatenated_image_all = []
        for i in range(frames):
            current_y = 0
            if mode == "horizontal":
                concatenated_image = np.zeros(
                    (max_height, total_width, 3), dtype=np.uint8
                )
                for j in range(len(image_mutil.keys())):
                    key = key_list[j]
                    img = image_mutil[key][i]

                    height, width = img.shape[:2]
                    concatenated_image[
                        :height, current_y : current_y + width
                    ] = img

                    current_y += width
                concatenated_image_all.append(concatenated_image)
            else:
                concatenated_image = np.zeros(
                    (total_hegiht, max_width, 3), dtype=np.uint8
                )
                for j in range(len(image_mutil.keys())):
                    key = key_list[j]
                    if i >= len(image_mutil[key]):
                        img = image_mutil[key][-1]
                    else:
                        img = image_mutil[key][i]

                    height, width = img.shape[:2]
                    concatenated_image[
                        current_y : current_y + height, :width
                    ] = img

                    current_y += height
                concatenated_image_all.append(concatenated_image)

        return concatenated_image_all

    def draw_tokens(self, tokens, images_all, H=32, W=32):

        scale = 5
        tokens = tokens.reshape(-1, H, W)
        token_size = int(self.width / H * scale)  # 
        generated_imgs = []
        for k in range(len(tokens)):
            if images_all is not None:
                image_pre = images_all[k]
            else:
                image_pre = np.zeros(
                    (self.width * scale, self.height * scale, 3),
                    dtype=np.uint8,
                )
                image_pre[:, :] = self.back_ground_color

            for i in range(H):
                for j in range(W):
                    c = tokens[k, i, j]
                    text = f"{int(c)}"
                    x = j * token_size
                    y = i * token_size
                    cv2.putText(
                        image_pre,
                        text,
                        (x + 2, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 0, 255),
                        1,
                    )

                    if j < W - 1:  # 
                        cv2.line(
                            image_pre,
                            (x + token_size, 0),
                            (x + token_size, self.height * scale),
                            (255, 0, 0),
                            1,
                        )
                    if i < H - 1:  # 
                        cv2.line(
                            image_pre,
                            (0, y + token_size),
                            (self.width * scale, y + token_size),
                            (255, 0, 0),
                            1,
                        )
            generated_imgs.append(image_pre)
        return generated_imgs

    def resort_box(self, bbox3d):
        if len(bbox3d.shape) == 3:
            T, N, S = bbox3d.shape
        else:
            N, S = bbox3d.shape
            T = 1
            bbox3d = bbox3d.reshape(T, N, S)
        if bbox3d is None:
            return None
        new_bbox3d_list = []
        for i in range(len(bbox3d)):
            new_bbox3d = np.zeros(shape=(N, 10))
            for j in range(len(bbox3d[i])):
                # copy speed from pose_diff
                new_bbox3d[j][0:2] = bbox3d[i][j][3:5]  # x, y
                new_bbox3d[j][2] = 0  # z

                # copy attritube from bbox3d
                new_bbox3d[j][3:5] = bbox3d[i][j][5:7]  # l w h
                new_bbox3d[j][5] = 1  # l w h
                new_bbox3d[j][6] = bbox3d[i][j][7]
                new_bbox3d[j][7:10] = bbox3d[i][j][
                    0:3
                ]  # dx,dy,dhead -> dx, dy, dz
            new_bbox3d_list.append(new_bbox3d)

        return np.array(new_bbox3d_list)


    def draw_point_map(self, map_polylines, images_all=None):
        # [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]

        new_image = []

        # 画出地图
        for frame in range(map_polylines.shape[0]):
            map_frame = map_polylines[frame]
            if images_all is None:
                # 
                image_pre = np.zeros(
                    (self.width, self.height, 3), dtype=np.uint8
                )
                # 
                image_pre[:, :] = self.back_ground_color
            else:
                image_pre = images_all[frame]

            for i in range(map_frame.shape[0]):
                polyline = map_frame[i]
                x_coords = polyline[:, 0]
                y_coords = polyline[:, 1]
                line_type = polyline[:, -3]
                # 
                mask = (
                    (x_coords > -64)
                    & (x_coords < 64)
                    & (y_coords > -64)
                    & (y_coords < 64)
                )
                x_coords_filtered = x_coords[mask]
                y_coords_filtered = y_coords[mask]

                for x, y, point_type in zip(
                    x_coords_filtered, y_coords_filtered, line_type
                ):
                    # 
                    img_x = int((-x + 64) * (self.width / 128))
                    img_y = int((-y + 64) * (self.height / 128))

                    set_color = self.waymo_color_setting[str(int(point_type))]

                    if (
                        np.sum(
                            image_pre[img_x, img_y]
                            == self.back_ground_color[0]
                        )
                        == 3
                    ) or (np.sum(image_pre[img_x, img_y] == 0) == 3):
                        image_pre[img_x, img_y] = set_color

            new_image.append(image_pre)

        return new_image

    def merage_image_to_video(
        self,
        video_path,
        image,
        start_index=10,
        scene_name="test",
        image_text="diffusion",
    ):
        # 
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        # 
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 
        video_save_path = os.path.join(
            self.video_save_base_path, f"{self.video_pretext}_{scene_name}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        image = self.postprocess_image(image, renormalize=True)
        image = image.transpose(0, 2, 3, 1)
        img_height, img_width = image.shape[1:3]
        if img_height > video_height / 2 or img_width > video_width / 2:
            sp_flag = True
            combined_height = video_height + img_height
            combined_width = max(video_width, img_width)
        else:
            combined_height = video_height
            combined_width = video_width
            sp_flag = False

        out = cv2.VideoWriter(
            video_save_path, fourcc, fps, (combined_width, combined_height)
        )

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index >= start_index:
                # 
                image_index = frame_index - start_index

                if image_index >= len(image):
                    break

                set_color = (255, 255, 255)
                image_now = image[image_index]
                image_now = cv2.putText(
                    image_now.copy(),
                    image_text,
                    (1, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    set_color,
                    2,
                )
                img_height, img_width, _ = image_now.shape
                x_offset = video_width - img_width
                y_offset = video_height - img_height

                # 
                if sp_flag:
                    # 
                    combined_frame = np.zeros(
                        (combined_height, combined_width, 3), dtype=np.uint8
                    )
                    combined_frame[:img_height, :img_width] = image_now
                    combined_frame[
                        img_height : img_height + video_height, :video_width
                    ] = frame
                    frame = combined_frame
                else:
                    frame[
                        y_offset : y_offset + img_height,
                        x_offset : x_offset + img_width,
                    ] = image_now

            else:
                if sp_flag:
                    combined_frame = np.zeros(
                        (combined_height, combined_width, 3), dtype=np.uint8
                    )
                    combined_frame[
                        img_height : img_height + video_height, :video_width
                    ] = frame
                    frame = combined_frame

            out.write(frame)
            frame_index += 1

        cap.release()
        out.release()
        print("Video processing completed and saved as", video_save_path)

    def visulize_diffusion_image(
        self, images_dict, scene_name, cond_frame=3, renormalize=False
    ):

        # 
        self.temp_cache_path = os.path.join(
            self.temp_cache_base_path, self.project_name
        )  # 
        if not os.path.exists(self.temp_cache_path):
            os.makedirs(self.temp_cache_path, exist_ok=True)

        image_mutil = {}
        for key, images in images_dict.items():

            if key == "ori_image":
                ori_image = images
                ori_image = self.postprocess_image(
                    ori_image, renormalize=renormalize
                )  # 
                ori_image = ori_image.transpose(0, 2, 3, 1)
                set_color = (0, 0, 255)
                for i in range(len(ori_image)):
                    ori_image[i] = cv2.putText(
                        ori_image[i].copy(),
                        key,
                        (1, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        set_color,
                        2,
                    )

                image_mutil["ori_image"] = ori_image

            elif key == "vq_recon":
                decoded_image = images
                decoded_image = self.postprocess_image(
                    decoded_image
                )  # 默认不需要renormalize
                decoded_image = decoded_image.transpose(0, 2, 3, 1)
                set_color = (0, 0, 255)
                for i in range(len(decoded_image)):
                    decoded_image[i] = cv2.putText(
                        decoded_image[i].copy(),
                        key,
                        (1, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        set_color,
                        2,
                    )
                image_mutil["decoded_image"] = decoded_image

            elif key == "diffusion_image":
                diffusion_image = images
                diffusion_image = self.postprocess_image(
                    diffusion_image, renormalize=renormalize
                )
                diffusion_image = diffusion_image.transpose(0, 2, 3, 1)
                set_color = (255, 255, 255)
                for i in range(len(diffusion_image)):
                    if i < cond_frame:
                        set_color = (0, 0, 255)
                    else:
                        set_color = (255, 255, 255)
                    diffusion_image[i] = cv2.putText(
                        diffusion_image[i].copy(),
                        key,
                        (1, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        set_color,
                        2,
                    )

                image_mutil["diffusion_image"] = diffusion_image

            elif key == "diffusion_image_cond_past":
                diffusion_image = images
                diffusion_image = self.postprocess_image(
                    diffusion_image, renormalize=renormalize
                )
                diffusion_image = diffusion_image.transpose(0, 2, 3, 1)
                for i in range(len(diffusion_image)):
                    if i < cond_frame:
                        set_color = (0, 0, 255)
                    else:
                        set_color = (255, 255, 255)
                    diffusion_image[i] = cv2.putText(
                        diffusion_image[i].copy(),
                        key,
                        (1, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        set_color,
                        2,
                    )

                image_mutil["diffusion_image_cond_past"] = diffusion_image

        # 
        # 
        cat_imgs = self.concatenate_images(image_mutil, mode="vertical")
        # 

        self.generate_img_and_video(cat_imgs, scene_name=scene_name)

    def vis_pred_video(
        self, decoded_image, scene_name, video_type="pred", renormalize=True
    ):
        self.temp_cache_path = os.path.join(
            self.temp_cache_base_path, self.project_name, scene_name
        )  # 
        if not os.path.exists(self.temp_cache_path):
            os.makedirs(self.temp_cache_path, exist_ok=True)

        # 
        if not isinstance(decoded_image, torch.Tensor):
            decoded_image = torch.tensor(decoded_image)

        decoded_image = self.postprocess_image(
            decoded_image, renormalize=renormalize
        )

        decoded_image = decoded_image.transpose(0, 2, 3, 1)
        basename = os.path.basename(self.video_save_base_path)  # 
        if len(basename) == 0:
            basename = os.path.basename(self.video_save_base_path[:-1])
        video_save_base_path = self.video_save_base_path.replace(
            basename, f"{basename}_{video_type}"
        )
        self.generate_img_and_video(
            decoded_image, None, scene_name, video_save_base_path
        )

    def visulize(
        self,
        box=None,  # 
        anno_box=None,  # 
        collision=None,  # 
        anno_collision=None,  # 
        test_object=0,  # 
        set_index=None,  # 
        scene_name="0",
        maps=None,
        pose=None,
        real_pose=None,
        decoded_image=None,
        view_mask=None,
    ):  # 

        if self.resort_attritube is not None:
            box = self.resort_box(box)
            anno_box = self.resort_box(anno_box)
        if isinstance(box, List):
            box = np.array(box)
        if isinstance(anno_box, List):
            anno_box = np.array(anno_box)
        self.temp_cache_path = os.path.join(
            self.temp_cache_base_path, self.project_name, scene_name
        )  # 
        image_mutil = {}
        images_all = None
        images_all, a_box_count_all, p_box_count_all = self.visulize_objects(
            box,
            anno_box,
            collision,
            anno_collision,
            test_object,
            set_index,
            scene_name,
        )
        if (pose is not None) and self.addtion_ego:
            images_all = self.draw_ego(
                images=images_all, poses=pose.copy(), color_draw=self.ego_color
            )
        if maps is not None:
            if "map" in maps:
                map = maps["map"]
                assert self.map_type in ["token", "point"]
                if self.map_type == "token":
                    images_all = self.draw_map(map, images_all)
                else:
                    images_all = self.draw_point_map(map, images_all)
            if "map_trans" in maps:
                map = maps["map_trans"]
                images_map = self.draw_map(map, None)
                if self.put_text_on_img:
                    images_map = self.put_text(
                        images_map, spe_text="transformed_map"
                    )
                image_mutil["map_trans"] = images_map
            if "map_tokens" in maps:
                map = maps["map_tokens"]
                images_map = self.draw_tokens(map, None)
                if self.put_text_on_img:

                    images_map = self.put_text(
                        images_map, spe_text="map tokens"
                    )
                image_mutil["map_tokens"] = images_map
        if self.put_text_on_img:
            images_all = self.put_text(
                images_all,
                a_box_count_all=a_box_count_all,
                p_box_count_all=p_box_count_all,
                scene_name=scene_name,
                pose=pose,
                real_pose=real_pose,
            )
        image_mutil["images_all"] = images_all
        if decoded_image is not None:
            decoded_image = self.postprocess_image(decoded_image)
            decoded_image = decoded_image.transpose(0, 2, 3, 1)
            image_mutil["decoded_image"] = decoded_image
        cat_imgs = self.concatenate_images(image_mutil, mode="vertical")
        self.generate_img_and_video(cat_imgs, set_index, scene_name)