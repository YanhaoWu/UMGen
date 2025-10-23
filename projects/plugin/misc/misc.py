import os
import numba
import numpy as np
import torch
import torch.nn as nn
from numba.cuda.decorators import jit
from torchmetrics import Metric
from tqdm import tqdm

#
catgories_list = ["none", "vehicle", "bicycle", "pedestrian"]

def trans_to_dict(box, cat):
    data = {}
    bbox3d = box
    target_key = (
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

    merge_key = ("bbox_posi", "bbox_whl", "bbox_yaw", "bbox_speed", "bbox_cat")
    for key in merge_key:
        data[key] = []

    for index in range(len(bbox3d)):  #
        data["bbox_posi"].append(
            np.concatenate(
                [
                    bbox3d[index][:, 0].reshape(-1, 1),
                    bbox3d[index][:, 1].reshape(-1, 1),
                    bbox3d[index][:, 2].reshape(-1, 1),
                ],
                axis=-1,
            )
        )
        data["bbox_whl"].append(
            np.concatenate(
                [
                    bbox3d[index][:, 3].reshape(-1, 1),
                    bbox3d[index][:, 4].reshape(-1, 1),
                    bbox3d[index][:, 5].reshape(-1, 1),
                ],
                axis=-1,
            )
        )
        data["bbox_yaw"].append(np.stack(bbox3d[index][:, 6].reshape(-1, 1)))
        data["bbox_speed"].append(
            np.concatenate(
                [
                    bbox3d[index][:, 7].reshape(-1, 1),
                    bbox3d[index][:, 8].reshape(-1, 1),
                    bbox3d[index][:, 9].reshape(-1, 1),
                ],
                axis=-1,
            )
        )

        for i in range(len(cat[index])):
            cat[index][i] = catgories_list.index(cat[index][i])

        data["bbox_cat"].append(np.stack([np.array(cat[index])]))

    for key in merge_key:
        data[key] = np.stack(data[key], axis=0)
    return data



def calculate_box_vertices(centers, whl, rotation_angles, device="cuda"):
    # 计算顶点，输入为中心点，长宽高，角度(角度要乘-1)
    # Extract the dimensions
    lengths, widths, heights = whl[:, 0], whl[:, 1], whl[:, 2]
    # Calculate half of the dimensions
    half_lengths = lengths / 2
    half_widths = widths / 2
    half_heights = heights / 2
    # Create a rotation matrix for each box
    rotation_matrices = torch.stack(
        [
            torch.tensor(
                [
                    [torch.cos(angle), -torch.sin(angle), 0],
                    [torch.sin(angle), torch.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
            for angle in rotation_angles
        ]
    )
    # Define the 8 corners of each box
    corners = torch.stack(
        [
            torch.tensor(
                [
                    [-half_length, -half_width, -half_height],
                    [half_length, -half_width, -half_height],
                    [half_length, half_width, -half_height],
                    [-half_length, half_width, -half_height],
                    [-half_length, -half_width, half_height],
                    [half_length, -half_width, half_height],
                    [half_length, half_width, half_height],
                    [-half_length, half_width, half_height],
                ]
            )
            for half_length, half_width, half_height in zip(
                half_lengths, half_widths, half_heights
            )
        ]
    )
    if device == "cuda":
        corners = corners.cuda()
        rotation_matrices = rotation_matrices.cuda()
    # Rotate the corners for each box
    rotated_corners = torch.einsum("ijk,ikl->ijl", corners, rotation_matrices)
    # Translate the corners to the center coordinates for each box
    translated_corners = rotated_corners + centers.unsqueeze(1)
    return translated_corners


def BoxIoU(boxes1, boxes2, index=None):
    # Assume inputs: boxes1 (M, 8, 3) and boxes2 (N, 8, 3)
    if index is not None:
        boxes1 = boxes1[index]
        boxes2 = boxes2[index]
        if boxes1.ndim == 2:
            boxes1 = boxes1.unsqueeze(0)
        if boxes2.ndim == 2:
            boxes2 = boxes2.unsqueeze(0)
    intersection_vol, iou_3d = box3d_overlap(
        boxes1, boxes2
    )  # box3d_overlap(boxes1, boxes2)
    return intersection_vol, iou_3d


def bbox3d2bevcorners(bboxes):
    # from PointPillars
    """
    bboxes: shape=(n, 7)

                ^ x (-0.5 * pi)
                |
                |                (bird's eye view)
       (-pi)  o |
        y <-------------- (0)
                 \ / (ag)
                  \
                   \

    return: shape=(n, 4, 2)
    """
    centers, dims, angles = bboxes[:, :2], bboxes[:, 3:5], bboxes[:, 6]

    # 1.generate bbox corner coordinates, clockwise from minimal point
    bev_corners = np.array(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=np.float32
    )
    bev_corners = (
        bev_corners[None, ...] * dims[:, None, :]
    )  # (1, 4, 2) * (n, 1, 2) -> (n, 4, 2)

    # 2. rotate
    rot_sin, rot_cos = np.sin(angles), np.cos(angles)
    rot_mat = np.array([[rot_cos, -rot_sin], [rot_sin, rot_cos]])  # (2, 2, n)
    rot_mat = np.transpose(rot_mat, (2, 1, 0))  # (N, 2, 2)
    bev_corners = bev_corners @ rot_mat  # (n, 4, 2)

    # 3. translate to centers
    bev_corners += centers[:, None, :]
    return bev_corners.astype(np.float32)


# from PointPillars
@numba.jit(nopython=True)
def bevcorner2alignedbbox(bev_corners):
    """
    bev_corners: shape=(N, 4, 2)
    return: shape=(N, 4)
    """
    # xmin, xmax = np.min(bev_corners[:, :, 0], axis=-1), np.max(bev_corners[:, :, 0], axis=-1)
    # ymin, ymax = np.min(bev_corners[:, :, 1], axis=-1), np.max(bev_corners[:, :, 1], axis=-1)

    # why we don't implement like the above ? please see
    # https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html#calculation
    n = len(bev_corners)
    alignedbbox = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        cur_bev = bev_corners[i]
        alignedbbox[i, 0] = np.min(cur_bev[:, 0])
        alignedbbox[i, 2] = np.max(cur_bev[:, 0])
        alignedbbox[i, 1] = np.min(cur_bev[:, 1])
        alignedbbox[i, 3] = np.max(cur_bev[:, 1])
    return alignedbbox


@numba.jit(nopython=True)  #
def box_collision_test(boxes, qboxes, clockwise=True):
    """Box collision test.
    Args:
        boxes (np.ndarray): Corners of current boxes. # (n1, 4, 2)
        qboxes (np.ndarray): Boxes to be avoid colliding. # (n2, 4, 2)
        clockwise (bool, optional): Whether the corners are in
            clockwise order. Default: True.
    return: shape=(n1, n2)
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    collision_list = []
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2
    )  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = bevcorner2alignedbbox(boxes)
    qboxes_standup = bevcorner2alignedbbox(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]
            )
            if iw > 0:
                ih = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]
                )
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (
                                C[1] - A[1]
                            ) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (
                                C[1] - B[1]
                            ) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]
                                ) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]
                                ) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] == True:
                            collision_list.append([i, j])
                            break
                    if (
                        ret[i, j] is False
                    ):  # is False 
                        # now check complete overlap.
                        # box overlap qbox:
                        # print("complete check")
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, box_l, 0]
                                )
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, box_l, 1]
                                )
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, box_l, 0]
                                    )
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, box_l, 1]
                                    )
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                                collision_list.append([i, j])

                        else:
                            ret[i, j] = True  # collision.
                            collision_list.append([i, j])

    return ret, collision_list


def generate_collsion_attribute(
    bbox3d,
    time_steps=20,
    iou_threshold=0,
    sampling_gap=1,
    speed_scale=1,
    device="cuda",
    mode="2d",
    stop_speed=0.05,
    box_scale_list=[1],
):
    test_index = 1
    predict_time_step = time_steps  
    collision_attribute_scale = []
    for box_scale_list in box_scale_list:
        collision_attribute = []
        for i in range(len(bbox3d)):
            bbox3d_i = bbox3d[i]
            speed_x = bbox3d_i[:, 7]
            speed_y = bbox3d_i[:, 8]

            length = bbox3d_i[:, 3] * box_scale_list
            width = bbox3d_i[:, 4] * box_scale_list
            height = bbox3d_i[:, 5] * box_scale_list
            yaw = bbox3d_i[:, 6]
            x_posi_list = []
            y_posi_list = []
            time_steps = np.arange(1, predict_time_step + 1).reshape(
                -1, 1
            )  # 1 ~ time_steps
            stop_index = np.where(
                (np.abs(speed_x) <= stop_speed) & (speed_y <= stop_speed)
            )[
                0
            ]
            # 
            small_index = np.where((length <= 1) & (width <= 1))[0]
            speed_x[np.abs(speed_x) <= stop_speed] = 0
            speed_y[np.abs(speed_y) <= stop_speed] = 0
            x_posi = (
                bbox3d_i[:, 0].reshape(1, -1)
                + speed_x.reshape(1, -1)
                * time_steps
                * sampling_gap
                * speed_scale
            )
            y_posi = (
                bbox3d_i[:, 1].reshape(1, -1)
                + speed_y.reshape(1, -1)
                * time_steps
                * sampling_gap
                * speed_scale
            )
            z_posi = (
                bbox3d_i[:, 2].reshape(1, -1) + 0 * time_steps * sampling_gap
            )

            z_posi = z_posi - z_posi + 1  

            collision_list = []
            collision_i_attitube = []
            # 
            for j in range(len(time_steps)):
                num_list = np.zeros(
                    (bbox3d_i.shape[0], 1), dtype=np.int32
                ).flatten() + len(time_steps)

                if mode == "3d":
                    d_centers = np.stack(
                        [x_posi[j], y_posi[j], z_posi[j]], axis=1
                    )
                    d_lwh = bbox3d_i[:, 3:6]  # 
                    d_rotation_angles = -bbox3d_i[:, 6]  # 
                    d_lwh[:, -1] = 1  # 
                    if not isinstance(d_centers, torch.Tensor):
                        d_centers = torch.tensor(
                            d_centers, dtype=torch.float32, device=device
                        )
                        d_lwh = torch.tensor(
                            d_lwh, dtype=torch.float32, device=device
                        )
                        d_rotation_angles = torch.tensor(
                            d_rotation_angles,
                            dtype=torch.float32,
                            device=device,
                        )

                    vertices = calculate_box_vertices(
                        d_centers, d_lwh, d_rotation_angles, device=device
                    )  # 
                    intersection_vol, iou_3d = BoxIoU(vertices, vertices)

                    # 
                    for k in range(iou_3d.shape[0]):
                        iou_3d[k, k] = 0

                    # 
                    collision = (
                        torch.sum(iou_3d > 0, dim=1) > iou_threshold
                    )  # 
                    collision_list.append(collision)

                    num_list[collision.cpu().numpy()] = j
                    collision_i_attitube.append(num_list)
                    # print("warning, in mode 3d, we do not fliter the stop objects when condiering the collision attribute.")

                else:
                    bbox3d_nospeed = np.concatenate(
                        [
                            x_posi[j].reshape(-1, 1),
                            y_posi[j].reshape(-1, 1),
                            z_posi[j].reshape(-1, 1),
                            length.reshape(-1, 1),
                            width.reshape(-1, 1),
                            height.reshape(-1, 1),
                            -yaw.reshape(-1, 1),
                        ],
                        axis=1,
                    )
                    bbox2d = bbox3d2bevcorners(bbox3d_nospeed)
                    collision, collision_index = box_collision_test(
                        bbox2d, bbox2d
                    )  
                    collision = np.sum(collision == True, axis=1) > 0

                    collision_index = np.array(collision_index)
                    if collision_index.shape[0] > 0:
                        unique_query = np.unique(collision_index[:, 0])
                        for query in unique_query:
                            collision_object = collision_index[
                                collision_index[:, 0] == query
                            ]
                            if (
                                np.all(np.isin(collision_object, stop_index))
                                and query in stop_index
                            ):
                                collision[query] = False
                            if (
                                np.any(np.isin(collision_object, small_index))
                                and query in small_index
                            ):
                                collision[query] = False

                    num_list[collision] = j
                    collision_i_attitube.append(num_list)

                # print("collision", collision)
            collision_i_attitube = np.stack(
                collision_i_attitube
            ).transpose()  
            collision_i_attitube = np.min(collision_i_attitube, axis=1)

            collision_attribute.append(collision_i_attitube)
        collision_attribute_scale.append(collision_attribute)

    if len(collision_attribute_scale) == 1:
        collision_attribute_scale = collision_attribute_scale[0]

    return collision_attribute_scale


def fliter_and_map_object(box):
    non_empty_index = []
    for i in range(len(box)):
        bbox_i = box[i]
        if not bbox_i[0] >= 63:  # 
            non_empty_index.append(i)
    return box[non_empty_index], non_empty_index


class MMD_loss:
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.scenario_count = 0
        self.mmd_score = []
        self.count_scenario = 0

    def reset(self):
        self.mmd_score = []
        self.count_scenario = 0

    def add(self, score):

        self.mmd_score.append(score)

    def average(self):

        average_score = np.mean(self.mmd_score)
        return average_score

    def guassian_kernel(
        self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None
    ):
        n_samples_source = int(source.size()[0])
        n_samples_target = int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (
                n_samples_source * n_samples_target
            )
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [
            bandwidth * (kernel_mul ** i) for i in range(kernel_num)
        ]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp)
            for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    def update(self, source, target):
        # source 放 real data, target 
        n_samples_source = int(source.size()[0])
        n_samples_target = int(target.size()[0])
        kernels = self.guassian_kernel(
            source,
            target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num,
            fix_sigma=self.fix_sigma,
        )
        XX = kernels[:n_samples_source, :n_samples_source]
        YY = kernels[n_samples_source:, n_samples_source:]
        XY = kernels[:n_samples_source, n_samples_source:]
        YX = kernels[n_samples_source:, :n_samples_source]
        mmd_score = (
            torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        )

        self.add(mmd_score)
        self.count_scenario += 1
        # return mmd_score


class BoxOverlap:
    def __init__(self, scale=1, ped2ped=True):
        super(BoxOverlap, self).__init__()

        self.ratio_all = []
        self.ratio_scenario_all = []

        self.count_scenario = 0

        self.scale = scale
        self.ped2ped = ped2ped  # 

    def reset(self):

        self.ratio_all = []
        self.ratio_scenario_all = []
        self.count_scenario = 0

    def add(self, ratio, ratio_scenario):

        self.ratio_all.append(ratio)
        self.ratio_scenario_all.append(ratio_scenario)

    def average(self):

        ratio = np.mean(self.ratio_all)
        ratio_scenario = np.mean(self.ratio_scenario_all)

        return ratio, ratio_scenario

    def check_collision(self, box, fliter=False):
        if len(box) == 1:
            return False
        else:
            bbox3d = np.array(box)

            if fliter:
                bbox3d, non_empty_index = fliter_and_map_object(bbox3d)

            if bbox3d.shape[0] <= 1:
                return False
            x_posi = bbox3d[:, 0].reshape(1, -1)
            y_posi = bbox3d[:, 1].reshape(1, -1)
            z_posi = bbox3d[:, 2].reshape(1, -1)

            length = bbox3d[:, 3]
            width = bbox3d[:, 4]
            height = bbox3d[:, 5]
            yaw = -bbox3d[:, 6]

            bbox3d_nospeed = np.concatenate(
                [
                    x_posi.reshape(-1, 1),
                    y_posi.reshape(-1, 1),
                    z_posi.reshape(-1, 1),
                    length.reshape(-1, 1),
                    width.reshape(-1, 1),
                    height.reshape(-1, 1),
                    yaw.reshape(-1, 1),
                ],
                axis=1,
            )
            bbox2d = bbox3d2bevcorners(bbox3d_nospeed)
            collision_mat, collision_index = box_collision_test(
                boxes=bbox2d, qboxes=bbox2d[-1:]
            )
            if np.any(collision_mat[:, 0]):
                return True
            else:
                return False

    def find_ped(self, bboxes):
        new_bboxes = []

        for i in range(len(bboxes)):
            bbox_i = bboxes[i]
            if (bbox_i[3] < 2) and (bbox_i[4] < 1.5):  # 
                new_bboxes.append(i)

        return new_bboxes

    def compute_overlap_count(
        self, box, fliter=False, return_collision_box_id=False
    ):
        self.count_scenario += 1

        ratio_all = []  # average perscene
        total_num = []
        total_collision = []  # average scenario

        collision_id_all = []

        for i in range(0, len(box)):
            if isinstance(box[i], np.ndarray):
                bbox3d = box[i].tolist()
            else:
                bbox3d = box[i]
            if len(bbox3d) == 0:
                self.add(ratio=0, ratio_scenario=0)
                collision_id_all.append([])
                continue

            # if bbox3d.dtype == 'object':
            bbox3d = np.stack([np.array(bbox) for bbox in bbox3d])

            if fliter:
                bbox3d, non_empty_index = fliter_and_map_object(bbox3d)

            if not self.ped2ped:
                ped_id = self.find_ped(bbox3d)

            x_posi = bbox3d[:, 0].reshape(1, -1)
            y_posi = bbox3d[:, 1].reshape(1, -1)
            z_posi = bbox3d[:, 2].reshape(1, -1)

            length = bbox3d[:, 3] * self.scale
            width = bbox3d[:, 4] * self.scale
            height = bbox3d[:, 5]
            # yaw = bbox3d[:, 6] #  yaw = -bbox3d[:, 6]
            yaw = bbox3d[:, 6]

            bbox3d_nospeed = np.concatenate(
                [
                    x_posi.reshape(-1, 1),
                    y_posi.reshape(-1, 1),
                    z_posi.reshape(-1, 1),
                    length.reshape(-1, 1),
                    width.reshape(-1, 1),
                    height.reshape(-1, 1),
                    yaw.reshape(-1, 1),
                ],
                axis=1,
            )
            bbox2d = bbox3d2bevcorners(bbox3d_nospeed)
            collision_mat, collision_index = box_collision_test(bbox2d, bbox2d)
            if not self.ped2ped:
                new_collision_index = []
                for ped_test_i in range(len(collision_index)):
                    agent_a = collision_index[ped_test_i][0]
                    agent_b = collision_index[ped_test_i][1]
                    if not (agent_a in ped_id and agent_b in ped_id):
                        new_collision_index.append(collision_index[ped_test_i])
                    else:
                        assert collision_mat[agent_a, agent_b] == True
                        collision_mat[agent_a, agent_b] = False
                collision_index = new_collision_index

            if len(collision_index) != 0:
                collision_id = np.unique(np.stack(collision_index))
                if fliter:
                    collision_id = np.array(
                        [non_empty_index[i] for i in collision_id]
                    )
                collision_id_all.append(collision_id)
            else:
                collision_id_all.append([])

            collision = np.sum(collision_mat == True, axis=1) > 0
            num_collision = np.sum(collision)

            ratio = num_collision / len(collision)
            ratio_all.append(ratio)

            total_num.append(len(bbox3d))
            total_collision.append(num_collision)

        ratio = np.mean(ratio_all)

        ratio_scenario = np.sum(total_collision) / np.sum(total_num)
        if not np.isnan(ratio_scenario) and not np.isnan(ratio):
            self.add(ratio=ratio, ratio_scenario=ratio_scenario)
        else:
            print("Warning, the ratio is nan")
            self.add(ratio=0, ratio_scenario=0)
        if return_collision_box_id:
            return collision_id_all

