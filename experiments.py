import math

import json
from collections import OrderedDict

import numpy as np
import torch
import trimesh
from PIL import Image
from torchvision import transforms

from configs.config_utils import CONFIG
from configs.data_config import Relation_Config
import argparse
from dataset.front3d_recon_dataset import Front3D_Recon_Dataset
from dataset.front3d_bg_dataset import FRONT_bg_dataset
from torch.utils.data import DataLoader
from models.instPIFu.InstPIFu_net import InstPIFu
from models.bg_PIFu.BGPIFu_net import BGPIFu_Net
import datetime
import os
import time
import cv2

INSTPIFU_CONFIG_PATH = "./configs/demo_instPIFu.yaml"
OBJ_DET_CONFIG_PATH = "./configs/demo_inference_object_detection.yaml"
INPUT_DIR = "demo/inputs/1"

data_transforms_patch = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transforms_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_name_to_label_mapping = {
    "table": 0,
    "sofa": 1,
    "cabinet": 2,
    "night_stand": 3,
    "chair": 4,
    "bookshelf": 5,
    "bed": 6,
    "desk": 7,
    "dresser": 8
}


def get_instPIFu_model(config):
    model = InstPIFu(config).cuda()

    instPIFu_checkpoints = torch.load(config["weight"])
    instPIFu_net_weight = instPIFu_checkpoints['net']

    instPIFu_new_net_weight = {}
    # instPIFu_new_net_weight = OrderedDict()
    for key in instPIFu_net_weight:
        if key.startswith("module."):
            k_ = key[7:]
            instPIFu_new_net_weight[k_] = instPIFu_net_weight[key]

    model.load_state_dict(instPIFu_new_net_weight)
    model.eval()

    return model


def get_3d_detection_input():
    input_dict = {}
    camera = {}
    boxes = {}

    with open(os.path.join(INPUT_DIR, "detections.json")) as f:
        detections = json.load(f)

    K = np.loadtxt(os.path.join(INPUT_DIR, "cam_K.txt"))
    # XXX: looks like front3d_detect_dataset uses camera matrix for a 2 times bigger image
    camera['K'] = torch.unsqueeze(torch.tensor(K) * 2, 0)
    input_dict['camera'] = camera

    image = Image.open(os.path.join(INPUT_DIR, "img.jpg"))
    width, height = image.width, image.height
    input_dict['image'] = torch.unsqueeze(data_transforms_patch(image), 0)

    bdb2D_poses = []
    cls_codes = []
    box_feats = []
    patches = []
    for i, detection in enumerate(detections):
        if detection['class'] not in class_name_to_label_mapping:
            continue

        bdb2d = detection['bbox']

        # XXX: That multiplication is weird, but front3d_recon_dataset.py seems to use bb coordinates
        # from a 2 times bigger image.
        bdb2D_poses.append(torch.tensor(bdb2d).float() * 2)
        box_feats.append(torch.tensor(
            [2 * (bdb2d[2] - bdb2d[0]) / width, 2 * (bdb2d[3] - bdb2d[1]) / height,
             2 * (bdb2d[2] + bdb2d[0]) / width, 2 * (bdb2d[3] + bdb2d[1]) / height]))

        # XXX: But for croping patches we use the correct size
        patch = image.crop(bdb2d)
        patch = data_transforms_patch(np.asarray(patch) / 255.0).float()
        patches.append(patch)

        cls_code = np.zeros([9])
        cls_code[class_name_to_label_mapping[detection['class']]] = 1
        cls_codes.append(cls_code)

    boxes['bdb2D_pos'] = torch.stack(bdb2D_poses)
    boxes['size_cls'] = torch.tensor(cls_codes)
    boxes['box_feat'] = torch.stack(box_feats)
    boxes['patch'] = torch.stack(patches)

    # g_feature part
    rel_cfg = Relation_Config()
    d_model = int(rel_cfg.d_g / 4)

    n_objects = boxes['bdb2D_pos'].shape[0]
    g_feature = [[((loc2[0] + loc2[2]) / 2. - (loc1[0] + loc1[2]) / 2.) / (loc1[2] - loc1[0]),
                  ((loc2[1] + loc2[3]) / 2. - (loc1[1] + loc1[3]) / 2.) / (loc1[3] - loc1[1]),
                  math.log((loc2[2] - loc2[0]) / (loc1[2] - loc1[0])),
                  math.log((loc2[3] - loc2[1]) / (loc1[3] - loc1[1]))] \
                 for id1, loc1 in enumerate(boxes['bdb2D_pos'])
                 for id2, loc2 in enumerate(boxes['bdb2D_pos'])]

    locs = [num for loc in g_feature for num in loc]

    pe = torch.zeros(len(locs), d_model)
    position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    boxes['g_feature'] = pe.view(n_objects * n_objects, rel_cfg.d_g)

    input_dict['boxes_batch'] = boxes
    input_dict['obj_split'] = torch.tensor([0, len(cls_codes)]).view(1, 2)
    input_dict['sequence_id'] = ["rgb_demo"]

    return input_dict


def get_3d_detection_output(input_data):
    from net_utils.tools import convert_result_no_gt, total3d_todevice_inference_only
    from net_utils.train_test_utils import load_device, get_model, CheckpointIO

    # setup
    cfg = CONFIG(OBJ_DET_CONFIG_PATH)
    cfg.config['mode'] = 'test'
    checkpoint = CheckpointIO(cfg)
    device = load_device(cfg)
    model = get_model(cfg.config, device=device).cuda().float()
    checkpoint.register_modules(net=model)

    # inference
    config = cfg.config
    log_dir = os.path.join(config['other']["model_save_dir"], config['exp_name'])

    os.makedirs(log_dir, exist_ok=True)
    cfg.write_config()

    if config['resume']:
        if isinstance(config['weight'], list):
            for weight_path in config["weight"]:
                checkpoint.load(weight_path)

    with torch.no_grad():
        object_input = total3d_todevice_inference_only(input_data, device)
        est_data, _ = model(object_input)

    K = object_input['K']
    patch_size = object_input['patch'].shape[0]
    obj_split = object_input['split'].long()
    K_array = torch.zeros((patch_size, 3, 3)).to(object_input['patch'].device)
    for idx, (start, end) in enumerate(obj_split.long()):
        K_array[start:end] = K[idx:idx + 1]

    save_dict_list = convert_result_no_gt(object_input, est_data)
    for idx, item in enumerate(save_dict_list):
        item['K'] = K_array[idx].cpu().numpy()

    return save_dict_list


def R_from_yaw_pitch_roll(yaw, pitch, roll):
    Rp = np.zeros((3, 3))
    Ry = np.zeros((3, 3))
    Rr = np.zeros((3, 3))
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)
    Rp[0, 0] = 1
    Rp[1, 1] = cp
    Rp[1, 2] = -sp
    Rp[2, 1] = sp
    Rp[2, 2] = cp

    Ry[0, 0] = cy
    Ry[0, 2] = sy
    Ry[1, 1] = 1
    Ry[2, 0] = -sy
    Ry[2, 2] = cy

    Rr[0, 0] = cr
    Rr[0, 1] = -sr
    Rr[1, 0] = sr
    Rr[1, 1] = cr
    Rr[2, 2] = 1

    R = np.dot(np.dot(Rr, Rp), Ry)
    return R


def get_centroid_from_proj(centroid_depth, proj_centroid, K):
    x_temp = (proj_centroid[0] - K[0, 2]) / K[0, 0]
    y_temp = (proj_centroid[1] - K[1, 2]) / K[1, 1]
    z_temp = 1
    ratio = centroid_depth / np.sqrt(x_temp ** 2 + y_temp ** 2 + z_temp ** 2)
    x_cam = x_temp * ratio
    y_cam = y_temp * ratio
    z_cam = z_temp * ratio
    p = np.stack([x_cam, y_cam, z_cam])
    return p


def get_reconstruction_inputs():
    inputs = []

    with open(os.path.join(INPUT_DIR, "detections.json")) as f:
        detections = json.load(f)

    org_K = np.loadtxt(os.path.join(INPUT_DIR, "cam_K.txt"))
    K = org_K.copy()
    target_f = 584  # 3D FRONT dataset focal lenght in pixels
    f = K[0, 0]
    scale_factor = f / target_f
    K[0] = K[0] / scale_factor
    K[1] = K[1] / scale_factor

    image = Image.open(os.path.join(INPUT_DIR, "img.jpg"))
    width, height = image.width, image.height
    # image.show()
    image_t = image.resize(size=(int(image.width / scale_factor), int(image.height / scale_factor)))
    image_t = torch.unsqueeze(data_transforms_image(image_t).float(), 0)

    i = 0
    for detection in detections:
        if detection['class'] not in class_name_to_label_mapping:
            continue

        input_dict = {}
        bdb2d = detection['bbox']

        input_dict['obj_id'] = [str(i)]

        cls_codes = np.zeros([9])
        cls_codes[class_name_to_label_mapping[detection['class']]] = 1
        input_dict['cls_codes'] = torch.tensor(cls_codes).view(1, 9)

        patch = torch.unsqueeze(data_transforms_patch(np.asarray(image.crop(bdb2d)) / 255.0).float(), 0)
        input_dict['image'] = patch
        input_dict['patch'] = patch
        input_dict['whole_image'] = image_t

        input_dict['K'] = torch.unsqueeze(torch.tensor(K), 0)

        bdb2d_pos = bdb2d / scale_factor
        input_dict['bdb2D_pos'] = torch.unsqueeze(torch.tensor(bdb2d_pos), 0)

        bdb_x = np.linspace(bdb2d[0], bdb2d[2], 64)
        bdb_y = np.linspace(bdb2d[1], bdb2d[3], 64)
        bdb_X, bdb_Y = np.meshgrid(bdb_x, bdb_y)
        bdb_X = (bdb_X - width / 2) / width * 2
        bdb_Y = (bdb_Y - height / 2) / height * 2
        bdb_grid = np.concatenate([bdb_X[:, :, np.newaxis], bdb_Y[:, :, np.newaxis]], axis=-1)
        input_dict['bdb_grid'] = torch.unsqueeze(torch.tensor(bdb_grid), 0)

        detections_3d = get_3d_detection_output(get_3d_detection_input())[0]

        yaw = detections_3d['bboxes'][i]['yaw']
        pitch = detections_3d['layout']['pitch']
        roll = detections_3d['layout']['roll']
        input_dict['rot_matrix'] = torch.tensor([R_from_yaw_pitch_roll(-yaw, pitch, roll)])

        input_dict['bbox_size'] = torch.tensor([detections_3d['bboxes'][i]['size']])

        project_center = detections_3d['bboxes'][i]['project_center']
        centroid_depth = detections_3d['bboxes'][i]['centroid_depth']
        input_dict['obj_cam_center'] = torch.tensor([get_centroid_from_proj(centroid_depth, project_center, org_K)])

        inputs.append(input_dict)

        i += 1

    return inputs


def main():
    instPIFu_config = CONFIG(INSTPIFU_CONFIG_PATH).config

    instPIFu_model = get_instPIFu_model(instPIFu_config)
    input_objects = get_reconstruction_inputs()

    save_folder = os.path.join("outputs", "raw_input_demo")
    os.makedirs(save_folder, exist_ok=True)

    scene = trimesh.Scene()

    '''inference all objects'''
    start_t = time.time()
    for i, object_dict in enumerate(input_objects):
        for key in object_dict:
            if not isinstance(object_dict[key], list):
                object_dict[key] = object_dict[key].float().cuda()

        with torch.no_grad():
            mesh = instPIFu_model.extract_mesh(object_dict, instPIFu_config['data']['marching_cube_resolution'])
            rot_matrix = object_dict["rot_matrix"][0].cpu().numpy()
            obj_cam_center = object_dict["obj_cam_center"][0].cpu().numpy()
            bbox_size = object_dict["bbox_size"][0].cpu().numpy()

            '''transform mesh to camera coordinate'''
            obj_vert = np.asarray(mesh.vertices)
            obj_vert = obj_vert / 2 * bbox_size
            obj_vert = np.dot(obj_vert, rot_matrix.T)
            obj_vert[:, 0:2] = -obj_vert[:, 0:2]
            obj_vert += obj_cam_center
            mesh.vertices = np.asarray(obj_vert.copy())

            object_id = object_dict["obj_id"][0]
            mesh_save_path = os.path.join(save_folder, object_id + ".ply")
            mesh.export(mesh_save_path)
            # mesh.show()
            scene.add_geometry(mesh)

        msg = "{:0>8},[{}/{}]".format(
            str(datetime.timedelta(seconds=round(time.time() - start_t))),
            i + 1,
            len(input_objects),
        )
        print(msg)

    whole_image = object_dict["whole_image"][0].cpu() * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + \
                  torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    whole_image = (whole_image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(save_folder, "input.jpg"), whole_image)

    scene.apply_transform(trimesh.transformations.rotation_matrix(3.14, [1, 0, 0]))
    scene_save_path = os.path.join(save_folder, "full.ply")
    scene.export(scene_save_path)
    scene.show()


if __name__ == "__main__":
    main()
