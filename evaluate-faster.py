import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import time

import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms

from os2d.modeling.model import build_os2d_from_config

from os2d.data.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from os2d.engine.train import trainval_loop
from os2d.engine.evaluate import evaluate
from os2d.utils import set_random_seed, get_trainable_parameters, mkdir, save_config, setup_logger, get_data_path, read_image, get_image_size_after_resize_preserving_aspect_ratio
from os2d.engine.optimization import create_optimizer
from os2d.config import cfg
from os2d.structures.feature_map import FeatureMapSize
from os2d.structures.bounding_box import BoxList
import matplotlib.pyplot as plt
import  os2d.utils.visualization as visualizer



def generate_predictions(input_image, class_images):
    h, w = get_image_size_after_resize_preserving_aspect_ratio(h=input_image.size[1],
                                                               w=input_image.size[0],
                                                               target_size=1500)
    transform_image = transforms.Compose([
                      transforms.ToTensor(),
                      #transforms.Resize((2*w, 2*h)),
                      #transforms.CenterCrop(square_size),
                      transforms.Normalize(img_normalization["mean"], img_normalization["std"])
                      ])
    input_image = input_image.resize((w, h))
    square_size = min(w, h)
    input_image_th = transform_image(input_image)

    input_image_th = input_image_th.unsqueeze(0)
    if cfg.is_cuda:
        input_image_th = input_image_th.cuda()

    ## Resize class image
    class_images_th = []
    for class_image in class_images:
        h, w = get_image_size_after_resize_preserving_aspect_ratio(h=class_image.size[1],
                                                                w=class_image.size[0],
                                                                target_size=cfg.model.class_image_size)
        class_image = class_image.resize((w, h))
        square_size = min(w, h)
        transform_image = transforms.Compose([
                      transforms.ToTensor(),
                      #transforms.Resize((3*w, 3*h)),
                      #transforms.CenterCrop(square_size),
                      transforms.Normalize(img_normalization["mean"], img_normalization["std"])
                      ])
        class_image_th = transform_image(class_image)
        if cfg.is_cuda:
            class_image_th = class_image_th.cuda()

        class_images_th.append(class_image_th)


    with torch.no_grad():
        loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(images=input_image_th, class_images=class_images_th)


    image_loc_scores_pyramid = [loc_prediction_batch[0]]
    image_class_scores_pyramid = [class_prediction_batch[0]]
    img_size_pyramid = [FeatureMapSize(img=input_image_th)]
    transform_corners_pyramid = [transform_corners_batch[0]]
    boxes = box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
                                           img_size_pyramid, class_ids,
                                           nms_iou_threshold=cfg.eval.nms_iou_threshold,
                                           nms_score_threshold=cfg.eval.nms_score_threshold,
                                           transform_corners_pyramid=transform_corners_pyramid)
    return boxes

cfg.init.model = "models/os2d_v2-train.pth"
cfg.is_cuda = torch.cuda.is_available()
# set this to use faster convolutions
if cfg.is_cuda:
    assert torch.cuda.is_available(), "Do not have available GPU, but cfg.is_cuda == 1"
    torch.backends.cudnn.benchmark = True

# random seed
set_random_seed(cfg.random_seed, cfg.is_cuda)

# Model
cfg.init.model = "models/os2d_v2-train.pth"
#cfg.model.backbone_arch = 'simclr'

net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)

annspath = '../data/LogoDet-3K_os2d/val-logodet3k/classes/val-annotations.csv'
imgspath = '../data/LogoDet-3K_os2d/val-logodet3k/src/images'
querypath = '../data/LogoDet-3K_os2d/val-logodet3k/classes/images'
save_path = 'detections/logodet3k_detections.pth'

anns = pd.read_csv(annspath)

imageids = np.unique(anns['imageid'])

boxes = []
gt_boxes = []
image_ids = []

t0 = time.time()
for imageid in imageids:
    image_ids.append(imageid)
    imgdf = anns[anns['imageid'] == imageid]
    img = Image.open(f'{imgspath}/{imageid}.jpg').convert("RGB")
    size = FeatureMapSize(img=img)
    gt_box_t = torch.tensor(np.array(anns[['lx','ty','rx','by']]))
    gt_box = BoxList(gt_box_t, size)
    gt_box.add_field('labels', torch.tensor(np.array(imgdf['classid'])))
    
    class_ids = np.unique(imgdf['classid'])
    class_imgs = [Image.open(f'{querypath}/{classid}.jpg').convert("RGB") for classid in class_ids]
    
    boxes.append(generate_predictions(img, class_imgs))
    gt_boxes.append(gt_box)
    tnow = time.time()
    print(imageid, tnow - t0)
    t0 = tnow

boxes_xyxy = []
for box in boxes:
    box_xyxy = box.bbox_xyxy.clone()
    box_xyxy[:,0] = box_xyxy[:,0] / box.image_size.w
    box_xyxy[:,1] = box_xyxy[:,1] / box.image_size.h
    box_xyxy[:,2] = box_xyxy[:,2] / box.image_size.w
    box_xyxy[:,3] = box_xyxy[:,3] / box.image_size.h
    boxes_xyxy.append(box_xyxy)

labels = [box.get_field('labels') for box in boxes]
gt_labels = [box.get_field('labels') for box in gt_boxes]
scores = [box.get_field('scores') for box in boxes]

data = {"image_ids" : image_ids,
        "boxes_xyxy" : boxes_xyxy, 
        "labels" : labels,
        "scores" : scores,
        "gt_boxes_xyxy" : [bb.bbox_xyxy for bb in gt_boxes],
        "gt_labels" : gt_labels
        }
torch.save(data, save_path)