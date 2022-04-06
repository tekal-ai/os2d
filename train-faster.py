import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import time
from collections import OrderedDict
from os2d.structures.transforms import TransformList, crop
import copy
import math
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms

from os2d.modeling.model import build_os2d_from_config

from os2d.data.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from os2d.engine.train import trainval_loop
from os2d.engine.evaluate import evaluate
from os2d.utils import checkpoint_model, set_random_seed, add_to_meters_in_dict, print_meters, get_trainable_parameters, mkdir, save_config, setup_logger, get_data_path, read_image, get_image_size_after_resize_preserving_aspect_ratio
from os2d.engine.optimization import create_optimizer
from os2d.config import cfg
from os2d.structures.feature_map import FeatureMapSize
from os2d.structures.bounding_box import BoxList
from os2d.engine.augmentation import DataAugmentation
import matplotlib.pyplot as plt
import  os2d.utils.visualization as visualizer
import os2d.structures.transforms as transforms_boxes
from os2d.engine.optimization import setup_lr, get_learning_rate, set_learning_rate

imgspath = '../data/val-logodet3k/src/images'
querypath = '../data/val-logodet3k/classes/images'
annspath = '../data/val-logodet3k/classes/val-annotations.csv'

train_df = pd.read_csv(annspath)

cfg.init.model = "models/os2d_v2-train.pth"
cfg.is_cuda = torch.cuda.is_available()
cfg.train.batch_size = 1
# set this to use faster convolutions
if cfg.is_cuda:
    assert torch.cuda.is_available(), "Do not have available GPU, but cfg.is_cuda == 1"
    torch.backends.cudnn.benchmark = True

# random seed
set_random_seed(cfg.random_seed, cfg.is_cuda)

# Model
cfg.init.model = "models/os2d_v2-train.pth"

net, box_coder, criterion, img_normalization, optimizer_state = build_os2d_from_config(cfg)
transform_image = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize(img_normalization["mean"], img_normalization["std"])
                  ])

parameters = get_trainable_parameters(net)
optimizer = create_optimizer(parameters, cfg.train.optim, optimizer_state)

data_augmentation = DataAugmentation(random_flip_batches=False,
                                      random_crop_size=FeatureMapSize(w=600, h=600),
                                      random_crop_scale=0.39215686274509803,
                                      jitter_aspect_ratio=0.9,
                                      scale_jitter=0.7,
                                      random_color_distortion=True,
                                      random_crop_label_images=False,
                                      min_box_coverage=0.7)

cfg.output.save_iter = 30000
cfg.output.path = 'trained-models'
cfg.eval.iter = 5000
cfg.train.optim.max_iter = len(train_df)

def get_batch(i_batch):
    idxs = [np.unique(train_df['imageid'])[i_batch]]
    batch_data = _prepare_batch(idxs)
    return batch_data

def trainval_loop2():
    # setup the learning rate schedule
    _, anneal_lr_func = setup_lr(optimizer, None, cfg.train.optim.anneal_lr, cfg.eval.iter)

    # save initial model
    if cfg.output.path:
        checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, i_iter=0)

    # start training
    i_epoch = 0
    i_batch = 0  # to start a new epoch at the first iteration
    for i_iter in range(cfg.train.optim.max_iter):
        if i_iter % 25 == 0: 
            print(i_iter)
        # restart dataloader if needed
        if i_batch >= len(train_df) // cfg.train.batch_size:
            i_epoch += 1
            i_batch = 0
            # shuffle dataset
            dataloader_train.shuffle()

        # get data for training
        t_start_loading = time.time()
        batch_data = get_batch(i_batch)
        
        t_data_loading = time.time() - t_start_loading

        i_batch += 1

        # train on one batch
        meters = train_one_batch(batch_data, net, cfg, criterion, optimizer)
        meters["loading_time"] = t_data_loading

        # save intermediate model
        if cfg.output.path and cfg.output.save_iter and i_iter % cfg.output.save_iter == 0:
            print("Saving...")
            checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, i_iter=i_iter)

    # save the final model
    if cfg.output.path:
        checkpoint_model(net, optimizer, cfg.output.path, cfg.is_cuda, i_iter=cfg.train.optim.max_iter)

def prepare_batch_data(batch_data, is_cuda):
    """Helper function to parse batch_data and put tensors on a GPU.
    Used in train_one_batch
    """
    images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
        batch_box_inverse_transform, batch_boxes, batch_img_size = \
        batch_data
    if is_cuda:
        images = images.cuda()
        class_images = [im.cuda() for im in class_images]
        loc_targets = loc_targets.cuda()
        class_targets = class_targets.cuda()

    return images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
        batch_box_inverse_transform, batch_boxes, batch_img_size

def train_one_batch(batch_data, net, cfg, criterion, optimizer):
    t_start_batch = time.time()

    net.train(freeze_bn_in_extractor=cfg.train.model.freeze_bn,
              freeze_transform_params=cfg.train.model.freeze_transform,
              freeze_bn_transform=cfg.train.model.freeze_bn_transform)
    
    optimizer.zero_grad()
    
    images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
        batch_box_inverse_transform, batch_boxes, batch_img_size  = \
            prepare_batch_data(batch_data, cfg.is_cuda)
    
    images, class_images, loc_targets, class_targets, class_ids, class_image_sizes, \
        batch_box_inverse_transform, batch_boxes, batch_img_size = batch_data
    
    
    loc_scores, class_scores, class_scores_transform_detached, fm_sizes, corners = \
        net(images, class_images,
            train_mode=True,
            fine_tune_features=cfg.train.model.train_features)
    
    cls_targets_remapped, ious_anchor, ious_anchor_corrected = \
        box_coder.remap_anchor_targets(loc_scores, batch_img_size, class_image_sizes, batch_boxes)
    
    losses = criterion(loc_scores, loc_targets,
                       class_scores, class_targets,
                       cls_targets_remapped=cls_targets_remapped,
                       cls_preds_for_neg=class_scores_transform_detached if not cfg.train.model.train_transform_on_negs else None)
    
    main_loss = losses["loss"]
    main_loss.backward()
    
    
    # save full grad
    grad = OrderedDict()
    for name, param in net.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad[name] = param.grad.clone().cpu()


    grad_norm = torch.nn.utils.clip_grad_norm_(get_trainable_parameters(net), cfg.train.optim.max_grad_norm, norm_type=2)
    # save error state if grad appears to be nan
    if math.isnan(grad_norm):
        # remove some unsavable objects
        batch_data = [b for b in batch_data]
        batch_data[6] = None

        data_nan = {"batch_data":batch_data, "state_dict":net.state_dict(), "optimizer": optimizer.state_dict(),
                    "cfg":cfg,  "grad": grad}
        time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
        dump_file = "error_nan_appeared-"+time_stamp+".pth"
        if cfg.output.path:
            dump_file = os.path.join(cfg.output.path, dump_file)

        print("gradient is NaN. Saving dump to {}".format(dump_file))
        torch.save(data_nan, dump_file)
    else:
        optimizer.step()

    # convert everything to numbers
    meters = OrderedDict()
    for l in losses:
        meters[l] = losses[l].mean().item()
    meters["grad_norm"] = grad_norm
    meters["batch_time"] = time.time() - t_start_batch
    return meters

def get_class_images_and_sizes(class_ids):
    class_images = [Image.open(f'{querypath}/{class_id}.jpg') for class_id in class_ids]
    class_image_sizes = [FeatureMapSize(img=img) for img in class_images]
    return class_images, class_image_sizes


def _transform_image_gt(img, do_augmentation=True, hflip=False, vflip=False, do_resize=True):

    # batch level data augmentation
    img, _ = transforms_boxes.transpose(img, hflip=hflip, vflip=vflip, boxes=None, transform_list=None)

    if do_augmentation:
        # color distortion
        img = data_augmentation.random_distort(img)
        # random crop
        img = data_augmentation.random_crop_label_image(img)

    # resize image
    if do_resize:
        random_interpolation = data_augmentation.random_interpolation if do_augmentation else False

        # get the new size - while preserving aspect ratio
        size_old = FeatureMapSize(img=img)
        h, w = get_image_size_after_resize_preserving_aspect_ratio(h=size_old.h, w=size_old.w,
            target_size=240)
        size_new = FeatureMapSize(w=w, h=h)

        img, _  = transforms_boxes.resize(img, target_size=size_new, random_interpolation=random_interpolation)

    transforms_th = [transforms.ToTensor()]
    if img_normalization is not None:
        transforms_th += [transforms.Normalize(img_normalization["mean"], img_normalization["std"])]
    img = transforms.Compose(transforms_th)(img)
    return img

def get_boxes_from_image_dataframe(image_data, image_size):
    if not image_data.empty:
        # get the labels
        label_ids_global = torch.tensor(list(image_data["classid"]), dtype=torch.long)

        # get the boxes
        boxes = image_data[["lx", "ty", "rx", "by"]].to_numpy()
        # renorm boxes using the image size
        boxes[:, 0] *= image_size.w
        boxes[:, 2] *= image_size.w
        boxes[:, 1] *= image_size.h
        boxes[:, 3] *= image_size.h
        boxes = torch.FloatTensor(boxes)

        boxes = BoxList(boxes, image_size=image_size, mode="xyxy")
    else:
        boxes = BoxList.create_empty(image_size)
        label_ids_global = torch.tensor([], dtype=torch.long)
        difficult_flag = torch.tensor([], dtype=torch.bool)

    boxes.add_field("labels", label_ids_global)
    boxes.add_field("labels_original", label_ids_global)
    return boxes

def convert_label_ids_global_to_local(label_ids_global, class_ids):
    label_ids_local = [] # local indices w.r.t. batch_class_images
    if label_ids_global is not None:
        for label_id in label_ids_global:
            label_id = label_id.item()
            label_ids_local.append( class_ids.index(label_id) if label_id in class_ids else -1 )
    label_ids_local = torch.tensor(label_ids_local, dtype=torch.long)
    return label_ids_local


def update_box_labels_to_local(boxes, class_ids):
    label_ids_global = boxes.get_field("labels")
    label_ids_local = convert_label_ids_global_to_local(label_ids_global, class_ids)
    boxes.add_field("labels", label_ids_local)

def _transform_image_to_pyramid(image_id, boxes=None,
                                      do_augmentation=True, hflip=False, vflip=False,
                                      pyramid_scales=(1,),
                                      mined_data=None ):
    img = Image.open(f'{imgspath}/{image_id}.jpg')
    img_size = FeatureMapSize(img=img)

    num_pyramid_levels = len(pyramid_scales)

    if boxes is None:
        boxes = BoxList.create_empty(img_size)
    mask_cutoff_boxes = torch.zeros(len(boxes), dtype=torch.bool)
    mask_difficult_boxes = torch.zeros(len(boxes), dtype=torch.bool)

    box_inverse_transform = TransformList()
    # batch level data augmentation
    img, boxes = transforms_boxes.transpose(img, hflip=hflip, vflip=vflip, 
                                            boxes=boxes,
                                            transform_list=box_inverse_transform)


    if do_augmentation:        
        if data_augmentation.do_random_crop:
            img, boxes, mask_cutoff_boxes, mask_difficult_boxes = \
                        data_augmentation.random_crop(img,
                                                           boxes=boxes,
                                                           transform_list=box_inverse_transform)

            img, boxes = transforms_boxes.resize(img, target_size=data_augmentation.random_crop_size,
                                                 random_interpolation=data_augmentation.random_interpolation,
                                                 boxes=boxes,
                                                 transform_list=box_inverse_transform)

        # color distortion
        img = data_augmentation.random_distort(img)

    random_interpolation = data_augmentation.random_interpolation
    img_size = FeatureMapSize(img=img)
    pyramid_sizes = [ FeatureMapSize(w=int(img_size.w * s), h=int(img_size.h * s)) for s in pyramid_scales ]
    img_pyramid = []
    boxes_pyramid = []
    pyramid_box_inverse_transform = []
    for p_size in pyramid_sizes:
        box_inverse_transform_this_scale = copy.deepcopy(box_inverse_transform)
        p_img, p_boxes = transforms_boxes.resize(img, target_size=p_size, random_interpolation=random_interpolation,
                                                 boxes=boxes,
                                                 transform_list=box_inverse_transform_this_scale)

        pyramid_box_inverse_transform.append(box_inverse_transform_this_scale)
        img_pyramid.append( p_img )
        boxes_pyramid.append( p_boxes )

    transforms_th = [transforms.ToTensor()]
    if img_normalization is not None:
        transforms_th += [transforms.Normalize(img_normalization["mean"], img_normalization["std"])]

    for i_p in range(num_pyramid_levels):
        img_pyramid[i_p] = transforms.Compose(transforms_th)( img_pyramid[i_p] )

    return img_pyramid, boxes_pyramid, mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform

    
    
def _transform_image(image_id, boxes=None, do_augmentation=True, hflip=False, vflip=False, mined_data=None):
    img_pyramid, boxes_pyramid, mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform = \
            _transform_image_to_pyramid(image_id, boxes=boxes,
                                             do_augmentation=do_augmentation, hflip=hflip, vflip=vflip,
                                             pyramid_scales=(1,), mined_data=mined_data) 

    return img_pyramid[0], boxes_pyramid[0], mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform[0]

    
def _prepare_batch(image_ids, use_all_labels=False):
    print(image_ids)
    batch_images = []
    batch_class_images = []
    batch_loc_targets = []
    batch_class_targets = []

    # flag to use hard neg mining
    use_mined_data = False

    # collect labels for this batch
    batch_data = train_df[train_df['imageid'].isin(image_ids)]

    class_ids = batch_data["classid"].unique()
    # select labels for mined hardnegs
    mined_labels = []

    # randomly prune label images if too many
    max_batch_labels = class_ids.size + len(mined_labels) + 1

    class_ids = np.unique(class_ids)
    np.random.shuffle(class_ids)
    class_ids = class_ids[:max_batch_labels - len(mined_labels)]

    class_ids = np.unique(np.concatenate((class_ids, np.array(mined_labels).astype(class_ids.dtype)), axis=0))
    class_ids = sorted(list(class_ids))

    # decide on batch level data augmentation
    batch_vflip = random.random() < 0.5 if data_augmentation.batch_random_vflip else False
    batch_hflip = random.random() < 0.5 if data_augmentation.batch_random_hflip else False

    # prepare class images
    num_classes = len(class_ids)
    class_images, class_image_sizes = get_class_images_and_sizes(class_ids)
    batch_class_images = [_transform_image_gt(img, hflip=batch_hflip, vflip=batch_vflip) for img in class_images]
    # get the image sizes after resize in self._transform_image_gt, format - width, height
    class_image_sizes = [FeatureMapSize(img=img) for img in batch_class_images]

    # prepare images and boxes
    img_size = None
    batch_box_inverse_transform = []
    batch_boxes = []
    batch_img_size = []
    for image_id in image_ids:
        # get annotation
        fm_size = FeatureMapSize(Image.open(f'{imgspath}/{image_id}.jpg'))
        boxes = get_boxes_from_image_dataframe(batch_data[batch_data['imageid']  == image_id], fm_size)

        # convert global indices to local
        # if use_global_labels==False then local indices will be w.r.t. labels in this batch
        # if use_global_labels==True then local indices will be w.r.t. labels in the whole dataset (not class_ids)
        update_box_labels_to_local(boxes, class_ids)

        # prepare image and boxes: convert image to tensor, data augmentation: some boxes might be cut off the image
        image_mined_data = None 
        img, boxes, mask_cutoff_boxes, mask_difficult_boxes, box_inverse_transform = \
                 _transform_image(image_id, boxes, hflip=batch_hflip, vflip=batch_vflip, mined_data=image_mined_data)

        # mask_difficult_boxes is set True for boxes that are largely chopped off, those are not used for training
        if boxes.has_field("difficult"):
            old_difficult = boxes.get_field("difficult")
            boxes.add_field("difficult", old_difficult | mask_difficult_boxes)
        boxes.get_field("labels")[mask_cutoff_boxes] = -2

        # check image size in this batch
        if img_size is None:
            img_size = FeatureMapSize(img=img)
        else:
            assert img_size == FeatureMapSize(img=img), "Images in a batch should be of the same size"

        loc_targets, class_targets = box_coder.encode(boxes, img_size, num_classes)
        batch_loc_targets.append(loc_targets)
        batch_class_targets.append(class_targets)
        batch_images.append(img)
        batch_box_inverse_transform.append( [box_inverse_transform] )
        batch_boxes.append(boxes)
        batch_img_size.append(img_size)

    # stack data
    batch_images = torch.stack(batch_images, 0)
    batch_loc_targets = torch.stack(batch_loc_targets, 0)
    batch_class_targets = torch.stack(batch_class_targets, 0)

    return batch_images, batch_class_images, batch_loc_targets, batch_class_targets, class_ids, class_image_sizes, \
           batch_box_inverse_transform, batch_boxes, batch_img_size


trainval_loop2()
