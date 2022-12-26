from PIL import Image
import numpy as np
import pandas as pd
import random
import os
from torch.utils.data import Dataset, DataLoader
import copy
import torch
from os2d.config import cfg
from os2d.modeling.model import build_os2d_from_config
from os2d.utils import get_image_size_after_resize_preserving_aspect_ratio
from os2d.engine.augmentation import DataAugmentation
from os2d.structures.feature_map import FeatureMapSize
import os2d.structures.transforms as transforms_boxes
import torchvision.transforms as transforms
from os2d.structures.bounding_box import BoxList
from os2d.structures.transforms import TransformList, crop
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def collate_fn(data):
    reference_images, logos, loc_targets, class_targets, class_ids, logo_sizes, \
    box_inverse_transform, boxes, img_size = zip(*data)
    reference_images_th = torch.stack(reference_images, dim=0)
    loc_targets_th = torch.stack(loc_targets, dim=0)
    class_targets_th = torch.stack(class_targets, dim=0)
    return reference_images_th, list(logos), loc_targets_th, class_targets_th, list(class_ids), list(logo_sizes), list(
        box_inverse_transform), list(boxes), list(img_size)


class LITWDataset(Dataset):
    def __init__(self, reference_path, class_path, annotations_path, box_coder, train=True):
        self.reference_path = reference_path
        self.class_path = class_path
        self.annotations_df = pd.read_csv(annotations_path)
        print(self.annotations_df.columns)
        self.box_coder = box_coder
        if not train:
            fm_size = None
        else:
            fm_size = FeatureMapSize(w=600, h=600)
        self.data_augmentation = DataAugmentation(random_flip_batches=False,
                                                  random_crop_size=fm_size,
                                                  random_crop_scale=0.39215686274509803,
                                                  jitter_aspect_ratio=0.9,
                                                  scale_jitter=0.7,
                                                  random_color_distortion=True,
                                                  random_crop_label_images=False,
                                                  min_box_coverage=0.7)

    def __len__(self):
        return len(self.annotations_df["imageid"].unique())

    def _add_colored_background(self, img, bg_color=None, mode="RGBA"):
        assert mode in ["RGB", "RGBA"], "Invalid mode."
        if bg_color is not None:
            assert len(bg_color) == len(mode)
        else:
            bg_color = tuple(np.random.choice(range(256), size=3))
            if mode == "RGBA":
                bg_color = tuple(list(bg_color) + [255])
        colored_bg = Image.new(mode, img.size, bg_color)
        colored_bg.paste(img, mask=img.split()[3])
        return colored_bg

    def __getitem__(self, idx):
        imageid = self.annotations_df["imageid"].unique()[idx]
        idx_df = self.annotations_df[self.annotations_df["imageid"] == imageid]
        reference_image = Image.open(os.path.join(self.reference_path, str(imageid))).convert("RGB")
        names = idx_df["name"].unique()

        class_images = []

        for name in names:
            name = name.strip()
            images = [Image.open(f'{self.class_path}/{name}/{image}').convert("RGBA") \
                      for image in os.listdir(f'{self.class_path}/{name}') if
                      image[-4:] == '.jpg' or image[-4:] == '.png' or image[-5:] == '.jpeg']
            #choice = random.choice(images)
            #class_images.append(choice)
            class_images.extend(images)

        for i in range(len(class_images)):
            class_images[i] = self._add_colored_background(class_images[i])
            class_images[i] = class_images[i].convert("RGB")

        class_ids = idx_df["classid"].tolist()
        bbox = torch.Tensor(np.array(idx_df[['lx', 'ty', 'rx', 'by']]))

        w, h = reference_image.size
        bbox[:, 0] = bbox[:, 0] * w
        bbox[:, 1] = bbox[:, 1] * h
        bbox[:, 2] = bbox[:, 2] * w
        bbox[:, 3] = bbox[:, 3] * h

        # decide on image level data augmentation
        vflip = random.random() < 0.5 if self.data_augmentation.batch_random_vflip else False
        hflip = random.random() < 0.5 if self.data_augmentation.batch_random_hflip else False

        # prepare class images
        num_classes = 1
        class_th = [self._transform_image_gt(class_img, hflip=hflip, vflip=vflip) for class_img in class_images]

        # get the image sizes after resize in self._transform_image_gt, format - width, height
        class_size = [FeatureMapSize(img=class_img) for class_img in class_images]

        # get annotation
        fm_size = FeatureMapSize(img=reference_image)
        boxes = BoxList(bbox, image_size=fm_size, mode="xyxy")
        boxes.add_field("labels", torch.tensor(idx_df["classid"].tolist()))
        boxes.add_field("labels_original", torch.tensor(idx_df["classid"].tolist()))

        # convert global indices to local
        self.update_box_labels_to_local(boxes, idx_df["classid"].tolist())

        # prepare image and boxes: convert image to tensor, data augmentation: some boxes might be cut off the image
        image_mined_data = None

        reference_image_th, boxes, mask_cutoff_boxes, mask_difficult_boxes, box_inverse_transform = \
            self._transform_image(copy.deepcopy(reference_image), boxes, hflip=hflip, vflip=vflip,
                                  mined_data=image_mined_data)

        img_size = FeatureMapSize(img=reference_image_th)

        boxes.get_field("labels")[mask_cutoff_boxes] = -2
        loc_targets, class_targets = self.box_coder.encode(boxes, img_size, num_classes)

        # boxes = torch.Tensor(np.array(idx_df[['lx', 'ty', 'rx', 'by']]))
        return reference_image_th, class_th, loc_targets, class_targets, idx_df[
            "classid"].unique().tolist(), class_size, \
               box_inverse_transform, boxes, img_size, imageid, idx_df["classid"].tolist()

    def _transform_image_gt(self, img, do_augmentation=True, hflip=False, vflip=False, do_resize=True):
        img, _ = transforms_boxes.transpose(img, hflip=hflip, vflip=vflip, boxes=None, transform_list=None)

        if do_augmentation:
            # color distortion
            img = self.data_augmentation.random_distort(img)
            # random crop
            img = self.data_augmentation.random_crop_label_image(img)

        # resize image
        if do_resize:
            try:
                random_interpolation = self.data_augmentation.random_interpolation if do_augmentation else False
            except:
                random_interpolation = False

            # get the new size - while preserving aspect ratio
            size_old = FeatureMapSize(img=img)
            h, w = get_image_size_after_resize_preserving_aspect_ratio(h=size_old.h, w=size_old.w, target_size=240)
            size_new = FeatureMapSize(w=w, h=h)

            img, _ = transforms_boxes.resize(img, target_size=size_new, random_interpolation=random_interpolation)

        transforms_th = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = transforms_th(img)

        return img

    def convert_label_ids_global_to_local(self, label_ids_global, class_ids):
        label_ids_local = []  # local indices w.r.t. batch_class_images
        if label_ids_global is not None:
            for label_id in label_ids_global:
                label_id = label_id.item()
                label_ids_local.append(class_ids.index(label_id) if label_id in class_ids else -1)
        label_ids_local = torch.tensor(label_ids_local, dtype=torch.long)
        return label_ids_local

    def update_box_labels_to_local(self, boxes, class_ids):
        label_ids_global = boxes.get_field("labels")
        label_ids_local = self.convert_label_ids_global_to_local(label_ids_global, class_ids)
        boxes.add_field("labels", label_ids_local)

    def _transform_image(self, img, boxes=None, do_augmentation=True, hflip=False, vflip=False, mined_data=None):
        img_pyramid, boxes_pyramid, mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform = \
            self._transform_image_to_pyramid(img, boxes=boxes,
                                             do_augmentation=do_augmentation, hflip=hflip, vflip=vflip,
                                             pyramid_scales=(1,), mined_data=mined_data)

        return img_pyramid[0], boxes_pyramid[0], mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform[
            0]

    def _transform_image_to_pyramid(self, img, boxes=None,
                                    do_augmentation=True, hflip=False, vflip=False,
                                    pyramid_scales=(1,),
                                    mined_data=None):
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
            if self.data_augmentation.do_random_crop:
                img, boxes, mask_cutoff_boxes, mask_difficult_boxes = \
                    self.data_augmentation.random_crop(img,
                                                       boxes=boxes,
                                                       transform_list=box_inverse_transform)

                img, boxes = transforms_boxes.resize(img, target_size=self.data_augmentation.random_crop_size,
                                                     random_interpolation=self.data_augmentation.random_interpolation,
                                                     boxes=boxes,
                                                     transform_list=box_inverse_transform)

            # color distortion
            img = self.data_augmentation.random_distort(img)

        #random_interpolation = self.data_augmentation.random_interpolation
        random_interpolation = False
        img_size = FeatureMapSize(img=img)
        pyramid_sizes = [FeatureMapSize(w=int(img_size.w * s), h=int(img_size.h * s)) for s in pyramid_scales]
        img_pyramid = []
        boxes_pyramid = []
        pyramid_box_inverse_transform = []
        for p_size in pyramid_sizes:
            box_inverse_transform_this_scale = copy.deepcopy(box_inverse_transform)
            p_img, p_boxes = transforms_boxes.resize(img, target_size=p_size, random_interpolation=random_interpolation,
                                                     boxes=boxes,
                                                     transform_list=box_inverse_transform_this_scale)

            pyramid_box_inverse_transform.append(box_inverse_transform_this_scale)
            img_pyramid.append(p_img)
            boxes_pyramid.append(p_boxes)

        transforms_th = [transforms.ToTensor()]
        transforms_th += [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        for i_p in range(num_pyramid_levels):
            img_pyramid[i_p] = transforms.Compose(transforms_th)(img_pyramid[i_p])
        return img_pyramid, boxes_pyramid, mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform


def os2d_collate_fn(data):
    reference_images, logos, loc_targets, class_targets, class_ids, logo_sizes, \
    box_inverse_transform, boxes, img_size, image_ids, class_labels = zip(*data)
    reference_images_th = torch.stack(reference_images)
    loc_targets_th = torch.stack(loc_targets, dim=0)
    class_targets_th = torch.stack(class_targets, dim=0)
    logos_ls = []
    for logo in list(logos):
        logos_ls.extend(logo)
    #boxes = torch.stack(boxes)
    return reference_images_th, logos_ls, loc_targets_th, class_targets_th, list(class_ids), list(logo_sizes), list(
        box_inverse_transform), list(boxes), list(img_size), list(image_ids), list(class_labels)

