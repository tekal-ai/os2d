from PIL import Image, ImageDraw
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
import copy
import sys
import torch
from os2d.config import cfg
from os2d.modeling.model import build_os2d_from_config
from os2d.utils import checkpoint_model, set_random_seed, add_to_meters_in_dict, print_meters, get_trainable_parameters, mkdir, save_config, setup_logger, get_data_path, read_image, get_image_size_after_resize_preserving_aspect_ratio
from os2d.engine.augmentation import DataAugmentation
from os2d.structures.feature_map import FeatureMapSize
import os2d.structures.transforms as transforms_boxes
import torchvision.transforms as transforms
from os2d.structures.bounding_box import BoxList
from os2d.structures.transforms import TransformList, crop


class SyntheticAugmentationsDataset(Dataset):
    def __init__(self, reference_path, class_path, box_coder):
        self.reference_path = reference_path
        self.class_path = class_path
        self.reference_images = os.listdir(reference_path)
        self.reference_images.sort()
        self.class_images = os.listdir(class_path)
        self.class_images.sort()
        self.box_coder = box_coder
        self.data_augmentation = DataAugmentation(random_flip_batches=False,
                                      random_crop_size=FeatureMapSize(w=600, h=600),
                                      random_crop_scale=0.39215686274509803,
                                      jitter_aspect_ratio=0.9,
                                      scale_jitter=0.7,
                                      random_color_distortion=True,
                                      random_crop_label_images=False,
                                      min_box_coverage=0.7) 
        self.class_idx = 0
        
    
    def __len__(self):
        return len(self.reference_images) #* len(self.class_images)
    
    def _get_image(self, image_idx, image_type):
        assert image_type.lower() in ["reference", "class"], "Invalid image type."
        if image_type == "reference":
            return Image.open(os.path.join(self.reference_path, self.reference_images[image_idx])).convert("RGBA")
        elif image_type == "class":
            return Image.open(os.path.join(self.class_path, self.class_images[image_idx])).convert("RGBA")
        
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
    
    def _sample_object_placement(self, img_size, object_size, max_occlusion=0.25):
        x = int(random.uniform(-object_size[0] * max_occlusion, img_size[0] - (1 - max_occlusion) * object_size[1]))
        y = int(random.uniform(-object_size[1] * max_occlusion, img_size[1] - (1 - max_occlusion) * object_size[1]))
        return x, y
    
    def _sample_angle(self, min_angle=-45, max_angle=45, p=0.5):
        if random.uniform(0, 1) > 0.5:
            angle = random.uniform(-45, 45)
        else:
            angle = 0.0
        return angle
    
    def _random_distort(self, img):
        if random.uniform(0, 1) > 0.75:
            img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
        if random.uniform(0, 1) > 0.5:
            factor = random.uniform(0.75, 1.25)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        if random.uniform(0, 1) > 0.5:
            factor = random.uniform(0.0, 2.0)
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(factor)
        return img

    def __getitem__(self, idx):
        # class image modifications to add:
        #   - background colors for logo
        #   - different aspect ratios (not high priority)
        
        # input image:
        #   - background colors for logo
        #     - sample RGBA with opacity
        #     - get average color, get complementary background and sample something around it
        #     - sample light/dark backgrounds
        #   - rotate logo with 0.5 prob (-45,45) angles
        #   - resizing the logo 
        #   - pasting the logo
        #   - crop logo
        #   - gaussian noise, saturation, brightness, etc
        self.class_idx = (self.class_idx + 1) % len(self.class_images)
        class_id = idx % len(self.class_images)
        image_id = idx 
        
        reference_image = self._get_image(image_id, "reference")
        class_img = self._get_image(class_id, "class")
        class_img_to_paste = copy.deepcopy(class_img)
        
        class_img_to_paste = self._add_colored_background(class_img_to_paste)
        class_img = self._add_colored_background(class_img)
  

        w_ref, h_ref = reference_image.size
        w_class, h_class = class_img_to_paste.size
        ratio = random.uniform(0.075, 0.55)
        
        h_new, w_new = get_image_size_after_resize_preserving_aspect_ratio(h=h_class, w=w_class, target_size=int(ratio * np.sqrt(w_ref * h_ref)))
        class_img_to_paste = class_img_to_paste.resize((w_new, h_new))

        angle = self._sample_angle()
        class_img_to_paste = class_img_to_paste.rotate(angle, expand=1)
        
        x_corner, y_corner = self._sample_object_placement((w_ref, h_ref), (w_new, h_new))
        reference_image.paste(class_img_to_paste, (x_corner, y_corner), class_img_to_paste)
        
        reference_image = self._random_distort(reference_image)
        
        class_img = class_img.convert("RGB")
        reference_image = reference_image.convert("RGB")
        
        rad = np.radians(angle)
        x_left = max(0, x_corner)
        x_right = min(x_corner + np.abs(w_new * np.cos(rad)) + np.abs(h_new * np.sin(rad)), w_ref)
        y_top = max(0, y_corner)
        y_bottom = min(y_corner + np.abs(h_new * np.cos(rad)) + np.abs(w_new * np.sin(rad)), h_ref)
        bbox = torch.tensor([[x_left, y_top, x_right, y_bottom]])
        
        
        
        # decide on image level data augmentation
        vflip = random.random() < 0.5 if self.data_augmentation.batch_random_vflip else False
        hflip = random.random() < 0.5 if self.data_augmentation.batch_random_hflip else False
        
        # prepare class images
        num_classes = 1
        logo_th = self._transform_image_gt(class_img, hflip=hflip, vflip=vflip)

        # get the image sizes after resize in self._transform_image_gt, format - width, height
        logo_size = FeatureMapSize(img=class_img)
       
        # get annotation
        fm_size = FeatureMapSize(img=reference_image)
        boxes = BoxList(bbox, image_size=fm_size, mode="xyxy")
        boxes.add_field("labels", torch.tensor([class_id]))
        boxes.add_field("labels_original", torch.tensor([class_id]))

        # convert global indices to local
        self.update_box_labels_to_local(boxes, [class_id])

        # prepare image and boxes: convert image to tensor, data augmentation: some boxes might be cut off the image
        image_mined_data = None 

        reference_image_th, boxes, mask_cutoff_boxes, mask_difficult_boxes, box_inverse_transform = \
                 self._transform_image(copy.deepcopy(reference_image), boxes, hflip=hflip, vflip=vflip, mined_data=image_mined_data)
      
        img_size = FeatureMapSize(img=reference_image_th)
        
        boxes.get_field("labels")[mask_cutoff_boxes] = -2
        loc_targets, class_targets = box_coder.encode(boxes, img_size, num_classes)
        
        return reference_image_th, logo_th, loc_targets, class_targets, class_id, logo_size, \
               box_inverse_transform, boxes, img_size 
    
    def _transform_image_gt(self, img, do_augmentation=True, hflip=False, vflip=False, do_resize=True):
        img, _ = transforms_boxes.transpose(img, hflip=hflip, vflip=vflip, boxes=None, transform_list=None)

        if do_augmentation:
            # color distortion
            img = self.data_augmentation.random_distort(img)
            # random crop
            img = self.data_augmentation.random_crop_label_image(img)

        # resize image
        if do_resize:
            random_interpolation = self.data_augmentation.random_interpolation if do_augmentation else False

            # get the new size - while preserving aspect ratio
            size_old = FeatureMapSize(img=img)
            h, w = get_image_size_after_resize_preserving_aspect_ratio(h=size_old.h, w=size_old.w, target_size=240)
            size_new = FeatureMapSize(w=w, h=h)

            img, _  = transforms_boxes.resize(img, target_size=size_new, random_interpolation=random_interpolation)

        transforms_th = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = transforms_th(img)

        return img

    def convert_label_ids_global_to_local(self, label_ids_global, class_ids):
        label_ids_local = [] # local indices w.r.t. batch_class_images
        if label_ids_global is not None:
            for label_id in label_ids_global:
                label_id = label_id#.item()
                label_ids_local.append( class_ids.index(label_id) if label_id in class_ids else -1 )
        label_ids_local = torch.tensor(label_ids_local, dtype=torch.long)
        return label_ids_local


    def update_box_labels_to_local(self, boxes, class_ids):
        label_ids_global = boxes.get_field("labels")
        label_ids_local = self.convert_label_ids_global_to_local([label_ids_global], class_ids)
        boxes.add_field("labels", label_ids_local)

    def _transform_image(self, img, boxes=None, do_augmentation=True, hflip=False, vflip=False, mined_data=None):
        img_pyramid, boxes_pyramid, mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform = \
                self._transform_image_to_pyramid(img, boxes=boxes,
                                                 do_augmentation=do_augmentation, hflip=hflip, vflip=vflip,
                                                 pyramid_scales=(1,), mined_data=mined_data) 

        return img_pyramid[0], boxes_pyramid[0], mask_cutoff_boxes, mask_difficult_boxes, pyramid_box_inverse_transform[0]

    def _transform_image_to_pyramid(self, img, boxes=None,
                                          do_augmentation=True, hflip=False, vflip=False,
                                          pyramid_scales=(1,),
                                          mined_data=None ):
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

        random_interpolation = self.data_augmentation.random_interpolation
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
      
 def collate_fn(data):
    reference_images, logos, loc_targets, class_targets, class_ids, logo_sizes, \
        box_inverse_transform, boxes, img_size  = zip(*data)
    reference_images_th = torch.stack([ref.unsqueeze(0) for ref in reference_images], dim=0)
    loc_targets_th = torch.stack([loc.unsqueeze(0) for loc in loc_targets], dim=0)
    class_targets_th = torch.stack([target.unsqueeze(0) for target in class_targets], dim=0)
    return reference_images_th, list(logos), loc_targets_th, class_targets_th, list(class_ids), list(logo_sizes), list(box_inverse_transform), list(boxes), list(img_size)

  
