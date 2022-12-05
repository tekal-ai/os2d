from synthetic_agumentations_dataloader_2 import SyntheticAugmentationsDataset
from torch.utils.data import DataLoader
from os2d.config import cfg
from os2d.utils import set_random_seed
from os2d.modeling.model import build_os2d_from_config
import torch

def collate_fn(data):
    reference_image, class_img, image_id, class_id, bbox = zip(*data)
    reference_image = list(reference_image)
    class_img = list(class_img)
    image_id = list(image_id)
    class_id = list(class_id)
    bbox = torch.stack(bbox, dim=0)
    return reference_image, class_img, image_id, class_id, bbox

reference_images_path = "../../data/os2d-v3/client-assets-val"
logos_path = "../../data/os2d-v3/client-logos-transparent"

set_random_seed(cfg.random_seed, cfg.is_cuda)

_, box_coder, _, _, _ = build_os2d_from_config(cfg)

dataset = SyntheticAugmentationsDataset(reference_images_path, logos_path, box_coder)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

annotations_dict = {
    "imageid" : [],
    "classid" : [],
    "gtbboxid": [],
    "lx": [],
    "ty": [],
    "rx": [],
    "by": []
}

for i, batch in enumerate(dataloader):
    print(i)
    reference_image, class_img, image_id, class_id, bbox = batch
    w, h = reference_image[0].size
    
    lx = bbox[0][0][0].item() / w
    ty = bbox[0][0][1].item() / h
    rx = bbox[0][0][2].item() / w
    by = bbox[0][0][3].item() / h

    annotations_dict["imageid"].append(image_id[0])
    annotations_dict["classid"].append(class_id[0])
    annotations_dict["gtbboxid"].append(i)
    annotations_dict["lx"].append(lx)
    annotations_dict["ty"].append(ty)
    annotations_dict["rx"].append(rx)
    annotations_dict["by"].append(by)
    image_id0 = image_id[0]
    class_id0 = class_id[0]
    reference_image[0].save(f"../../data/os2d-v3/eval-dataset/src/{image_id0}.jpg")
    class_img[0].save(f"../../data/os2d-v3/eval-dataset/classes/{class_id0}.jpg")
    

import pandas as pd

pd.DataFrame(annotations_dict).to_csv(f"../../data/os2d-v3/eval-dataset/classes/annotations.csv")
