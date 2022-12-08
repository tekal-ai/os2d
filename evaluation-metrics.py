import torch
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
import torch.nn as nn
import torchvision.transforms as transforms

# detections_path should be the path to a pytorch file of a dict with keys:
#    - 'image_ids': list of length num_images of ints
#    - 'boxes_xyxy': list of of length num_images of tensors of shape num_predicted_detections x 4
#    - 'labels': list of of length num_images of tensors of shape num_predicted_detections
#    - 'scores': list of of length num_images of tensors of shape num_predicted_detections
#    - 'gt_boxes_xyxy': list of of length num_images of tensors of shape num_gt_detections x 4
#    - 'gt_labels': list of of length num_images of tensors of shape num_gt_detections
# images_path and classes_path should be the paths to the input images and class images. if class 
#    images is None, gets the class images by cropping the input image using the ground truth annotations
# ann_path should be the path to annotations in the format described here:
#     https://www.notion.so/tekalai/Keymakr-Proposal-930cbd6315cc42069b7a2c4c49d3d706

#images_path = 'data/LogosInTheWild-v2/cleaned-data/voc_format'
images_path = "../../data/KEY-950/assets"
#images_path = "../../data/KEY-100/images"
#images_path = '../../data/LigiLog-100/src/images'
#classes_path = 'data/LogosInTheWild-v2/cleaned-data/brandROIs'
#ann_path = 'data/LogosInTheWild-v2/cleaned-data/annotations.csv'
ann_path = "../../data/KEY-950/annotations_cleaned.csv"
#ann_path = "../../data/KEY-100/annotations.csv"
#ann_path = '../../data/LigiLog-100/classes/industry-benchmark.csv'
#detections_path = 'new-os2d/os2d/detections/litw_detections_litw_30k.pth'
#detections_path = "/home/user/Documents/memorable/detections.pth"
#detections_path = "/home/user/Documents/memorable/sam-os2d/detections_ligilog.pth"
#detections_path = "key100_detections.pth"
#detections_path = "ligilog100_detections_2.pth"
#detections_path = "key950_detections_2.pth"
detections_path = "../../data/KEY-950/os2d_detections.pth"

def get_iou(bb1, bb2, coords_type='cxcywh'):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters
    ----------
    bb1 : (cx,cy,w,h)
    bb2 : (cx,cy,w,h)
    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = np.array(bb1)
    bb2 = np.array(bb2)

    if coords_type == 'cxcywh':
        x1_left = bb1[0] - bb1[2]/2
        x1_right = bb1[0] + bb1[2]/2
        y1_top = bb1[1] - bb1[3]/2
        y1_bottom = bb1[1] + bb1[3]/2

        x2_left = bb2[0] - bb2[2]/2
        x2_right = bb2[0] + bb2[2]/2
        y2_top = bb2[1] - bb2[3]/2
        y2_bottom = bb2[1] + bb2[3]/2
    elif coords_type == 'xyxy':
        x1_left = bb1[0]
        x1_right = bb1[2]
        y1_top = bb1[1]
        y1_bottom = bb1[3]

        x2_left = bb2[0]
        x2_right = bb2[2]
        y2_top = bb2[1]
        y2_bottom = bb2[3] 

    assert x1_left <= x1_right
    assert y1_top <= y1_bottom
    assert x2_left <= x2_right
    assert y2_top <= y2_bottom

    x_left = max(x1_left, x2_left)
    y_top = max(y1_top, y2_top)
    x_right = min(x1_right, x2_right)
    y_bottom = min(y1_bottom, y2_bottom)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (x1_right - x1_left) * (y1_bottom - y1_top)
    bb2_area = (x2_right - x2_left) * (y2_bottom - y2_top)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou


def get_batch_iou(bb1, bb2, coords_type='cxcywh'):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : shape (batch_size, 4), coordinates of type cx,cy,w,h
    bb2 : shape (batch_size, 4), coordinates of type cx,cy,w,h
    """
    bb1 = np.array(bb1)
    bb2 = np.array(bb2)

    if (len(bb1.shape) == 1):
        bb1 = bb1[np.newaxis, :]
    if (len(bb2.shape) == 1):
        bb2 = bb2[np.newaxis, :]

    if coords_type == 'cxcywh':
        x1_left = bb1[:, 0] - bb1[:, 2]/2
        x1_right = bb1[:, 0] + bb1[:, 2]/2
        y1_top = bb1[:, 1] - bb1[:, 3]/2
        y1_bottom = bb1[:, 1] + bb1[:, 3]/2

        x2_left = bb2[:, 0] - bb2[:, 2]/2
        x2_right = bb2[:, 0] + bb2[:, 2]/2
        y2_top = bb2[:, 1] - bb2[:, 3]/2
        y2_bottom = bb2[:, 1] + bb2[:, 3]/2
    elif coords_type == 'xyxy':
        x1_left = bb1[:, 0]
        x1_right = bb1[:, 2]
        y1_top = bb1[:, 1]
        y1_bottom = bb1[:, 3]

        x2_left = bb2[:, 0]
        x2_right = bb2[:, 2]
        y2_top = bb2[:, 1]
        y2_bottom = bb2[:, 3]  

    x_left = np.maximum(x1_left[:,np.newaxis], x2_left[np.newaxis,:])
    y_top = np.maximum(y1_top[:,np.newaxis], y2_top[np.newaxis,:])
    x_right = np.minimum(x1_right[:,np.newaxis], x2_right[np.newaxis,:])
    y_bottom = np.minimum(y1_bottom[:,np.newaxis], y2_bottom[np.newaxis,:])

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (x1_right - x1_left) * (y1_bottom - y1_top)
    bb2_area = (x2_right - x2_left) * (y2_bottom - y2_top)

    bb1_area = bb1_area[:,np.newaxis]
    bb2_area = bb2_area[np.newaxis,:]

    iou = intersection_area / (bb1_area + bb2_area - intersection_area)
    iou[iou < 0.0] = 0.0

    return iou

def get_top_boxes(scores, boxes_xyxy, n=5, threshold=1.0):
    ious = get_batch_iou(boxes_xyxy, boxes_xyxy, coords_type='xyxy')
    indices_ordered = np.argsort(scores)
    ious_ordered = ious[indices_ordered]

    top_boxes = []
    mask = np.ones_like(indices_ordered, dtype=bool)
    for i in range(n):
        new_idx = -1
        for idx in indices_ordered:
            if mask[idx.item()]:
                new_idx = idx.item()
                break
        if new_idx < 0:
            break
        top_boxes.append(new_idx)
        mask = mask & (ious[new_idx] < threshold)
    return top_boxes


def get_top_candidate_idx(detections, image_idx, imageid):
    scores = detections['scores'][image_idx]
    idx = np.argmax(scores)
    return idx


if __name__ == "__main__":
    fig, ax = plt.subplots()
    detections = torch.load(detections_path)
    #iou_thres = 0.5
    iou_thres_ls = [0.9, 0.8, 0.7, 0.6, 0.5]
    #confidence_thres = 0.75
    confidence_thres_ls = [0.9, 0.85, 0.8, 0.75, 0.7, 0.675, 0.65, 0.6375, 0.625, 0.6125, 0.6, 0.5875, 0.575, 0.5375, 0.55, 0.525, 0.5,
                           0.45, 0.35, 0.0]

    num_images = len(detections['scores'])
    import pandas as pd
    df = pd.read_csv(ann_path)
    print(df.columns)
    for iou_thres in iou_thres_ls:
        print('iou thres', iou_thres)

        recalls = []
        precisions = []
        for confidence_thres in confidence_thres_ls:
            ious = []
            tp = 0
            fn = 0
            fp = 0
            k = 0
            for i, imageid in enumerate(detections['image_ids']):
                #idx = get_top_candidate_idx(detections, i, imageid)
                #top_box = detections['boxes_xyxy'][i][idx]
                image_df = df[df["imageid"] == imageid]
                mask = detections['scores'][i] > confidence_thres
                img = Image.open(f"{images_path}/{imageid}.jpg")
                w, h = img.size
                gt_boxes = torch.tensor(np.array(image_df[['lx','ty', 'rx', 'by']]))
                boxes = detections['boxes_xyxy'][i][mask]
                detection_of_boxes = []
                ious_of_boxes = []
                try:
                    if boxes == torch.tensor([]):
                        boxes = torch.tensor([float('nan'), float('nan'), float('nan'), float('nan')])
                except:
                    pass
                for box in boxes:
                    ious_i = []
                    for gt_box in gt_boxes:
                        iou = get_iou(box.cpu(), gt_box.cpu(), coords_type='xyxy')
                        ious_i.append(iou)
                    if ious_i:
                        detection_of_boxes.append(np.argmax(ious_i))
                        ious_of_boxes.append(np.max(ious_i))
                    else:
                        detection_of_boxes.append(None)
                        ious_of_boxes.append(None)
                detection_of_boxes = np.array(detection_of_boxes)
                ious_of_boxes = np.array(ious_of_boxes)
                for j, gt_box in enumerate(gt_boxes):
                    boxes_of_gtbox = ious_of_boxes[detection_of_boxes == j]
                    k = k + 1
                    tp = tp + np.sum(boxes_of_gtbox > iou_thres)
                    fp = fp + np.sum(boxes_of_gtbox < iou_thres)
                    if len(boxes_of_gtbox) > 0:
                        ious.append(np.max(boxes_of_gtbox))
                        if np.max(boxes_of_gtbox) < iou_thres:
                            fn = fn + 1
                    else:
                        ious.append(0.0)
                        fn = fn + 1

            print('confidence thres', confidence_thres)
            print('mean iou', np.mean(ious))
            print('recall', tp / (tp + fn))
            recalls.append(tp / (tp + fn))
            print('precision', tp / (tp + fp))
            precisions.append(tp / (tp + fp))
        #print(recalls)
        #print(precisions)

        # create precision recall curve
        ax.plot(recalls, precisions, label=iou_thres)

    # add axis labels to plot
    ax.set_title('KEY-950 with os2d')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.legend(loc="upper left")

    # display plot
    #plt.show()
    plt.savefig("../../data/KEY-950/os2d_eval.png")
