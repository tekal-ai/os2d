import os
import argparse

import torch
from torchvision.utils import save_image

from os2d.modeling.model import build_os2d_from_config

from os2d.data.dataloader import build_eval_dataloaders_from_cfg, build_train_dataloader_from_config
from os2d.engine.train import trainval_loop
from os2d.engine.evaluate import evaluate
from os2d.utils import set_random_seed, get_trainable_parameters, mkdir, save_config, setup_logger, get_data_path, read_image, get_image_size_after_resize_preserving_aspect_ratio
from os2d.engine.optimization import create_optimizer
from os2d.config import cfg


def parse_opts():
    parser = argparse.ArgumentParser(description="Training and evaluation of the OS2D model")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    return cfg, args.config_file


def init_logger(cfg, config_file):
    output_dir = cfg.output.path
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("OS2D", output_dir if cfg.output.save_log_to_file else None)

    if config_file:
        logger.info("Loaded configuration file {}".format(config_file))
        with open(config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    else:
        logger.info("Config file was not provided")

    logger.info("Running with config:\n{}".format(cfg))

    # save config file only when training (to run multiple evaluations in the same folder)
    if output_dir and cfg.train.do_training:
        output_config_path = os.path.join(output_dir, "config.yml")
        logger.info("Saving config into: {}".format(output_config_path))
        # save overloaded model config in the output directory
        save_config(cfg, output_config_path)


def main():
    #cfg, config_file = parse_opts()
    #init_logger(cfg, config_file)

    logger = setup_logger("OS2D")
    cfg.init.model = "models/os2d_v2-train.pth"
    cfg.is_cuda = False
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
    
  #  input_image = read_image("data/products-internal/src/images/14.png")
  #  class_images = [read_image("data/products-internal/classes/images/2.jpg")]
  #  class_ids = [0]
    
  #  import torchvision.transforms as transforms

  #  h, w = get_image_size_after_resize_preserving_aspect_ratio(h=input_image.size[1],
  #                                                             w=input_image.size[0],
  #                                                             target_size=1500)
  #  input_image = input_image.resize((w, h))
  #  square_size = min(w, h)
  #  transform_image = transforms.Compose([
  #                    transforms.ToTensor(),
  #                    transforms.Resize((2*w, 2*h)),
  #                    transforms.CenterCrop(square_size),
  #                    transforms.Normalize(img_normalization["mean"], img_normalization["std"])
  #                    ])
  #  input_image_th = transform_image(input_image)
  #  save_image(input_image_th, 'input_image.png')
    
  #  input_image_th = input_image_th.unsqueeze(0)
  #  if cfg.is_cuda:
  #      input_image_th = input_image_th.cuda()

    ## Resize class image
  #  class_images_th = []
  #  for class_image in class_images:
  #      h, w = get_image_size_after_resize_preserving_aspect_ratio(h=class_image.size[1],
  #                                                              w=class_image.size[0],
  #                                                              target_size=cfg.model.class_image_size)
  #      class_image = class_image.resize((w, h))
  #      square_size = min(w, h)
  #      transform_image = transforms.Compose([
  #                    transforms.ToTensor(),
                      #transforms.Resize((3*w, 3*h)),
                      #transforms.CenterCrop(square_size),
  #                    transforms.Normalize(img_normalization["mean"], img_normalization["std"])
  #                    ])
  #      class_image_th = transform_image(class_image)
  #      save_image(class_image_th, 'class_image.png')
  #      if cfg.is_cuda:
  #          class_image_th = class_image_th.cuda()

  #      class_images_th.append(class_image_th)

  #  get_image_size_after_resize_preserving_aspect_ratio(h=class_image.size[1],
  #                                                             w=class_image.size[0],
  #                                                             target_size=cfg.model.class_image_size)
  #  with torch.no_grad():
  #      loc_prediction_batch, class_prediction_batch, _, fm_size, transform_corners_batch = net(images=input_image_th, class_images=class_images_th)
    #print(loc_prediction_batch.shape)
    #print(class_prediction_batch.shape)
    #print(fm_size)
    #print(transform_corners_batch.shape)

  #  from os2d.structures.feature_map import FeatureMapSize
  #  import matplotlib.pyplot as plt
  #  import  os2d.utils.visualization as visualizer

  #  image_loc_scores_pyramid = [loc_prediction_batch[0]]
  #  image_class_scores_pyramid = [class_prediction_batch[0]]
  #  img_size_pyramid = [FeatureMapSize(img=input_image_th)]
  #  transform_corners_pyramid = [transform_corners_batch[0]]
  #  boxes = box_coder.decode_pyramid(image_loc_scores_pyramid, image_class_scores_pyramid,
  #                                         img_size_pyramid, class_ids,
  #                                         nms_iou_threshold=cfg.eval.nms_iou_threshold,
  #                                         nms_score_threshold=cfg.eval.nms_score_threshold,
  #                                         transform_corners_pyramid=transform_corners_pyramid)
    # remove some fields to lighten visualization                                       
  #  boxes.remove_field("default_boxes")

    # Note that the system outputs the correaltions that lie in the [-1, 1] segment as the detection scores (the higher the better the detection).
  #  scores = boxes.get_field("scores")

  #  figsize = (8, 8)
  #  fig=plt.figure(figsize=figsize)
  #  columns = len(class_images)
  #  for i, class_image in enumerate(class_images):
  #      fig.add_subplot(1, columns, i + 1)
  #      plt.imshow(class_image)
  #      plt.axis('off')
  #  plt.savefig('logo.png')

  #  plt.rcParams["figure.figsize"] = figsize

  #  cfg.visualization.eval.max_detections = 4
  #  cfg.visualization.eval.score_threshold = float("-inf")
  #  fig = visualizer.show_detections(boxes, input_image,
  #                          cfg.visualization.eval, )
    
  #  fig.savefig("detections15.jpg")
    
    # Optimizer
    parameters = get_trainable_parameters(net)
    optimizer = create_optimizer(parameters, cfg.train.optim, optimizer_state)
    #cfg.train.dataset_name = 'logodet-3k'

    # load the dataset
    data_path = get_data_path()
    print(data_path)
    data_path = 'data'
    dataloader_train, datasets_train_for_eval = build_train_dataloader_from_config(cfg, box_coder, img_normalization,
                                                                              data_path=data_path)

    cfg.eval.dataset_names = ["industry-benchmark"]
    cfg.visualization.eval.path_to_save_detections = 'detections'
    #dataloaders_eval = build_eval_dataloaders_from_cfg(cfg, box_coder, img_normalization,
    #                                                   datasets_for_eval=[],
    #                                                   data_path=data_path)

    # start training (validation is inside)
    trainval_loop(dataloader_train, net, cfg, criterion, optimizer, dataloaders_eval=[])#dataloaders_eval)
    #evaluate(dataloaders_eval[0], net, cfg, criterion, print_per_class_results=True)
if __name__ == "__main__":
    main()
